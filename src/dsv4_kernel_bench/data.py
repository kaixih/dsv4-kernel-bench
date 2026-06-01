from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

SelectionPattern = Literal["random", "swa", "csa", "hca"]


@dataclass(frozen=True)
class SparseAttentionInputs:
    q: torch.Tensor
    kv: torch.Tensor
    attn_sink: torch.Tensor
    topk_idxs: torch.Tensor
    sm_scale: float | None
    metadata: dict[str, int | float | str]


def local_window_indices(
    *,
    batch: int,
    seqlen_q: int,
    raw_seqlen_kv: int,
    window_size: int,
    query_start: int,
    device: str | torch.device,
) -> torch.Tensor:
    """Return causal sliding-window indices into the raw-KV section."""

    base = query_start + torch.arange(seqlen_q, device=device).unsqueeze(1)
    offsets = torch.arange(window_size, device=device)
    matrix = (base - window_size + 1).clamp(min=0) + offsets
    invalid = (matrix > base) | (matrix >= raw_seqlen_kv)
    matrix = torch.where(invalid, -1, matrix)
    return matrix.unsqueeze(0).expand(batch, -1, -1).to(torch.int32).contiguous()


def newest_visible_compressed_indices(
    *,
    batch: int,
    seqlen_q: int,
    raw_seqlen_kv: int,
    n_compressed: int,
    compress_ratio: int,
    compressed_topk: int,
    query_start: int,
    device: str | torch.device,
) -> torch.Tensor:
    """Return latest visible compressed ids, shifted into the logical KV pool."""

    if compressed_topk == 0:
        return torch.empty(batch, seqlen_q, 0, device=device, dtype=torch.int32)

    positions = query_start + torch.arange(1, seqlen_q + 1, device=device).unsqueeze(1)
    visible = (positions // compress_ratio).clamp(max=n_compressed)
    slots = torch.arange(compressed_topk, device=device).unsqueeze(0)
    start = (visible - compressed_topk).clamp(min=0)
    ids = start + slots
    valid = ids < visible
    ids = torch.where(valid, ids + raw_seqlen_kv, -1)
    return ids.unsqueeze(0).expand(batch, -1, -1).to(torch.int32).contiguous()


def all_visible_compressed_indices(
    *,
    batch: int,
    seqlen_q: int,
    raw_seqlen_kv: int,
    n_compressed: int,
    compress_ratio: int,
    query_start: int,
    device: str | torch.device,
) -> torch.Tensor:
    """Return all causally visible compressed ids, shifted into the KV pool."""

    if n_compressed == 0:
        return torch.empty(batch, seqlen_q, 0, device=device, dtype=torch.int32)

    matrix = torch.arange(n_compressed, device=device).repeat(seqlen_q, 1)
    visible = (query_start + torch.arange(1, seqlen_q + 1, device=device).unsqueeze(1)) // compress_ratio
    matrix = torch.where(matrix < visible, matrix + raw_seqlen_kv, -1)
    return matrix.unsqueeze(0).expand(batch, -1, -1).to(torch.int32).contiguous()


def make_sparse_attention_inputs(
    *,
    batch: int = 1,
    seqlen_q: int = 16,
    seqlen_kv: int = 32,
    heads: int = 2,
    dim: int = 64,
    topk: int = 8,
    invalid_fraction: float = 0.0,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cuda",
    seed: int = 0,
    sm_scale: float | None = None,
    selection_pattern: SelectionPattern = "random",
    raw_seqlen_kv: int | None = None,
    window_size: int | None = None,
    compressed_topk: int | None = None,
    compress_ratio: int | None = None,
    query_start: int = 0,
) -> SparseAttentionInputs:
    """Create synthetic canonical sparse attention inputs."""

    if not 0.0 <= invalid_fraction <= 1.0:
        raise ValueError("invalid_fraction must be in [0, 1]")
    if topk < 0:
        raise ValueError("topk must be non-negative")

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    raw_len = seqlen_kv if raw_seqlen_kv is None else raw_seqlen_kv
    if raw_len <= 0:
        raise ValueError("raw_seqlen_kv must be positive")
    if query_start < 0:
        raise ValueError("query_start must be non-negative")

    metadata: dict[str, int | float | str] = {
        "selection_pattern": selection_pattern,
        "raw_seqlen_kv": raw_len,
        "query_start": query_start,
    }

    if selection_pattern == "random":
        total_seqlen_kv = seqlen_kv
        topk_idxs = torch.randint(
            0,
            total_seqlen_kv,
            (batch, seqlen_q, topk),
            device=device,
            dtype=torch.int32,
            generator=gen,
        )
        metadata.update(
            {
                "window_size": 0,
                "compress_ratio": 0,
                "n_compressed": 0,
                "compressed_topk": 0,
            }
        )
    elif selection_pattern == "swa":
        window = topk if window_size is None else window_size
        if window <= 0:
            raise ValueError("window_size/topk must be positive for SWA")
        total_seqlen_kv = raw_len
        topk_idxs = local_window_indices(
            batch=batch,
            seqlen_q=seqlen_q,
            raw_seqlen_kv=raw_len,
            window_size=window,
            query_start=query_start,
            device=device,
        )
        metadata.update(
            {
                "window_size": window,
                "compress_ratio": 0,
                "n_compressed": 0,
                "compressed_topk": 0,
            }
        )
    elif selection_pattern in ("csa", "hca"):
        ratio = compress_ratio or (4 if selection_pattern == "csa" else 128)
        if ratio <= 1:
            raise ValueError("compress_ratio must be greater than 1")
        window = min(4, topk) if window_size is None else window_size
        if window < 0:
            raise ValueError("window_size must be non-negative")
        n_compressed = raw_len // ratio
        total_seqlen_kv = raw_len + n_compressed
        window_idxs = local_window_indices(
            batch=batch,
            seqlen_q=seqlen_q,
            raw_seqlen_kv=raw_len,
            window_size=window,
            query_start=query_start,
            device=device,
        )
        if selection_pattern == "csa":
            comp_topk = max(0, topk - window) if compressed_topk is None else compressed_topk
            comp_idxs = newest_visible_compressed_indices(
                batch=batch,
                seqlen_q=seqlen_q,
                raw_seqlen_kv=raw_len,
                n_compressed=n_compressed,
                compress_ratio=ratio,
                compressed_topk=comp_topk,
                query_start=query_start,
                device=device,
            )
        else:
            comp_topk = n_compressed
            comp_idxs = all_visible_compressed_indices(
                batch=batch,
                seqlen_q=seqlen_q,
                raw_seqlen_kv=raw_len,
                n_compressed=n_compressed,
                compress_ratio=ratio,
                query_start=query_start,
                device=device,
            )
        topk_idxs = torch.cat([window_idxs, comp_idxs], dim=-1)
        metadata.update(
            {
                "window_size": window,
                "compress_ratio": ratio,
                "n_compressed": n_compressed,
                "compressed_topk": comp_topk,
            }
        )
    else:
        raise ValueError(f"unknown selection_pattern: {selection_pattern}")

    q = torch.randn(batch, seqlen_q, heads, dim, device=device, dtype=dtype, generator=gen)
    kv = torch.randn(batch, total_seqlen_kv, dim, device=device, dtype=dtype, generator=gen)
    attn_sink = torch.zeros(heads, device=device, dtype=torch.float32)

    if invalid_fraction:
        mask = torch.rand(topk_idxs.shape, device=device, generator=gen) < invalid_fraction
        topk_idxs = topk_idxs.masked_fill(mask, -1)

    metadata["total_seqlen_kv"] = total_seqlen_kv
    metadata["total_topk"] = topk_idxs.shape[-1]

    return SparseAttentionInputs(
        q=q.contiguous(),
        kv=kv.contiguous(),
        attn_sink=attn_sink.contiguous(),
        topk_idxs=topk_idxs.contiguous(),
        sm_scale=sm_scale,
        metadata=metadata,
    )
