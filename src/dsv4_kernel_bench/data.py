from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SparseAttentionInputs:
    q: torch.Tensor
    kv: torch.Tensor
    attn_sink: torch.Tensor
    topk_idxs: torch.Tensor
    sm_scale: float | None


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
) -> SparseAttentionInputs:
    """Create synthetic canonical sparse attention inputs."""

    if not 0.0 <= invalid_fraction <= 1.0:
        raise ValueError("invalid_fraction must be in [0, 1]")
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    q = torch.randn(batch, seqlen_q, heads, dim, device=device, dtype=dtype, generator=gen)
    kv = torch.randn(batch, seqlen_kv, dim, device=device, dtype=dtype, generator=gen)
    attn_sink = torch.zeros(heads, device=device, dtype=torch.float32)
    topk_idxs = torch.randint(
        0,
        seqlen_kv,
        (batch, seqlen_q, topk),
        device=device,
        dtype=torch.int32,
        generator=gen,
    )

    if invalid_fraction:
        mask = torch.rand(batch, seqlen_q, topk, device=device, generator=gen) < invalid_fraction
        topk_idxs = topk_idxs.masked_fill(mask, -1)

    return SparseAttentionInputs(
        q=q.contiguous(),
        kv=kv.contiguous(),
        attn_sink=attn_sink.contiguous(),
        topk_idxs=topk_idxs.contiguous(),
        sm_scale=sm_scale,
    )
