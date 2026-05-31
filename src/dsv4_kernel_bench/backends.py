from __future__ import annotations

from typing import Literal

import torch

from dsv4_kernel_bench.indexing import (
    from_dsa_output_sbhd,
    local_topk_to_global_flat,
    to_dsa_kv_sbd,
    to_dsa_query_sbhd,
)

SparseAttentionBackend = Literal["miles_tilelang", "nvidia_dsa"]


class BackendUnavailable(RuntimeError):
    """Raised when an optional backend cannot be imported or initialized."""


def _first_tensor(result):
    return result[0] if isinstance(result, tuple) else result


def run_sparse_attention_backend(
    backend: SparseAttentionBackend,
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Run one sparse attention backend with canonical harness inputs."""

    if backend == "miles_tilelang":
        return run_miles_tilelang(q, kv, attn_sink, topk_idxs, sm_scale)
    if backend == "nvidia_dsa":
        return run_nvidia_dsa(q, kv, attn_sink, topk_idxs, sm_scale)
    raise ValueError(f"unknown sparse attention backend: {backend}")


def run_miles_tilelang(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Run Miles TileLang sparse MLA attention.

    Expected canonical surface:
      q: [B, S, H, D], kv: [B, S_kv, D], topk_idxs: [B, S, TopK].
    """

    try:
        from miles_plugins.models.deepseek_v4.ops.kernel.tilelang_sparse_mla import (
            sparse_attn_tilelang,
        )
    except Exception as exc:  # pragma: no cover - depends on remote image
        raise BackendUnavailable(f"Miles TileLang sparse attention unavailable: {exc}") from exc

    return sparse_attn_tilelang(
        q.contiguous(),
        kv.contiguous(),
        attn_sink.float().contiguous(),
        topk_idxs.to(torch.int32).contiguous(),
        sm_scale,
    )


def run_nvidia_dsa(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Run NVIDIA Megatron DSA sparse attention through a canonical adapter."""

    try:
        from megatron.core.transformer.experimental_attention_variant.dsa_kernels import (
            dsa_sparse_attn,
        )
    except Exception as exc:  # pragma: no cover - depends on remote image
        raise BackendUnavailable(f"NVIDIA Megatron DSA sparse attention unavailable: {exc}") from exc

    batch, _seqlen_q, heads, dim = q.shape
    seqlen_kv = kv.shape[1]
    q_sbhd = to_dsa_query_sbhd(q)
    kv_sbd = to_dsa_kv_sbd(kv)
    topk_flat = local_topk_to_global_flat(topk_idxs, seqlen_kv)
    softmax_scale = dim**-0.5 if sm_scale is None else sm_scale

    out_sbhd = _first_tensor(
        dsa_sparse_attn(
            q_sbhd,
            kv_sbd,
            attn_sink.float().contiguous(),
            topk_flat,
            softmax_scale,
        )
    )
    return from_dsa_output_sbhd(out_sbhd, batch, heads)
