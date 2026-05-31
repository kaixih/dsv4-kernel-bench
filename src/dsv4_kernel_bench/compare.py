from __future__ import annotations

from dataclasses import dataclass

import torch

from dsv4_kernel_bench.backends import SparseAttentionBackend, run_sparse_attention_backend
from dsv4_kernel_bench.data import SparseAttentionInputs


@dataclass(frozen=True)
class RunResult:
    output: torch.Tensor
    dq: torch.Tensor | None
    dkv: torch.Tensor | None
    d_attn_sink: torch.Tensor | None


@dataclass(frozen=True)
class DiffResult:
    max_abs: float
    mean_abs: float
    rel_l2: float


def tensor_diff(a: torch.Tensor, b: torch.Tensor) -> DiffResult:
    af = a.detach().float()
    bf = b.detach().float()
    diff = (af - bf).abs()
    denom = torch.linalg.vector_norm(af).clamp_min(1e-12)
    return DiffResult(
        max_abs=float(diff.max().item()),
        mean_abs=float(diff.mean().item()),
        rel_l2=float((torch.linalg.vector_norm(af - bf) / denom).item()),
    )


def run_forward_backward(
    backend: SparseAttentionBackend,
    inputs: SparseAttentionInputs,
    *,
    backward: bool = True,
) -> RunResult:
    q = inputs.q.detach().clone().requires_grad_(backward)
    kv = inputs.kv.detach().clone().requires_grad_(backward)
    attn_sink = inputs.attn_sink.detach().clone().requires_grad_(backward)
    out = run_sparse_attention_backend(
        backend,
        q,
        kv,
        attn_sink,
        inputs.topk_idxs,
        inputs.sm_scale,
    )

    if backward:
        out.float().sum().backward()

    return RunResult(
        output=out.detach(),
        dq=None if q.grad is None else q.grad.detach(),
        dkv=None if kv.grad is None else kv.grad.detach(),
        d_attn_sink=None if attn_sink.grad is None else attn_sink.grad.detach(),
    )
