import pytest
import torch

from dsv4_kernel_bench.backends import BackendUnavailable, run_sparse_attention_backend
from dsv4_kernel_bench.compare import run_forward_backward, tensor_diff
from dsv4_kernel_bench.data import make_sparse_attention_inputs


def _requires_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for sparse attention backend tests")


def test_sparse_attention_surface_miles():
    _requires_cuda()
    inputs = make_sparse_attention_inputs(batch=1, seqlen_q=8, seqlen_kv=16, heads=2, dim=64, topk=8)
    try:
        out = run_sparse_attention_backend(
            "miles_tilelang",
            inputs.q,
            inputs.kv,
            inputs.attn_sink,
            inputs.topk_idxs,
            inputs.sm_scale,
        )
    except BackendUnavailable as exc:
        pytest.skip(str(exc))
    assert out.shape == inputs.q.shape
    assert torch.isfinite(out.float()).all()


def test_sparse_attention_invalid_topk_miles():
    _requires_cuda()
    inputs = make_sparse_attention_inputs(
        batch=1,
        seqlen_q=8,
        seqlen_kv=16,
        heads=2,
        dim=64,
        topk=8,
        invalid_fraction=0.25,
    )
    try:
        out = run_sparse_attention_backend(
            "miles_tilelang",
            inputs.q,
            inputs.kv,
            inputs.attn_sink,
            inputs.topk_idxs,
            inputs.sm_scale,
        )
    except BackendUnavailable as exc:
        pytest.skip(str(exc))
    assert torch.isfinite(out.float()).all()


def test_sparse_attention_forward_backward_parity_when_nvidia_available():
    _requires_cuda()
    inputs = make_sparse_attention_inputs(batch=1, seqlen_q=8, seqlen_kv=16, heads=2, dim=64, topk=8)

    try:
        ref = run_forward_backward("miles_tilelang", inputs)
    except BackendUnavailable as exc:
        pytest.skip(str(exc))

    try:
        cmp = run_forward_backward("nvidia_dsa", inputs)
    except BackendUnavailable as exc:
        pytest.skip(str(exc))

    out_diff = tensor_diff(ref.output, cmp.output)
    assert out_diff.rel_l2 < 5e-2

    for name, a, b in [
        ("dq", ref.dq, cmp.dq),
        ("dkv", ref.dkv, cmp.dkv),
        ("d_attn_sink", ref.d_attn_sink, cmp.d_attn_sink),
    ]:
        assert a is not None and b is not None, name
        diff = tensor_diff(a, b)
        assert diff.rel_l2 < 1e-1, (name, diff)
