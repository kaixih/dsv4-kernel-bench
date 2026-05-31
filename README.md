# DSv4 Kernel Bench

Small harness for comparing DeepSeek-V4 sparse attention kernel paths.

The first target is sparse MLA / DSA attention:

- reference backend: Miles TileLang `sparse_attn_tilelang`
- comparison backend: NVIDIA Megatron `dsa_sparse_attn`

The harness uses one canonical input layout and keeps backend-specific layout
conversion inside adapters.

## Quick Start

```bash
pip install -e .[test]
pytest -q
python -m dsv4_kernel_bench.bench --backend miles_tilelang --device cuda --output bench.json
```

If optional runtime dependencies are missing, backend tests skip with a clear
reason. The index conversion tests run without GPU dependencies.

## Canonical Sparse Attention Inputs

```python
q: [B, S, H, D] bf16
kv: [B, S_kv, D] bf16
attn_sink: [H] fp32
topk_idxs: [B, S, TopK] int32  # -1 means invalid
sm_scale: float | None
```

See `docs/kernel_surface_sparse_attention.md` for the surface notes and
`docs/b200_repro_runbook.md` for the B200 container reproduction flow.
