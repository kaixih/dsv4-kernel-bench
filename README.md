# DSv4 Kernel Bench

Small harness for comparing DeepSeek-V4 hybrid-attention kernel paths.

The first target is the selected-KV sparse-attention consumer:

- reference backend: Miles TileLang `sparse_attn_tilelang`
- comparison backend: NVIDIA Megatron `dsa_sparse_attn`

This is the kernel after the KV pool and per-query selected KV indices already
exist. For CSA, those selected ids include compressed-KV top-k; for SWA/HCA
they can be deterministic local/all-visible compressed ids. The harness uses
one canonical input layout and keeps backend-specific layout conversion inside
adapters.

## Quick Start

```bash
pip install -e .[test]
pytest -q
python -m dsv4_kernel_bench.bench --backend miles_tilelang --device cuda --output bench.json
```

If optional runtime dependencies are missing, backend tests skip with a clear
reason. The index conversion tests run without GPU dependencies.

Use `--selection-pattern` to choose the synthetic selected-id pattern:

```bash
python -m dsv4_kernel_bench.bench --backend miles_tilelang --selection-pattern swa
python -m dsv4_kernel_bench.bench --backend miles_tilelang --selection-pattern csa --window-size 4 --compressed-topk 4
python -m dsv4_kernel_bench.bench --backend miles_tilelang --selection-pattern hca --seqlen-q 256 --seqlen-kv 256 --window-size 4
```

## Canonical Selected-KV Attention Inputs

```python
q: [B, S, H, D] bf16
kv: [B, S_kv, D] bf16  # logical pool: raw KV plus optional compressed KV
attn_sink: [H] fp32
topk_idxs: [B, S, TopK] int32  # indices into kv; -1 means invalid
sm_scale: float | None
```

See `docs/kernel_surface_sparse_attention.md` for the surface notes and
`docs/b200_repro_runbook.md` for the B200 container reproduction flow.
