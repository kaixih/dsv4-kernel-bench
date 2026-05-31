# DSv4 Sparse Attention Surface Notes

This harness starts with one kernel family: sparse MLA / DSA attention. Miles
TileLang is the reference path for DSv4 RL bring-up; NVIDIA DSA is the
comparison path.

## Terms

- Sparse MLA: sparse multi-latent attention style kernel used by the Miles
  DeepSeek-V4 plugin. It consumes query, single-head KV, attention sink, and
  top-k indices.
- DSA: NVIDIA/Megatron sparse attention wrapper around FlashMLA forward and
  cuDNN Frontend DSA helpers.
- Indexer top-k: the upstream selection step that decides which compressed KV
  entries each query attends to. v1 does not benchmark the indexer directly.
- HCA: Heavily/Highly Compressed Attention in DSv4 hybrid attention. In v1 it
  is background context only; the harness benchmarks the sparse attention
  consumer after compressed KV and top-k indices already exist.

## Canonical Harness Surface

```python
q: [B, S, H, D] bf16
kv: [B, S_kv, D] bf16
attn_sink: [H] fp32
topk_idxs: [B, S, TopK] int32  # -1 means invalid
sm_scale: float | None
```

The output is `out: [B, S, H, D]`.

## Miles TileLang Surface

Entrypoint:

```python
from miles_plugins.models.deepseek_v4.ops.kernel.tilelang_sparse_mla import sparse_attn_tilelang
```

Expected inputs match the canonical layout:

```python
sparse_attn_tilelang(q, kv, attn_sink, topk_idxs, sm_scale) -> out
```

Backward returns gradients for `q`, `kv`, and `attn_sink`; indices and scale do
not receive gradients.

## NVIDIA DSA Surface

Entrypoint:

```python
from megatron.core.transformer.experimental_attention_variant.dsa_kernels import dsa_sparse_attn
```

The DSA path expects flattened SB row order:

```python
q_flat: [S * B, H, D]
kv_flat: [S_kv * B, D]
topk_idxs_flat: [S * B, TopK]
```

For a local batch KV index `k` in row `(b, s)`, the global flat index is:

```python
global_k = k * B + b
row = s * B + b
```

Invalid `-1` entries remain `-1`.

## First Comparison Boundary

The v1 comparison intentionally excludes:

- compressor kernel parity
- indexer/top-k kernel parity
- mHC parity
- real DSv4-Flash production shapes

Those become follow-up targets after the sparse attention harness is stable.
