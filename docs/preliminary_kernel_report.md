# Preliminary DSv4 Selected-KV Kernel Report

Date: 2026-05-31

## Scope

This report covers the blue module in `kernel_surface_sparse_attention.md`:

```text
Q + logical KV pool + selected KV ids -> attention output
```

It intentionally does not benchmark the real compressor, real indexer/top-k, or
full Megatron layer integration yet.

## Backend Difference

| Area | Miles Megatron path | NVIDIA Megatron path | Current harness boundary |
| --- | --- | --- | --- |
| Selected-KV attention | TileLang `sparse_attn_tilelang(q, kv, attn_sink, topk_idxs, sm_scale)` with custom fwd/bwd | Megatron `dsa_sparse_attn`, backed by FlashMLA/cuDNN DSA pieces | Same canonical tensors: `q`, logical `kv`, selected ids |
| CSA indexer/top-k | TileLang indexer path in Miles/NeMo vendored kernels | cuDNN DSA indexer wrappers plus top-k selection | Not measured yet; synthetic selected ids stand in for its output |
| CSA/HCA compressor | Miles DeepSeek-V4 compressor path, TileLang-optimized variants where present | Megatron compressor path; SGLang/Miles discuss Flash Compressor-style fused variants | Not measured yet; synthetic compressed KV entries stand in for its output |
| mHC | Miles/TileKernels mHC kernels | NVIDIA Megatron/TransformerEngine mHC fusion work | Separate future kernel track |

Sources:

- DeepSeek-V4 Flash config: `n_heads=64`, `head_dim=512`, `window_size=128`,
  `index_topk=512`, and `compress_ratios=[0,0,4,128,...,4,0]`:
  https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/raw/main/inference/config.json
- HF DSv4 model docs describe SWA, CSA, HCA, cache objects, and default config:
  https://huggingface.co/docs/transformers/model_doc/deepseek_v4
- NeMo AutoModel documents the vendored Miles sparse attention API:
  https://docs.nvidia.com/nemo/automodel/nightly/apidocs/nemo_automodel/nemo_automodel.components.models.deepseek_v4.kernels.sparse_attention.html
- cuDNN DSA APIs cover NVIDIA's DSA indexer/sparse-attention family:
  https://docs.nvidia.com/deeplearning/cudnn/latest/fe-oss-apis/dsa.html
- SGLang/Miles write-up describes the fused hybrid attention call, ShadowRadix,
  Flash Compressor, and Lightning TopK:
  https://www.lmsys.org/blog/2026-04-25-deepseek-v4/

## Shape Rationale

The model-like dimensions for V4-Flash are:

```text
H = 64 query heads
D = 512 attention head dim
KV heads = 1 shared K=V head
SWA window = 128
CSA compression ratio = 4
CSA index_topk = 512
HCA compression ratio = 128
```

The selected-KV attention kernel sees:

```text
q:          [B, S_q, H, D]
kv:         [B, S_raw + S_comp, D]
topk_idxs:  [B, S_q, selected_count]
```

Recommended first shapes:

| Case | Purpose | Example |
| --- | --- | --- |
| tiny random | backend import/JIT smoke | `B=1, S_q=8, S_kv=16, H=2, D=64, TopK=8` |
| tiny SWA | local-window selected ids only | `B=1, S_q=8, S_raw=16, H=2, D=64, window=8` |
| tiny CSA | raw + C4 appended, synthetic top-k C4 ids | `B=1, S_q=8, S_raw=16, H=2, D=64, window=4, compressed_topk=4` |
| tiny HCA | raw + C128 appended, all-visible C128 ids | `B=1, S_q=256, S_raw=256, H=2, D=64, window=4` |
| Flash decode CSA | production-like selected count | `B=1, S_q=1, S_raw=2048/8192, H=64, D=512, window=128, compressed_topk=512` |
| Flash decode HCA | C128 all-visible branch | `B=1, S_q=1, S_raw=8192, H=64, D=512, window=128` |
| Flash prefill CSA/HCA | exercise multi-query rows | `B=1, S_q=256, S_raw=256, H=64, D=512` |

The decode-like cases should set:

```text
query_start = S_raw - S_q
```

That makes local-window ids point to the raw-KV tail instead of the beginning
of the sequence.

## Config-Driven Runs

Shapes live in JSON so the matrix can evolve without editing Python:

- `configs/blue_module_smoke.json`
- `configs/blue_module_flash_decode.json`
- `configs/blue_module_perf_forward.json`
- `configs/blue_module_perf_backward.json`

Run a matrix:

```bash
PYTHONPATH=$PWD/src python -m dsv4_kernel_bench.bench_matrix \
  --config configs/blue_module_smoke.json \
  --output-dir /tmp/dsv4-kernel-bench-runs/smoke
```

To compare NVIDIA, copy a config and change `defaults.backend` to
`nvidia_dsa`. The output `summary.json` keeps pass/skip/fail status for every
case, including exact backend import errors.

## Current Interpretation

For the blue module, SWA/CSA/HCA are not separate attention kernels in the
harness. They are different ways to construct `kv` and `topk_idxs`:

- SWA: `kv = raw KV`, selected ids are local-window raw ids.
- CSA: `kv = raw KV + synthetic C4 KV`, selected ids are local-window raw ids
  plus synthetic top-k C4 ids.
- HCA: `kv = raw KV + synthetic C128 KV`, selected ids are local-window raw ids
  plus all visible C128 ids.

This makes the first comparison narrow: it tests whether Miles TileLang and
NVIDIA DSA agree on the selected-KV attention consumer before we add real
compressor/indexer kernels.
