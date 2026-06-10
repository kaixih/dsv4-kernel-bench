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

## Miles Kernel Source Links

NeMo AutoModel's vendored Miles docs attribute the DSv4 sparse-attention kernels
to `yueming-yuan/miles` commit `e561465d0b9bbf06188b7a5e2020dc7fd691f732`,
`deepseek-v4` branch. That public repo is a fork of `radixark/miles`.

Pinned source links:

- Ops directory:
  https://github.com/yueming-yuan/miles/tree/e561465d0b9bbf06188b7a5e2020dc7fd691f732/miles_plugins/models/deepseek_v4/ops
- Sparse-attention autograd wrapper:
  https://github.com/yueming-yuan/miles/blob/e561465d0b9bbf06188b7a5e2020dc7fd691f732/miles_plugins/models/deepseek_v4/ops/attention_core.py
- Sparse MQA TileLang forward:
  https://github.com/yueming-yuan/miles/blob/e561465d0b9bbf06188b7a5e2020dc7fd691f732/miles_plugins/models/deepseek_v4/ops/kernel/tilelang_sparse_mla_fwd.py
- Sparse MQA TileLang backward:
  https://github.com/yueming-yuan/miles/blob/e561465d0b9bbf06188b7a5e2020dc7fd691f732/miles_plugins/models/deepseek_v4/ops/kernel/tilelang_sparse_mla_bwd.py
- TileLang indexer wrapper:
  https://github.com/yueming-yuan/miles/blob/e561465d0b9bbf06188b7a5e2020dc7fd691f732/miles_plugins/models/deepseek_v4/ops/kernel/tilelang_indexer.py
- TileLang indexer forward:
  https://github.com/yueming-yuan/miles/blob/e561465d0b9bbf06188b7a5e2020dc7fd691f732/miles_plugins/models/deepseek_v4/ops/kernel/tilelang_indexer_fwd.py
- TileLang indexer backward:
  https://github.com/yueming-yuan/miles/blob/e561465d0b9bbf06188b7a5e2020dc7fd691f732/miles_plugins/models/deepseek_v4/ops/kernel/tilelang_indexer_bwd.py
- V4 indexer glue:
  https://github.com/yueming-yuan/miles/blob/e561465d0b9bbf06188b7a5e2020dc7fd691f732/miles_plugins/models/deepseek_v4/ops/v4_indexer.py
- Compressor module:
  https://github.com/yueming-yuan/miles/blob/e561465d0b9bbf06188b7a5e2020dc7fd691f732/miles_plugins/models/deepseek_v4/ops/compressor.py
- Hyper-connection / mHC integration:
  https://github.com/yueming-yuan/miles/blob/e561465d0b9bbf06188b7a5e2020dc7fd691f732/miles_plugins/models/deepseek_v4/ops/hyper_connection.py

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

## B200 Perf Snapshot

Run context:

```text
Node: umbriel-b200-044, 8x NVIDIA B200
Repo commit: 460e019d0b879c3d9a2329c1c56a743c81a8d21f
Container: radixark/miles:deepseek-v4
NVIDIA DSA backward dependency: nvidia-cudnn-frontend[cutedsl]==1.24.0
Run output: /tmp/dsv4-kernel-bench-runs/perf_compare_20260601-042600/perf_compare_summary.json
```

Important benchmark caveat: these results measure the selected-KV sparse
attention consumer with synthetic selected ids. They do not include real
compressor, real indexer scoring, or real top-k selection.

Forward-only perf uses no-grad and reuses the same synthetic tensors across
iterations. Backward perf clones per iteration so gradients do not accumulate.

| Mode | Case | Shape | Selection | Miles ms | NVIDIA DSA ms | NVIDIA speedup |
| --- | --- | --- | --- | ---: | ---: | ---: |
| forward | `swa_decode_2k` | `B=1,S=1,S_raw=2048,H=64,D=512,TopK=128` | SWA window 128 | 0.3830 | 0.2108 | 1.82x |
| forward | `csa_decode_2k` | `B=1,S=1,S_raw=2048,H=64,D=512,TopK=640` | raw window 128 + C4 topk 512 | 0.3704 | 0.2075 | 1.79x |
| forward | `hca_decode_8k` | `B=1,S=1,S_raw=8192,H=64,D=512,TopK=192` | raw window 128 + all visible C128 | 0.3751 | 0.2058 | 1.82x |
| forward | `csa_decode_8k` | `B=1,S=1,S_raw=8192,H=64,D=512,TopK=640` | raw window 128 + C4 topk 512 | 0.3834 | 0.2055 | 1.87x |
| forward | `csa_prefill_256` | `B=1,S=256,S_raw=256,H=64,D=512,TopK=640` | raw window 128 + C4 topk 512 | 0.3814 | 0.2209 | 1.73x |
| forward | `hca_prefill_256` | `B=1,S=256,S_raw=256,H=64,D=512,TopK=130` | raw window 128 + all visible C128 | 0.4077 | 0.2356 | 1.73x |
| backward | `swa_prefill_128_bwd` | `B=1,S=128,S_raw=512,H=64,D=512,TopK=128` | SWA window 128 | 0.9843 | 0.8054 | 1.22x |
| backward | `csa_prefill_128_bwd` | `B=1,S=128,S_raw=512,H=64,D=512,TopK=640` | raw window 128 + C4 topk 512 | 1.0735 | 0.8393 | 1.28x |
| backward | `hca_prefill_256_bwd` | `B=1,S=256,S_raw=256,H=64,D=512,TopK=130` | raw window 128 + all visible C128 | 1.0612 | 0.6007 | 1.77x |

Observed directionally:

- NVIDIA DSA forward is about `1.7x-1.9x` faster than Miles TileLang for these
  synthetic selected-KV shapes.
- NVIDIA DSA backward is about `1.2x-1.8x` faster on the three prefill-like
  backward shapes.
- Numerical parity was previously checked on small D=512/H=64 SWA/CSA/HCA
  cases; this table is a perf snapshot, not a full end-to-end training proof.

### Train-Like Batch Scaling Addendum

The initial batch-scaling probe included decode shapes because they were already
covered by the forward/inference matrix. For training readiness, the more
relevant view is prefill-like fwd+bwd. That follow-up ran on `agent-evelyn:2.2`
using the new B200 allocation:

```text
Node:       umbriel-b200-074
Backend:    NVIDIA DSA
Mode:       forward + backward
Container:  radixark/miles:deepseek-v4
cuDNN FE:   nvidia-cudnn-frontend[cutedsl]==1.24.0
Run output: /home/scratch.kaixih_ent/dsv4-kernel-bench-runs/dsa_train_batch_scaling_20260601-192845
```

| Case | B=1 fwd+bwd ms | B=2 | B=4 | B=8 | Throughput scale at B=8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `swa_prefill_128_bwd`, `S=128,S_raw=512,TopK=128` | 0.8201 | 0.6718 | 0.6630 | 1.0213 | 6.42x |
| `csa_prefill_128_bwd`, `S=128,S_raw=512,TopK=640` | 1.1562 | 1.0760 | 0.8554 | 1.4253 | 6.49x |
| `hca_prefill_256_bwd`, `S=256,S_raw=256,TopK=130` | 0.6901 | 0.7092 | 1.0778 | 1.8005 | 3.07x |

Read: for SWA/CSA `S=128`, DSA gains substantial throughput from batching up to
`B=8`; latency is nearly flat or even lower through `B=4`, then rises at `B=8`.
For HCA `S=256`, throughput still improves with batch, but less dramatically.
So `B=1` understated training-like DSA scaling, especially for the `S=128`
prefill-like cases.

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
