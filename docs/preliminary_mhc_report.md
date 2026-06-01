# Preliminary DSv4 mHC Kernel Report

Date: 2026-06-01

## Scope

This report covers the DSv4 mHC path:

```text
n-stream residual state -> one-stream layer input -> n-stream residual update
```

mHC is not an attention kernel. It is the residual-stream mixing machinery
around attention/MLP blocks. The first experiment asks whether the Miles
TileKernels path and NVIDIA Megatron cuTile path are runnable, correct, and
fast enough to use as the DSv4 training baseline.

## High-Level Operation

mHC, or Manifold-Constrained Hyper-Connections, keeps multiple residual streams
per token. A typical DSv4 setting uses `n = 4` streams. At each transformer
layer boundary:

1. The current hidden state is viewed as `n` residual streams per token.
2. A small projection produces per-token mixing weights.
3. Sinkhorn normalization constrains part of those weights into a stable mixing
   matrix.
4. A pre-layer aggregate mixes the `n` streams into one stream for the normal
   attention/MLP computation.
5. A post-layer update expands the layer output back into the `n` streams and
   mixes it with the previous residual streams.
6. At block boundaries, a head/contract path maps the `n` streams back to one
   stream for consumers such as logits or downstream modules.

In short: selected-KV sparse attention decides which KV entries a query reads;
mHC decides how residual streams are mixed before and after the layer.

## Backend Difference

| Area | Miles path | NVIDIA Megatron path | Initial read |
| --- | --- | --- | --- |
| Kernel family | DeepSeek/Miles wrapper around `tile_kernels.modeling.mhc.ops` | Megatron fused mHC kernels in `megatron.core.fusions.fused_mhc_kernels` | Both target the same mHC math family |
| Core primitives | pre norm/split/apply mix, pre big fuse, post, head compute, Sinkhorn | `sinkhorn`, `h_aggregate`, `h_post_bda`, `proj_rms` | NVIDIA exposes more modular primitive kernels |
| Training path | grad-enabled pre path is split across TileKernels primitives; no-grad path has a bigger fused pre kernel | fused cuTile primitives with native PyTorch/Megatron reference fallback | NVIDIA path is easier to test against its local reference |
| Runtime dependency | `tile_kernels` package | `cuda.tile` / cuTile in CUDA Python | Current Miles container lacks `tile_kernels`; NVIDIA imports in the same container |
| Integration | Miles DeepSeek-V4 plugin `hyper_connection.py` | Megatron `HyperConnectionModule` integrated into transformer blocks/MTP | Both are integration-level, not just standalone kernels |

Miles is not carrying the raw mHC kernel source directly in the Miles plugin.
The Miles plugin delegates to the external `tile_kernels.modeling.mhc.ops`
package. In this environment that path is the TileKernels/TileLang path:
`tilelang==0.1.8` is present in the container, and `tile-kernels==1.0.0
--no-deps` is enough to make the Miles wrapper import and run.

NVIDIA has two mHC paths in the Megatron branch:

- Native mHC is the PyTorch/Megatron reference fallback in
  `megatron.core.transformer.hyper_connection`. It is useful for correctness
  and availability, but it is not the intended performance comparison against
  Miles TileLang kernels.
- Fused cuTile mHC is the target NVIDIA kernel path in
  `megatron.core.fusions.fused_mhc_kernels`. This is the path that should be
  compared with Miles once it compiles in the runtime.

## Source Locations

Known source snapshots used for the initial inspection:

```text
miles-pr1045:       032721cd61bf7164955f084425eb9f315352fd26
Megatron-LM-nvidia: f553f2fe4c45479d1add2bea88253f51148f25d2
```

Relevant files:

```text
Miles:
  /scratch/repo/miles-pr1045/miles_plugins/models/deepseek_v4/ops/hyper_connection.py

NVIDIA Megatron:
  /scratch/repo/Megatron-LM-nvidia/megatron/core/transformer/hyper_connection.py
  /scratch/repo/Megatron-LM-nvidia/megatron/core/fusions/fused_mhc_kernels.py
  /scratch/repo/Megatron-LM-nvidia/tests/unit_tests/fusions/test_fused_mhc_kernels.py
```

## Experiment Plan

The first B200 experiment is intentionally narrow:

1. Confirm the runtime can import Miles mHC and NVIDIA mHC dependencies.
2. Run Miles TileKernels and NVIDIA fused cuTile mHC kernels on small and
   model-like layer-boundary shapes.
3. Use local PyTorch/Megatron reference code only to sanity-check NVIDIA fused
   cuTile correctness, not as a performance backend.
4. Run Miles TileKernels mHC only if `tile_kernels` is available without
   breaking the current Miles/TransformerEngine container stack.

## Initial Environment Finding

In `radixark/miles:deepseek-v4` on B200, NVIDIA's cuTile dependency is present
and the fused mHC module imports. The base Miles wrapper import fails because
`tile_kernels` is not installed in the image:

```text
IMPORT_OK cuda.tile
IMPORT_OK megatron.core.fusions.fused_mhc_kernels
NVIDIA_CUTILE_AVAILABLE True
IMPORT_FAIL tile_kernels.modeling.mhc.ops ModuleNotFoundError: No module named 'tile_kernels'
IMPORT_FAIL miles_plugins.models.deepseek_v4.ops.hyper_connection ModuleNotFoundError: No module named 'tile_kernels'
```

Installing `tile-kernels==1.0.0` with dependencies is not a clean fix for this
container: it pulls a new Torch/CUDA stack and breaks the existing
TransformerEngine binary. However, a transient no-dependency install works for
the smoke test:

```bash
python3 -m pip install --target /tmp/tilekernels_nodeps --no-deps tile-kernels==1.0.0
export PYTHONPATH=/tmp/tilekernels_nodeps:${PYTHONPATH}
```

For NVIDIA fused cuTile, `cuda.tile` imports but the container also needs the
`tileiras` compiler. A transient install exposes the binary:

```bash
python3 -m pip install --target /tmp/cuda_tileiras_pkg "cuda-tile[tileiras]"
export PATH=/tmp/cuda_tileiras_pkg/nvidia/cu13/bin:${PATH}
export PYTHONPATH=/tmp/cuda_tileiras_pkg:${PYTHONPATH}
```

## Results

Run context:

```text
Node: umbriel-b200-044, 8x NVIDIA B200
Container: radixark/miles:deepseek-v4
dsv4-kernel-bench: c5b8f6e8bf236667c2b46aaae85d9a850776bbd9
miles-pr1045:       032721cd61bf7164955f084425eb9f315352fd26
Megatron-LM-nvidia: f553f2fe4c45479d1add2bea88253f51148f25d2
Torch:              2.9.1+cu129
tilelang:           0.1.8
cuda-tile overlay:  1.4.0
tileiras:           /tmp/cuda_tileiras_pkg/nvidia/cu13/bin/tileiras
Run output:         /home/scratch.kaixih_ent/dsv4-kernel-bench-runs/mhc_20260531-220613/mhc_summary.json
```

### Import And Smoke

| Backend path | Result | Notes |
| --- | --- | --- |
| Miles base image | fail | `tile_kernels` missing |
| Miles + `tile-kernels==1.0.0 --no-deps` | pass | wrapper imports; fwd+bwd smoke passes on `B=1,S=8,n=4,C=1024` |
| NVIDIA native Megatron mHC | pass | reference/sanity path only; not a target backend |
| NVIDIA fused cuTile mHC | partial/fail | imports and finds `tileiras`; toy kernels compile, but useful mHC shapes hit Tile IR compilation failures |

Miles smoke output:

```text
layer_input: [1, 8, 1024]
post:        [1, 8, 4, 1]
comb:        [1, 8, 4, 4]
out:         [1, 8, 4, 1024]
status:      pass, including backward
```

NVIDIA fused cuTile failure, after fixing `tileiras` PATH:

```text
TileCompilerExecutionError: Return code 5
failed to compile Tile IR program
Unknown location
```

### cuTile Compiler Probe

The follow-up probe tested whether the failure was just a bad `cuda-tile` or
`tileiras` wheel pairing. It was run in the existing `agent-evelyn:2.1` pane on
`umbriel-b200-044`.

Probe outputs:

```text
/home/scratch.kaixih_ent/dsv4-kernel-bench-runs/mhc_cudatile_probe_20260531-222233
/home/scratch.kaixih_ent/dsv4-kernel-bench-runs/mhc_cudatile_shape_probe_20260531-223352
/home/scratch.kaixih_ent/dsv4-kernel-bench-runs/mhc_tileir_dump_20260531-222844
/home/scratch.kaixih_ent/dsv4-kernel-bench-runs/cutile_official_probe_20260531-224746
```

Runtime combinations tested:

| `cuda-tile` path | `tileiras` path | Result |
| --- | --- | --- |
| base `cuda-tile==1.3.0` | CUDA toolkit/tileiras `13.1.2.0` | tiny cases compile; useful shapes fail |
| base `cuda-tile==1.3.0` | CUDA toolkit/tileiras `13.2.1` | all tested cases fail |
| base `cuda-tile==1.3.0` | CUDA toolkit/tileiras `13.3.0` | all tested cases fail |
| overlay `cuda-tile==1.2.0` | package extra tileiras | all tested cases fail |
| overlay `cuda-tile==1.3.0` | package extra tileiras | all tested cases fail |
| overlay `cuda-tile==1.4.0` | package extra tileiras | all tested cases fail |
| overlay `cuda-tile==1.0.0/1.0.1/1.1.0` | CUDA toolkit/tileiras `13.1.2.0` | same shape pattern as base `1.3.0` |

Shape-level result with `tileiras` available:

| Fused cuTile primitive | Tiny/toy case | Useful case | Read |
| --- | --- | --- | --- |
| `sinkhorn` | `n=2,s*b=1` fwd+bwd passes | `n=4,s*b=8` fails in backward compile | compiler works only for the smallest specialization |
| `h_aggregate` | `n=1/2,C=1,s*b=1` forward passes for bf16/fp32 | `n=2,C=256,s*b=1` forward compile fails | failure appears once channel tile is meaningful |
| `h_post_bda` | `n=2,C=1,s*b=1` forward passes | `n=2,C=256,s*b=1` compile fails in the broader probe | toy pass does not exercise the real path |
| `proj_rms` | none found | even `M=1,N=1,K=128` forward compile fails | blocks fused mHC pipeline immediately |

`CUDA_TILE_DUMP_BYTECODE` produced `.tileirbc` bytecode for the failing kernels,
but `CUDA_TILE_DUMP_TILEIR` could not emit readable MLIR in this wheel:

```text
Can't print MLIR because the internal extension is missing. This is currently
not a public feature.
```

This means the current blocker is no longer just a missing `tileiras` binary.
`tileiras` is invoked successfully, but rejects the generated Tile IR bytecode
for non-toy mHC specializations on `sm_100`.

### Official cuTile Sample Probe

To separate Megatron mHC issues from a general cuTile runtime problem, we also
ran the upstream `NVIDIA/cutile-python` samples in the same B200 allocation and
container.

Environment:

```text
Node:                umbriel-b200-044
GPU:                 NVIDIA B200, compute capability 10.0
Driver:              595.58.03
Container image:     radixark/miles:deepseek-v4
Container CUDA:      12.9.1
Torch:               2.9.1+cu129
Temporary cuTile:    cuda-tile==1.4.0
Temporary compiler:  cuda-toolkit==13.3.0
tileiras:            nvidia-cuda-tileiras==13.3.36
nvcc/nvvm:           nvidia-cuda-nvcc==13.3.33, nvidia-nvvm==13.3.33
```

Commands exercised:

```text
python3 samples/MatMul.py --correctness-check
python3 samples/BatchMatMul.py --correctness-check
python3 samples/AttentionFMHA.py --correctness-check
python3 -m pytest -q samples/test_samples.py
```

Result:

| Upstream cuTile command | Result | Failure |
| --- | --- | --- |
| `samples/MatMul.py --correctness-check` | fail | `tileiras ... --gpu-name sm_100` returns code 5 |
| `samples/BatchMatMul.py --correctness-check` | fail | same Tile IR compile failure |
| `samples/AttentionFMHA.py --correctness-check` | fail | same Tile IR compile failure |
| `pytest samples/test_samples.py` | fail | 9 sample tests failed, including vector add, matmul, FMHA, LayerNorm, MoE |

Common warning before sample failures:

```text
Failed to detect the maximum supported TileIR bytecode version; falling back to
13.1.
```

The upstream `cuda.tile._compile` code attempts to detect supported bytecode by
probing `tileiras` with `--gpu-name sm_120` across bytecode versions 13.3,
13.2, and 13.1. In this environment that detection fails and cuTile falls back
to bytecode 13.1. The subsequent real sample compiles are for `--gpu-name
sm_100`, but still fail with `TileCompilerExecutionError`.

This makes the current cuTile blocker broader than Megatron mHC. The official
cuTile samples do not compile in this container/runtime combination either.

`CUDA_TILE_ENABLE_CRASH_DUMP=1` was also tested on the Megatron mHC failure
cases. It did not produce a zip archive because cuTile 1.4.0 hit a Python-side
dump bug:

```text
AttributeError: 'Block' object has no attribute 'body'
```

The failing bytecode files were still left under:

```text
/home/scratch.kaixih_ent/dsv4-kernel-bench-runs/cutile_official_probe_20260531-224746/mhc_crash_tmp
```

### CUDA 13 SGLang Probe

The follow-up experiment moved only the runtime stack, not the B200 allocation
or Megatron source tree. There was no local image literally named
`sglang_dev`; the available SGLang CUDA 13 image on the node was
`lmsysorg/sglang:v0.5.11`.

Environment:

```text
Node:                umbriel-b200-044
GPU:                 8x NVIDIA B200, driver 595.58.03
Container image:     lmsysorg/sglang:v0.5.11
Container CUDA:      13.0.1
Torch:               2.11.0+cu130
Temporary cuTile:    cuda-tile==1.4.0
Temporary compiler:  cuda-toolkit==13.3.0
tileiras:            nvidia-cuda-tileiras==13.3.36
nvcc/nvvm:           nvidia-cuda-nvcc==13.3.33, nvidia-nvvm==13.3.33
Run output:          /home/scratch.kaixih_ent/dsv4-kernel-bench-runs/sglang_cuda13_cutile_probe_20260531-225517
Correctness output:  /home/scratch.kaixih_ent/dsv4-kernel-bench-runs/sglang_mhc_correctness_probe_20260531-230829
Perf output:         /home/scratch.kaixih_ent/dsv4-kernel-bench-runs/sglang_mhc_perf_probe_20260531-232024/mhc_perf_results.json
```

Official `NVIDIA/cutile-python` sample result in this stack:

| Upstream cuTile command | Result | Notes |
| --- | --- | --- |
| `samples/MatMul.py --correctness-check` | pass | all tested matmul cases passed |
| `samples/BatchMatMul.py --correctness-check` | pass | fp16 and fp8 BMM cases passed |
| `samples/AttentionFMHA.py --correctness-check` | pass | non-causal, causal, and autotuned causal FMHA passed |
| `samples/quickstart/VectorAdd_quickstart.py` | skip/fail | missing optional `cupy`, not a cuTile compile failure |

The CUDA13 SGLang stack also fixed the Megatron fused mHC compiler blocker.
Initial fused forward smokes passed:

| Fused cuTile primitive | Shape | Result |
| --- | --- | --- |
| `h_aggregate` | tiny `[1, 1, 1]` | pass |
| `h_aggregate` | `n=2,C=256,s*b=1` | pass |
| `proj_rms` | `n=1,k=128` | pass |

A direct fwd+bwd correctness probe against local PyTorch references also
passed four fused primitives:

| Case | Result |
| --- | --- |
| `fused_sinkhorn_s2_b4_n4_fwd_bwd` | pass |
| `fused_h_aggregate_s2_b4_n4_c1024_fwd_bwd` | pass |
| `fused_h_post_bda_s2_b4_n4_c1024_bias_fwd_bwd` | pass |
| `fused_proj_rms_m64_n8_k512_fwd_bwd` | pass |

Finally, the Megatron unit test suite for the fused mHC file passed in the same
container:

```text
python3 -m pytest -q tests/unit_tests/fusions/test_fused_mhc_kernels.py -k Fused
22 passed, 29 warnings in 41.51s
```

This means the previous cuTile failure was a runtime-stack compatibility issue
in `radixark/miles:deepseek-v4` with CUDA 12.9/Torch cu129 plus overlay cuTile,
not proof that Megatron's fused mHC kernels are intrinsically broken.

### Correctness Status

Current correctness evidence now has both per-backend checks and a first
cross-container Miles-vs-cuTile parity harness:

| Backend | Evidence today | What it proves |
| --- | --- | --- |
| Miles TileKernels | Imports and runs fwd+bwd smoke in `radixark/miles:deepseek-v4` after a no-deps `tile-kernels` install | The Miles mHC path is runnable and differentiable in its native container |
| NVIDIA fused cuTile | Official cuTile samples pass in CUDA13 SGLang; fused mHC custom fwd+bwd probes pass; Megatron fused mHC pytest reports `22 passed` | The cuTile stack and fused mHC kernels are correct against local PyTorch/Megatron references in the CUDA13 container |

Cross-container run:

```text
/home/scratch.kaixih_ent/dsv4-kernel-bench-runs/mhc_cross_parity_20260601-064125
```

The harness used one deterministic host input bundle, ran Miles in
`radixark/miles:deepseek-v4` on pane `agent-evelyn:2.1`, ran NVIDIA fused
cuTile in `lmsysorg/sglang:v0.5.11` on pane `agent-evelyn:2.2`, then compared
CPU fp32 dumps.

High-signal results:

| Case | Tensor | Result |
| --- | --- | --- |
| `S=2,B=4,n=4,C=1024` | `layer_input` | close: cosine `0.9999907`, rel-L2 `0.00433` |
| `S=2,B=4,n=4,C=1024` | `post`, `comb` | close: post cosine `0.9999979`, comb cosine `0.9999963` |
| `S=2,B=4,n=4,C=1024` | `post_output` | mismatch: cosine `0.8817`, rel-L2 `0.4869` |
| `S=64,B=1,n=4,C=7168` | `layer_input` | close: cosine `0.9999925`, rel-L2 `0.00388` |
| `S=64,B=1,n=4,C=7168` | `post` | close: cosine `0.9999979`, rel-L2 `0.00204` |
| `S=64,B=1,n=4,C=7168` | `post_output` | mismatch: cosine `0.9018`, rel-L2 `0.4449` |

The mismatch appears to be a semantic surface mismatch in `mhc_post` / `h_res`
orientation rather than a random correctness failure. Using the saved tensors:

| Backend actual output | Formula that matches | Rel-L2 range |
| --- | --- | ---: |
| Miles `mhc_post` | `comb.T @ orig_res + post * layer_out` | `0.00233` to `0.00234` |
| NVIDIA `fused_h_post_bda` | `comb @ orig_res + post * layer_out` | `0.00308` to `0.00347` |

Megatron's local reference explicitly defines fused post as:

```text
output = h_res @ original_residual + h_post * x
```

So the next correctness step is to align the adapter convention: either pass
`h_res.transpose(-1, -2)` into NVIDIA when treating Miles as reference, or
confirm that the upstream callers intentionally store opposite orientations.
Until that is resolved, fwd/bwd parity is not proven even though both kernels
run and their pre/weight/post scalar pieces mostly agree.

### CUDA12.9 Miles-Container Performance Snapshot

This is the Miles-side baseline. NVIDIA fused cuTile is listed only as
unavailable in this CUDA12.9 container/runtime combination.

| Case | Backend | Mode | Avg ms | Status |
| --- | --- | --- | ---: | --- |
| `S=2,B=4,n=4,C=1024` | Miles TileKernels no-deps | forward | 0.1103 | pass |
| `S=2,B=4,n=4,C=1024` | Miles TileKernels no-deps | fwd+bwd | 1.3878 | pass |
| `S=64,B=1,n=4,C=7168` | Miles TileKernels no-deps | forward | 0.1591 | pass |
| `S=64,B=1,n=4,C=7168` | Miles TileKernels no-deps | fwd+bwd | 1.3818 | pass |
| both cases | NVIDIA fused cuTile | forward/fwd+bwd | n/a | Tile IR compile fail |

### CUDA13 SGLang Performance Snapshot

After moving to the CUDA13 SGLang stack, NVIDIA fused cuTile mHC runs on
meaningful shapes.

Run output:

```text
/home/scratch.kaixih_ent/dsv4-kernel-bench-runs/sglang_mhc_perf_probe_20260531-232024/mhc_perf_results.json
```

NVIDIA fused cuTile:

| Case | Fused fwd ms | Fused fwd+bwd ms | Status |
| --- | ---: | ---: | --- |
| `S=2,B=4,n=4,C=1024` | 0.1555 | 0.7900 | pass |
| `S=64,B=1,n=4,C=7168` | 0.2727 | 0.8140 | pass |
| `S=256,B=1,n=4,C=7168` | 0.3091 | 1.0531 | pass |

Cross-container no-bias Miles CUDA12.9 vs NVIDIA fused CUDA13 comparison:

Run output:

```text
/home/scratch.kaixih_ent/dsv4-kernel-bench-runs/mhc_cross_perf_20260601-065713
```

The two containers were run concurrently on the same B200 node:

```text
Miles:       radixark/miles:deepseek-v4, Torch 2.9.1+cu129, GPU0
NVIDIA:      lmsysorg/sglang:v0.5.11, Torch 2.11.0+cu130, GPU1
warmup/iters: 5 / 20
bias:        disabled on NVIDIA fused post to match the raw Miles surface
```

| Case | Miles TileKernels fwd ms | NVIDIA fused fwd ms | Miles TileKernels fwd+bwd ms | NVIDIA fused fwd+bwd ms | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| `S=2,B=4,n=4,C=1024` | 0.1870 | 0.1686 | 1.2124 | 0.6516 | NVIDIA slightly faster forward and much faster fwd+bwd |
| `S=64,B=1,n=4,C=7168` | 0.1967 | 0.2741 | 1.1861 | 0.6645 | Miles faster forward; NVIDIA faster fwd+bwd |
| `S=256,B=1,n=4,C=7168` | 0.1909 | 0.3067 | 1.2095 | 0.6767 | Miles faster forward; NVIDIA faster fwd+bwd |

This is a better comparison than the earlier table because the NVIDIA bias term
is disabled and both runs use the same shape list and timing loop. It is still
not a same-image benchmark, and the post orientation mismatch above means this
is a native-surface performance comparison rather than a correctness-certified
drop-in replacement result.

Miles TileKernels was also attempted in the CUDA13 SGLang image. After adding
`z3-solver`, `tilelang`, and `tile-kernels`, imports succeeded:

```text
tile_kernels.modeling.mhc.ops: ok
tilelang: ok
```

but the first TileLang lowering failed:

```text
AttributeError: '_NestedLoopCheckVisitor' object has no attribute '_inst'
```

So the single-image Miles-vs-NVIDIA mHC comparison still needs either a
known-good TileLang build for the CUDA13 SGLang image or a Miles image that has
the matching CUDA13 cuTile stack.

## Preliminary Interpretation

Current readiness view:

- In the Miles CUDA12.9 container, Miles TileKernels remains the stronger
  runnable path. It needs a no-dependency `tile-kernels` install, but after
  that it imports and runs forward/backward on the tested pre/post pipeline
  shapes.
- NVIDIA fused cuTile mHC should be evaluated in the CUDA13 SGLang stack. In
  that runtime, upstream cuTile samples pass, Megatron fused mHC forward smokes
  pass, local fwd+bwd parity smokes pass, and Megatron's fused mHC pytest passes.
- The two-container comparison suggests Miles TileKernels is generally faster
  for larger forward-only shapes, while NVIDIA fused cuTile is materially faster
  for forward+backward on all tested shapes. Treat this as native-surface
  performance until the `mhc_post` orientation convention is aligned.
- The CUDA12.9 Miles-container failure should be recorded as a container/compiler
  compatibility issue. It should not block a fair NVIDIA fused cuTile perf
  comparison if the benchmark can be run in `lmsysorg/sglang:v0.5.11` or the
  intended `sglang_dev` CUDA13 image.
- Next experiment: align `h_res` orientation in the adapter and rerun the same
  cross-container parity harness. After that, fix the SGLang TileLang lowering
  issue or build a shared image for a same-runtime benchmark.
