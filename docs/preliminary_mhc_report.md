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
2. Run NVIDIA fused cuTile kernels against Megatron native references for
   correctness.
3. Measure NVIDIA fused vs native reference latency for the mHC primitives and
   a small end-to-end layer-boundary pipeline.
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
| NVIDIA native Megatron mHC | pass | native reference path runs and benchmarks |
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

### Performance Snapshot

The most comparable row is the layer-boundary pre/post pipeline. It is still
not byte-identical across implementations:

- Miles measures `hc_pre_raw + hc_post_raw` through TileKernels.
- NVIDIA native measures `proj_rms + compute_h + sinkhorn + aggregate +
  h_post_bda` through Megatron native functions.
- NVIDIA fused should be the intended comparison backend, but it does not
  compile in this container/runtime combination yet.

| Case | Backend | Mode | Avg ms | Status |
| --- | --- | --- | ---: | --- |
| `S=2,B=4,n=4,C=1024` | Miles TileKernels no-deps | forward | 0.1103 | pass |
| `S=2,B=4,n=4,C=1024` | NVIDIA native | forward | 0.3897 | pass |
| `S=2,B=4,n=4,C=1024` | Miles TileKernels no-deps | fwd+bwd | 1.3878 | pass |
| `S=2,B=4,n=4,C=1024` | NVIDIA native | fwd+bwd | 2.4783 | pass |
| `S=64,B=1,n=4,C=7168` | Miles TileKernels no-deps | forward | 0.1591 | pass |
| `S=64,B=1,n=4,C=7168` | NVIDIA native | forward | 1.0166 | pass |
| `S=64,B=1,n=4,C=7168` | Miles TileKernels no-deps | fwd+bwd | 1.3818 | pass |
| `S=64,B=1,n=4,C=7168` | NVIDIA native | fwd+bwd | 5.7900 | pass |
| both cases | NVIDIA fused cuTile | forward/fwd+bwd | n/a | Tile IR compile fail |

NVIDIA native primitive timings, for context:

| Case | Primitive | Forward ms | Fwd+Bwd ms |
| --- | --- | ---: | ---: |
| `S=2,B=4,n=4,C=1024` | `sinkhorn` | 0.1197 | 0.8945 |
| `S=2,B=4,n=4,C=1024` | `h_aggregate` | 0.0581 | 0.3454 |
| `S=2,B=4,n=4,C=1024` | `h_post_bda` | 0.0762 | 0.5645 |
| `S=2,B=4,n=4,C=1024` | `proj_rms` | 0.0794 | 0.5394 |
| `S=64,B=1,n=4,C=7168` | `sinkhorn` | 0.2713 | 3.8431 |
| `S=64,B=1,n=4,C=7168` | `h_aggregate` | 0.0712 | 0.5041 |
| `S=64,B=1,n=4,C=7168` | `h_post_bda` | 0.0947 | 0.7207 |
| `S=64,B=1,n=4,C=7168` | `proj_rms` | 0.1046 | 0.5853 |

## Preliminary Interpretation

Current readiness view:

- Miles TileKernels is the stronger runnable path in this environment. It needs
  a no-dependency `tile-kernels` install, but after that it imports, runs
  forward/backward, and is faster than NVIDIA native on the tested pre/post
  pipeline shapes.
- NVIDIA's fused cuTile implementation is architecturally attractive and more
  modular, but it is not ready in the tested runtime. A few toy specializations
  compile, but the mHC-relevant shapes needed for comparison fail during Tile
  IR compilation.
- Official upstream cuTile samples also fail in the same runtime, so the next
  debugging step should focus on the cuTile/`tileiras`/driver/container
  compatibility stack before spending more time on Megatron mHC code shape.
- NVIDIA native is useful as a local reference and fallback, not as the likely
  performance target.
- The next fair comparison should either use NVIDIA's exact cuTile/tileiras
  runtime expected by this Megatron branch, or file/debug the Tile IR compiler
  failure before drawing a final performance conclusion about NVIDIA fused mHC.
