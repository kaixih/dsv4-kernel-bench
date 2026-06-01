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
| NVIDIA fused cuTile mHC | fail | imports and finds `tileiras`, but Tile IR compilation fails |

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

This happens for all tested fused primitives: `sinkhorn`, `h_aggregate`,
`h_post_bda`, `proj_rms`, and the end-to-end layer-boundary pipeline.

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
  modular, but it is not ready in the tested runtime: every fused primitive
  reaches cuTile and then fails during Tile IR compilation.
- NVIDIA native is useful as a local reference and fallback, not as the likely
  performance target.
- The next fair comparison should either use NVIDIA's exact cuTile/tileiras
  runtime expected by this Megatron branch, or file/debug the Tile IR compiler
  failure before drawing a final performance conclusion about NVIDIA fused mHC.
