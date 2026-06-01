# B200 Reproduction Runbook

This records the working flow used on `agent-evelyn:2.1` to compare Miles
TileLang sparse attention against NVIDIA Megatron DSA on B200.

## Sources

Use these persistent host paths:

```bash
/home/scratch.kaixih_ent/repo/dsv4-kernel-bench
/home/scratch.kaixih_ent/repo/miles-pr1045
/home/scratch.kaixih_ent/repo/Megatron-LM-nvidia
```

Known-good source commits:

```text
dsv4-kernel-bench: v1-sparse-attn-harness at d883caf or newer
miles-pr1045:       032721cd61bf7164955f084425eb9f315352fd26
Megatron-LM-nvidia: f553f2fe4c45479d1add2bea88253f51148f25d2
```

Clone or update:

```bash
set +e
mkdir -p /home/scratch.kaixih_ent/repo
cd /home/scratch.kaixih_ent/repo

git clone -b v1-sparse-attn-harness https://github.com/kaixih/dsv4-kernel-bench || true
cd dsv4-kernel-bench
git pull --ff-only origin v1-sparse-attn-harness
cd ..

git clone -b deepseek-v4 https://github.com/yueming-yuan/miles miles-pr1045 || true
cd miles-pr1045
git checkout deepseek-v4
git pull --ff-only origin deepseek-v4
cd ..

git clone https://github.com/NVIDIA/Megatron-LM Megatron-LM-nvidia || true
cd Megatron-LM-nvidia
git fetch origin dev
git checkout f553f2fe4c45479d1add2bea88253f51148f25d2
```

## Container

Use `radixark/miles:deepseek-v4`. Do not run `pip install -e .` on the mounted
repo; the mount can reject root writes such as `egg-info`. Use `PYTHONPATH`
instead.

Always run `set +e` in the interactive allocation shell before experiments.

```bash
docker pull radixark/miles:deepseek-v4

docker run --rm --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /home/scratch.kaixih_ent/repo/dsv4-kernel-bench:/scratch/repo/dsv4-kernel-bench \
  -v /home/scratch.kaixih_ent/repo/miles-pr1045:/scratch/repo/miles-pr1045 \
  -v /home/scratch.kaixih_ent/repo/Megatron-LM-nvidia:/scratch/repo/Megatron-LM-nvidia \
  -w /scratch/repo/dsv4-kernel-bench \
  radixark/miles:deepseek-v4 bash
```

Inside the container:

```bash
set +e
export PYTHONPATH=/scratch/repo/dsv4-kernel-bench/src:/scratch/repo/miles-pr1045:/scratch/repo/Megatron-LM-nvidia:${PYTHONPATH}
```

## Required NVIDIA DSA Dependency

The image contains `nvidia-cudnn-frontend==1.17.0`, which lacks `cudnn.DSA`.
Forward-only NVIDIA DSA works through FlashMLA, but NVIDIA DSA backward fails
until cuDNN Frontend is upgraded. Pin the known-good version before running
NVIDIA DSA backward or parity tests:

```bash
python3 -m pip install --upgrade "nvidia-cudnn-frontend[cutedsl]==1.24.0"
python3 - <<'PY'
from cudnn import DSA
print("cudnn.DSA OK", DSA)
PY
```

Verification notes:

```text
nvidia-cudnn-frontend==1.17.0:
  import cudnn succeeds
  from cudnn import DSA fails

nvidia-cudnn-frontend[cutedsl]==1.24.0:
  from cudnn import DSA succeeds
  cudnn.deepseek_sparse_attention is present
  NVIDIA DSA backward passes Miles parity on D=512/H=64 CSA and SWA smoke shapes
```

Because this is installed inside a transient container, the upgrade is not
persisted in the base `radixark/miles:deepseek-v4` image.

## Smoke Commands

Index/layout tests:

```bash
python3 -m pytest tests/test_indexing.py -q
```

Miles TileLang fwd+bwd smoke:

```bash
python3 -m dsv4_kernel_bench.bench \
  --backend miles_tilelang --device cuda \
  --batch 1 --seqlen-q 8 --seqlen-kv 64 --heads 64 --dim 512 --topk 64 \
  --warmup 0 --iters 1 \
  --output /tmp/bench_miles_h64.json
```

NVIDIA DSA fwd+bwd smoke:

```bash
python3 -m dsv4_kernel_bench.bench \
  --backend nvidia_dsa --device cuda \
  --batch 1 --seqlen-q 8 --seqlen-kv 64 --heads 64 --dim 512 --topk 64 \
  --warmup 0 --iters 1 \
  --output /tmp/bench_dsa_h64.json
```

Forward-only DSA can run without `cudnn.DSA`:

```bash
python3 -m dsv4_kernel_bench.bench \
  --backend nvidia_dsa --device cuda \
  --batch 1 --seqlen-q 8 --seqlen-kv 64 --heads 64 --dim 512 --topk 64 \
  --warmup 0 --iters 1 --no-backward \
  --output /tmp/bench_dsa_h64_fwd.json
```

## Perf Matrix Commands

Forward-only perf uses no-grad and reuses the same synthetic tensors across
iterations. Backward perf intentionally clones per iteration so gradients do
not accumulate.

Run Miles forward perf:

```bash
python3 -m dsv4_kernel_bench.bench_matrix \
  --config configs/blue_module_perf_forward.json \
  --output-dir /tmp/dsv4-kernel-bench-runs/perf_forward_miles
```

Run NVIDIA forward perf:

```bash
python3 - <<'PY'
import json
src = "configs/blue_module_perf_forward.json"
dst = "/tmp/blue_module_perf_forward_nvidia.json"
with open(src) as f:
    cfg = json.load(f)
cfg["defaults"]["backend"] = "nvidia_dsa"
with open(dst, "w") as f:
    json.dump(cfg, f, indent=2, sort_keys=True)
print(dst)
PY

python3 -m dsv4_kernel_bench.bench_matrix \
  --config /tmp/blue_module_perf_forward_nvidia.json \
  --output-dir /tmp/dsv4-kernel-bench-runs/perf_forward_nvidia
```

For backward perf, upgrade cuDNN Frontend first, then use
`configs/blue_module_perf_backward.json` and the same backend-swap pattern.

## Known Results

Shape:

```text
B=1, S=8, S_kv=64, H=64, D=512, TopK=64
```

Miles TileLang fwd+bwd:

```text
first_call_s:      16.1090
avg_iter_s:         0.001637
peak_memory_bytes:  3482112
```

NVIDIA DSA fwd+bwd after upgrading cudnn-frontend:

```text
first_call_s:      7.3753
avg_iter_s:        0.001336
peak_memory_bytes: 2963968
```

Miles vs NVIDIA DSA fwd+bwd diff:

```text
output:      rel_l2=8.9884e-05, max_abs=0.00390625
dq:          rel_l2=2.1509e-04, max_abs=0.00390625
dkv:         rel_l2=1.4131e-04, max_abs=0.0625
d_attn_sink: rel_l2=7.1851e-05, max_abs=4.14997e-05
```

## Notes

- `H=2` fails in the bundled FlashMLA path with `RuntimeError: Unsupported h_q: 2`.
- The harness has a compatibility shim for older FlashMLA wheels that do not
  accept the `indexer_topk` keyword. It only applies to Path A/C sparse
  attention where `indexer_topk == 0`.
- Keep benchmark JSON in `/tmp` if the mounted repo is not writable from the
  container.

## mHC Notes

The `radixark/miles:deepseek-v4` image does not include `tile_kernels` by
default. Do not run a normal dependency install for `tile-kernels==1.0.0`; it
upgrades Torch/CUDA packages and breaks the existing TransformerEngine wheel.
Use a transient no-dependency target instead:

```bash
python3 -m pip install --target /tmp/tilekernels_nodeps --no-deps tile-kernels==1.0.0
export PYTHONPATH=/tmp/tilekernels_nodeps:${PYTHONPATH}
```

NVIDIA Megatron fused mHC imports with `cuda.tile`, but actual cuTile kernel
launch also needs `tileiras`:

```bash
python3 -m pip install --target /tmp/cuda_tileiras_pkg "cuda-tile[tileiras]"
export PATH=/tmp/cuda_tileiras_pkg/nvidia/cu13/bin:${PATH}
export PYTHONPATH=/tmp/cuda_tileiras_pkg:${PYTHONPATH}
```

On `umbriel-b200-044` with the source commits above, this made `tileiras`
discoverable, but NVIDIA fused mHC still failed during Tile IR compilation:

```text
TileCompilerExecutionError: Return code 5
failed to compile Tile IR program
Unknown location
```

Useful follow-up probe outputs:

```text
/home/scratch.kaixih_ent/dsv4-kernel-bench-runs/mhc_20260531-220613/mhc_summary.json
/home/scratch.kaixih_ent/dsv4-kernel-bench-runs/mhc_cudatile_probe_20260531-222233
/home/scratch.kaixih_ent/dsv4-kernel-bench-runs/mhc_cudatile_shape_probe_20260531-223352
/home/scratch.kaixih_ent/dsv4-kernel-bench-runs/mhc_tileir_dump_20260531-222844
/home/scratch.kaixih_ent/dsv4-kernel-bench-runs/cutile_official_probe_20260531-224746
```

The cuTile probe covered these combinations:

```text
base cuda-tile 1.3.0 + tileiras 13.1/13.2/13.3
overlay cuda-tile 1.2.0/1.3.0/1.4.0 with package extra tileiras
overlay cuda-tile 1.0.0/1.0.1/1.1.0 + explicit tileiras 13.1
```

Observed shape pattern:

```text
PASS: sinkhorn n=2, s*b=1 fwd+bwd
PASS: h_aggregate n=1/2, C=1, s*b=1 forward
PASS: h_post_bda n=2, C=1, s*b=1 forward
FAIL: sinkhorn n=4, s*b=8 backward compile
FAIL: h_aggregate n=2, C=256 forward compile
FAIL: h_post_bda n=2, C=256 forward compile
FAIL: proj_rms even at M=1, N=1, K=128 forward compile
```

So for mHC, the fair target remains Miles TileKernels/TileLang versus NVIDIA
fused cuTile, but the current `radixark/miles:deepseek-v4` runtime cannot run a
meaningful NVIDIA fused cuTile mHC benchmark. Use NVIDIA native only as a
reference/fallback baseline unless NVIDIA provides the exact cuTile/tileiras
runtime expected by this Megatron commit.

Official `NVIDIA/cutile-python` sample baseline in the same container:

```bash
python3 -m pip install --target /tmp/cutile_official "cuda-tile[tileiras]==1.4.0" pytest numpy
export PATH=/tmp/cutile_official/nvidia/cu13/bin:/tmp/cutile_official/nvidia/cu13/nvvm/bin:${PATH}
export PYTHONPATH=/tmp/cutile_official:${PYTHONPATH}
git clone --depth 1 https://github.com/NVIDIA/cutile-python.git /tmp/cutile-python
cd /tmp/cutile-python
python3 samples/MatMul.py --correctness-check
python3 samples/BatchMatMul.py --correctness-check
python3 samples/AttentionFMHA.py --correctness-check
python3 -m pytest -q samples/test_samples.py
```

Observed environment:

```text
container CUDA:      12.9.1
torch:               2.9.1+cu129
temporary cuTile:    cuda-tile==1.4.0
temporary tileiras:  nvidia-cuda-tileiras==13.3.36
temporary nvcc/nvvm: nvidia-cuda-nvcc==13.3.33, nvidia-nvvm==13.3.33
driver:              595.58.03
GPU:                 B200, sm_100
```

All official cuTile commands above failed with:

```text
TileCompilerExecutionError: Return code 5
failed to compile Tile IR program
Unknown location
```

They also printed:

```text
Failed to detect the maximum supported TileIR bytecode version; falling back to 13.1.
```

So the current cuTile failure is not isolated to Megatron mHC. Resolve the
official cuTile sample failure first, likely by changing the container/compiler
stack or using NVIDIA's known-good cuTile runtime for this driver/GPU.
