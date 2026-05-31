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

Known-good commits:

```text
dsv4-kernel-bench: bb437998e196bcfee0a8dbc214a2bce30e306fa1
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
Upgrade it before running NVIDIA DSA backward:

```bash
python3 -m pip install --upgrade "nvidia-cudnn-frontend[cutedsl]"
python3 - <<'PY'
from cudnn import DSA
print("cudnn.DSA OK", DSA)
PY
```

The successful run used `nvidia-cudnn-frontend==1.24.0`.

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
