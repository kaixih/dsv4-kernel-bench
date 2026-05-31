from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from dsv4_kernel_bench.backends import BackendUnavailable, run_sparse_attention_backend
from dsv4_kernel_bench.data import make_sparse_attention_inputs


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _run_once(backend, inputs, backward: bool) -> None:
    q = inputs.q.detach().clone().requires_grad_(backward)
    kv = inputs.kv.detach().clone().requires_grad_(backward)
    attn_sink = inputs.attn_sink.detach().clone().requires_grad_(backward)
    out = run_sparse_attention_backend(backend, q, kv, attn_sink, inputs.topk_idxs, inputs.sm_scale)
    if backward:
        out.float().sum().backward()


def measure(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    inputs = make_sparse_attention_inputs(
        batch=args.batch,
        seqlen_q=args.seqlen_q,
        seqlen_kv=args.seqlen_kv,
        heads=args.heads,
        dim=args.dim,
        topk=args.topk,
        invalid_fraction=args.invalid_fraction,
        device=device,
        seed=args.seed,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.perf_counter()
    _run_once(args.backend, inputs, args.backward)
    _sync(device)
    first_call_s = time.perf_counter() - t0

    for _ in range(args.warmup):
        _run_once(args.backend, inputs, args.backward)
    _sync(device)

    t0 = time.perf_counter()
    for _ in range(args.iters):
        _run_once(args.backend, inputs, args.backward)
    _sync(device)
    total_s = time.perf_counter() - t0

    peak_mem = None
    if device.type == "cuda":
        peak_mem = int(torch.cuda.max_memory_allocated(device))

    return {
        "backend": args.backend,
        "shape": {
            "B": args.batch,
            "S": args.seqlen_q,
            "S_kv": args.seqlen_kv,
            "H": args.heads,
            "D": args.dim,
            "TopK": args.topk,
        },
        "backward": args.backward,
        "warmup": args.warmup,
        "iters": args.iters,
        "first_call_s": first_call_s,
        "avg_iter_s": total_s / args.iters,
        "peak_memory_bytes": peak_mem,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark DSv4 sparse attention backends")
    parser.add_argument("--backend", choices=["miles_tilelang", "nvidia_dsa"], required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seqlen-q", type=int, default=16)
    parser.add_argument("--seqlen-kv", type=int, default=32)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--invalid-fraction", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--no-backward", action="store_false", dest="backward")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    try:
        result = measure(args)
    except BackendUnavailable as exc:
        print(f"SKIP: {exc}")
        return 77

    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
