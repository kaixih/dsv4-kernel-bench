from __future__ import annotations

import argparse
import json
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

from dsv4_kernel_bench.backends import BackendUnavailable
from dsv4_kernel_bench.bench import measure


DEFAULT_ARGS: dict[str, Any] = {
    "backend": "miles_tilelang",
    "device": "cuda",
    "batch": 1,
    "seqlen_q": 16,
    "seqlen_kv": 32,
    "heads": 2,
    "dim": 64,
    "topk": 8,
    "selection_pattern": "random",
    "raw_seqlen_kv": None,
    "window_size": None,
    "compressed_topk": None,
    "compress_ratio": None,
    "query_start": 0,
    "invalid_fraction": 0.0,
    "seed": 0,
    "warmup": 5,
    "iters": 20,
    "backward": True,
    "output": None,
}


def _normalize_keys(values: dict[str, Any]) -> dict[str, Any]:
    return {key.replace("-", "_"): value for key, value in values.items()}


def _case_args(defaults: dict[str, Any], case: dict[str, Any], output: Path | None) -> Namespace:
    merged = DEFAULT_ARGS | _normalize_keys(defaults) | _normalize_keys(case)
    merged.pop("name", None)
    merged["output"] = output
    return Namespace(**merged)


def run_matrix(config_path: Path, output_dir: Path | None) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    defaults = config.get("defaults", {})
    cases = config.get("cases", [])
    if not isinstance(cases, list) or not cases:
        raise ValueError("matrix config must contain a non-empty 'cases' list")

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for index, raw_case in enumerate(cases):
        case = _normalize_keys(raw_case)
        name = str(case.get("name", f"case_{index:03d}"))
        case_output = None if output_dir is None else output_dir / f"{index:03d}_{name}.json"
        args = _case_args(defaults, case, case_output)
        started = time.perf_counter()
        try:
            result = measure(args)
            status = "pass"
            error = None
        except BackendUnavailable as exc:
            result = None
            status = "skip"
            error = str(exc)
        except Exception as exc:  # pragma: no cover - preserves matrix progress remotely
            result = None
            status = "fail"
            error = f"{type(exc).__name__}: {exc}"

        record = {
            "name": name,
            "status": status,
            "elapsed_s": time.perf_counter() - started,
            "error": error,
            "result": result,
        }
        results.append(record)
        if case_output is not None and result is None:
            case_output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")

    summary = {
        "config": str(config_path),
        "output_dir": None if output_dir is None else str(output_dir),
        "results": results,
    }
    if output_dir is not None:
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a DSv4 selected-KV benchmark matrix")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    summary = run_matrix(args.config, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if any(item["status"] == "fail" for item in summary["results"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
