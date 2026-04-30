#!/usr/bin/env python3
"""Compare `cli default` behavior and runtime between Python AIC and aic-rust."""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


CASES: list[list[str]] = [
    ["--isl", "128", "--osl", "32", "--ttft", "2000", "--tpot", "30"],
    ["--isl", "256", "--osl", "32", "--ttft", "2000", "--tpot", "30"],
    ["--isl", "512", "--osl", "64", "--ttft", "2000", "--tpot", "30"],
    ["--isl", "1024", "--osl", "64", "--ttft", "2000", "--tpot", "30"],
    ["--isl", "128", "--osl", "64", "--ttft", "1000", "--tpot", "25"],
    ["--isl", "256", "--osl", "128", "--ttft", "1500", "--tpot", "35"],
    ["--isl", "512", "--osl", "128", "--ttft", "2500", "--tpot", "40"],
    ["--isl", "768", "--osl", "64", "--ttft", "2500", "--tpot", "30"],
    ["--isl", "1536", "--osl", "32", "--ttft", "3000", "--tpot", "35"],
    ["--isl", "2048", "--osl", "32", "--ttft", "3500", "--tpot", "40"],
]


def run(cmd: list[str]) -> tuple[float, int, str, str]:
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    return time.perf_counter() - start, proc.returncode, proc.stdout, proc.stderr


def parse_summary(text: str) -> dict[str, str | float | None]:
    patterns = {
        "chosen": r"Best Experiment Chosen:\s+(.+)",
        "best_throughput": r"Best Throughput:\s+([0-9,.]+)\s+tokens/s",
        "per_user_throughput": r"Per-user throughput:\s+([0-9,.]+)\s+tokens/s/user",
        "request_rate": r"Request Rate:\s+([0-9,.]+)\s+req/s",
        "ttft": r"TTFT:\s+([0-9,.]+)ms",
        "tpot": r"TPOT:\s+([0-9,.]+)ms",
        "request_latency": r"Request Latency:\s+([0-9,.]+)ms",
    }
    parsed: dict[str, str | float | None] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if not match:
            parsed[key] = None
            continue
        value = match.group(1).strip()
        if key == "chosen":
            parsed[key] = re.sub(r"\s+", " ", value)
        else:
            parsed[key] = float(value.replace(",", ""))
    return parsed


def summaries_close(left: dict[str, str | float | None], right: dict[str, str | float | None]) -> bool:
    if left.get("chosen") is None or right.get("chosen") is None:
        return False
    left_mode = str(left["chosen"]).split()[0]
    right_mode = str(right["chosen"]).split()[0]
    if left_mode != right_mode:
        return False
    for key in ("best_throughput", "request_rate", "ttft", "tpot", "request_latency"):
        lv = left.get(key)
        rv = right.get(key)
        if not isinstance(lv, float) or not isinstance(rv, float):
            return False
        tolerance = max(0.02, abs(lv) * 0.005)
        if abs(lv - rv) > tolerance:
            return False
    return True


def main() -> int:
    python_cli = ROOT / ".venv" / "bin" / "aiconfigurator"
    rust_bin = ROOT / "target" / "release" / "aic-rust"
    if not rust_bin.exists():
        build = subprocess.run(["cargo", "build", "--release"], cwd=ROOT)
        if build.returncode != 0:
            return build.returncode

    base = [
        "--model-path",
        "Qwen/Qwen3-0.6B",
        "--total-gpus",
        "1",
        "--system",
        "h200_sxm",
        "--backend",
        "trtllm",
        "--top-n",
        "1",
        "--no-color",
    ]

    rows = []
    for index, case_args in enumerate(CASES, start=1):
        default_args = base + case_args
        py_cmd = [str(python_cli), "cli", "default", *default_args]
        rs_cmd = [str(rust_bin), "cli", "default", *default_args]
        py_t, py_code, py_out, py_err = run(py_cmd)
        rs_t, rs_code, rs_out, rs_err = run(rs_cmd)
        py_summary = parse_summary(py_out)
        rs_summary = parse_summary(rs_out)
        rows.append(
            {
                "case": index,
                "args": case_args,
                "python_seconds": py_t,
                "rust_seconds": rs_t,
                "python_code": py_code,
                "rust_code": rs_code,
                "same_summary": summaries_close(py_summary, rs_summary),
                "python_summary": py_summary,
                "rust_summary": rs_summary,
                "python_stderr_tail": py_err.strip().splitlines()[-5:],
                "rust_stderr_tail": rs_err.strip().splitlines()[-5:],
            }
        )

    print(json.dumps(rows, indent=2))
    ok = all(row["python_code"] == 0 and row["rust_code"] == 0 and row["same_summary"] for row in rows)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
