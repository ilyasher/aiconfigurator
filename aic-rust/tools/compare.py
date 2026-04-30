#!/usr/bin/env python3
"""Small Python-vs-Rust comparison harness for the native Rust slices."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> tuple[float, str, str, int]:
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    return time.perf_counter() - start, proc.stdout, proc.stderr, proc.returncode


def main() -> int:
    python = str(ROOT / ".venv" / "bin" / "python")
    rust_bin = ROOT / "target" / "debug" / "aic-rust"
    if not rust_bin.exists():
        build = subprocess.run(["cargo", "build"], cwd=ROOT)
        if build.returncode != 0:
            return build.returncode
    rust = [str(rust_bin)]

    comparisons = [
        (
            "support",
            [
                python,
                "-c",
                (
                    "import json;"
                    "from aiconfigurator.sdk.common import check_support;"
                    "r=check_support('Qwen/Qwen3-32B-FP8','h200_sxm','trtllm',None);"
                    "print(json.dumps({'agg_supported':r.agg_supported,'disagg_supported':r.disagg_supported,"
                    "'exact_match':r.exact_match,'architecture':None,'agg_pass_count':0,'agg_total_count':0,"
                    "'disagg_pass_count':0,'disagg_total_count':0}))"
                ),
            ],
            rust
            + [
                "cli",
                "support",
                "--model-path",
                "Qwen/Qwen3-32B-FP8",
                "--system",
                "h200_sxm",
                "--backend",
                "trtllm",
                "--json",
            ],
        ),
        (
            "gemm exact",
            [
                python,
                "-c",
                (
                    "import json;"
                    "from aiconfigurator.sdk.perf_database import get_database;"
                    "from aiconfigurator.sdk import common;"
                    "db=get_database('h200_sxm','trtllm','1.2.0rc5');"
                    "r=db.query_gemm(8192,65536,10240,common.GEMMQuantMode.int4_wo);"
                    "print(json.dumps({'latency':float(r),'energy':r.energy}))"
                ),
            ],
            rust
            + [
                "query-gemm",
                "--system",
                "h200_sxm",
                "--backend",
                "trtllm",
                "--version",
                "1.2.0rc5",
                "--m",
                "8192",
                "--n",
                "65536",
                "--k",
                "10240",
                "--quant",
                "int4_wo",
            ],
        ),
        (
            "generate parallelism",
            [
                python,
                "-c",
                (
                    "import json;"
                    "from aiconfigurator.cli import cli_generate;"
                    "r=cli_generate(model_path='Qwen/Qwen3-32B-FP8', total_gpus=8, system='h200_sxm', backend='trtllm');"
                    "p=r['parallelism'];"
                    "print(json.dumps({'tensor_parallel_size':p['tp'],'pipeline_parallel_size':p['pp'],"
                    "'replicas':p['replicas'],'gpus_used':p['gpus_used']}))"
                ),
            ],
            rust
            + [
                "cli",
                "generate",
                "--model-path",
                "Qwen/Qwen3-32B-FP8",
                "--total-gpus",
                "8",
                "--system",
                "h200_sxm",
                "--backend",
                "trtllm",
                "--json",
            ],
        ),
        (
            "context attention exact",
            [
                python,
                "-c",
                (
                    "import json;"
                    "from aiconfigurator.sdk.perf_database import get_database;"
                    "from aiconfigurator.sdk import common;"
                    "db=get_database('h200_sxm','trtllm','1.2.0rc5');"
                    "r=db.query_context_attention(b=8,s=16384,prefix=0,n=96,n_kv=1,"
                    "kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,"
                    "fmha_quant_mode=common.FMHAQuantMode.bfloat16,window_size=128,head_size=64);"
                    "print(json.dumps({'latency':float(r),'energy':r.energy}))"
                ),
            ],
            rust
            + [
                "query-context-attention",
                "--system",
                "h200_sxm",
                "--backend",
                "trtllm",
                "--version",
                "1.2.0rc5",
                "--batch-size",
                "8",
                "--seq-len",
                "16384",
                "--num-heads",
                "96",
                "--num-kv-heads",
                "1",
                "--kv-cache-quant",
                "bfloat16",
                "--fmha-quant",
                "bfloat16",
                "--window-size",
                "128",
                "--head-size",
                "64",
            ],
        ),
    ]

    rows = []
    for name, py_cmd, rust_cmd in comparisons:
        py_t, py_out, py_err, py_code = run(py_cmd)
        rs_t, rs_out, rs_err, rs_code = run(rust_cmd)
        rows.append(
            {
                "name": name,
                "python_seconds": py_t,
                "rust_seconds": rs_t,
                "python_code": py_code,
                "rust_code": rs_code,
                "python_stdout": py_out.strip(),
                "rust_stdout": rs_out.strip(),
                "python_stderr": py_err.strip(),
                "rust_stderr": rs_err.strip(),
                "same_json": _same_json(py_out, rs_out),
            }
        )

    print(json.dumps(rows, indent=2))
    return 0 if all(row["python_code"] == 0 and row["rust_code"] == 0 and row["same_json"] for row in rows) else 1


def _same_json(left: str, right: str) -> bool:
    try:
        left_json = json.loads(left)
        right_json = json.loads(right)
        if isinstance(left_json, dict) and isinstance(right_json, dict):
            return all(right_json.get(key) == value for key, value in left_json.items())
        return left_json == right_json
    except Exception:
        return left.strip() == right.strip()


if __name__ == "__main__":
    raise SystemExit(main())
