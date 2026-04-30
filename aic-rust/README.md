# aic-rust

Rust rewrite-in-progress for the `aiconfigurator` SDK and CLI.

> Alpha status: this port is still experimental. The Python implementation remains the
> source of truth, and Rust parity is being built up piece by piece with tests.

This directory lives inside the `aiconfigurator` repository. The Rust code defaults to using the
parent checkout for Python model configs and system performance databases. Tests call the editable
Python install in `.venv` and compare Rust outputs against Python outputs.

## Current Coverage

- SDK enum types and `PerformanceResult`
- cached/local model config parsing for common AIC model configs
- system YAML loading
- supported database discovery and latest-version selection
- support matrix checks
- parallel config enumeration
- GEMM performance table loading
- GEMM `SILICON`, `HYBRID`, `EMPIRICAL`, and `SOL` queries for exact and simple interpolated points
- context/prefill attention performance table loading
- context/prefill attention `SILICON`, `HYBRID`, `EMPIRICAL`, and `SOL` queries, including Python-style extrapolated grids
- generation/decode attention performance table loading and queries
- regular context/generation MLA performance table loading and queries
- MoE performance table loading and native MoE query parity for regular tables
- custom all-reduce, NCCL, P2P, and memory-operation performance queries
- native dense GPT/LLAMA-style default estimator for agg and disagg serving across TRT-LLM, vLLM, and SGLang-style backends
- native traditional GPT/LLAMA-style MoE agg and disagg search for TRT-LLM-style backends
- CLI shims:
  - `aic-rust cli support`
  - `aic-rust cli generate`
  - `aic-rust cli default`
  - `aic-rust query-gemm`
  - `aic-rust query-context-attention`
  - `aic-rust query-generation-attention`
  - `aic-rust model-info`

`aic-rust cli default` now runs natively for dense GPT/LLAMA-style models and traditional
GPT/LLAMA-style MoE agg/disagg searches. The Python install is still used only by parity tests as
an oracle. Remaining native work is centered on full specialized model-family graphs such as
DeepSeek/Kimi MLA, DSA, Mamba, GDN, SGLang/DeepEP/WideEP paths, and deployment artifact rendering.

## Setup

From this directory:

```bash
. "$HOME/.cargo/env"
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e ..
```

Optional environment overrides:

```bash
export AIC_PYTHON_REPO=/path/to/aiconfigurator
export AIC_SYSTEMS_ROOT=/path/to/aiconfigurator/src/aiconfigurator/systems
export AIC_MODEL_CONFIGS_ROOT=/path/to/aiconfigurator/src/aiconfigurator/model_configs
```

## Test

```bash
. "$HOME/.cargo/env"
cargo test
```

Python parity tests are opt-in because they need the Python package and its dependencies:

```bash
. "$HOME/.cargo/env"
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e ..
cargo test --features python-parity
```

The integration tests in `tests/sdk_parity.rs` call Python and assert exact parity for the
ported pieces. Set `AIC_PYTHON_BIN=/path/to/python` if you want to use a specific Python
environment.

## Examples

```bash
cargo run -- cli support \
  --model-path Qwen/Qwen3-32B-FP8 \
  --system h200_sxm \
  --backend trtllm \
  --json

cargo run -- cli generate \
  --model-path Qwen/Qwen3-32B-FP8 \
  --total-gpus 8 \
  --system h200_sxm \
  --backend trtllm \
  --json

cargo run -- query-gemm \
  --system h200_sxm \
  --backend trtllm \
  --version 1.2.0rc5 \
  --m 8192 --n 65536 --k 10240 \
  --quant int4_wo

cargo run -- query-context-attention \
  --system h200_sxm \
  --backend trtllm \
  --version 1.2.0rc5 \
  --batch-size 8 \
  --seq-len 16384 \
  --num-heads 96 \
  --num-kv-heads 1 \
  --kv-cache-quant bfloat16 \
  --fmha-quant bfloat16 \
  --window-size 128 \
  --head-size 64

cargo run -- cli default \
  --model-path Qwen/Qwen3-0.6B \
  --total-gpus 1 \
  --system h200_sxm \
  --backend trtllm \
  --top-n 1 \
  --ttft 2000 \
  --tpot 30 \
  --isl 128 \
  --osl 32 \
  --no-color
```

## Benchmarks

```bash
. "$HOME/.cargo/env"
cargo build --release
tools/compare.py
tools/benchmark_default.py
```

`tools/benchmark_default.py` runs ten full `aiconfigurator cli default ...` commands and the
same ten commands through `aic-rust cli default ...`, then compares the parsed summary metrics
and wall-clock time.
