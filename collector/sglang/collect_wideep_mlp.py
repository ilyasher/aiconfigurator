# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Wideep MLP (Shared Expert) Collector for SGLang.

Profiles the shared expert MLP forward pass used in DeepSeek V2/V3 models.
Tests both prefill (context, direct execution) and decode (generation, CUDA Graph)
phases with FP8 block quantization.

Output files:
    wideep_context_mlp_perf.txt   — prefill phase performance data
    wideep_generation_mlp_perf.txt — decode phase performance data

Usage:
    # Direct mode
    SGLANG_LOAD_FORMAT=dummy SGLANG_TEST_NUM_LAYERS=2 \
        python collect_wideep_mlp.py --device cuda:0 --output-path /path/to/output/

    # Framework mode (via collect.py)
    python collect.py --backend sglang --ops wideep_mlp_context wideep_mlp_generation
"""

import argparse
import gc
import json
import os
import subprocess
import sys
import traceback
from importlib.metadata import version as get_version

import numpy as np
import torch

try:
    from helper import benchmark_with_power, get_sm_version, log_perf
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from helper import benchmark_with_power, get_sm_version, log_perf


# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

# DeepSeek-V3 shared expert dimensions
DEFAULT_HIDDEN_SIZE = 7168
DEFAULT_INTERMEDIATE_SIZE = 2048

# Token counts to sweep — matches README specification
_NUM_TOKENS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]

# Default model path — can be overridden via DEEPSEEK_MODEL_PATH env var.
# Use HuggingFace model ID so _resolve_local_model_path() finds the cached
# config in src/aiconfigurator/model_configs/ (CI/sample tests have no
# /deepseek-v3 mount).
DEFAULT_MODEL_PATH = os.environ.get("DEEPSEEK_MODEL_PATH", "deepseek-ai/DeepSeek-V3")

# AIC's cached HuggingFace model configs — avoids HF downloads in CI.
_MODEL_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src",
    "aiconfigurator",
    "model_configs",
)


def _resolve_local_model_path(model_id: str) -> str:
    """Resolve a HuggingFace model ID to a local config directory.

    Reuses the same patching logic as collect_mla_module.py for sglang
    AutoConfig compatibility.
    """
    config_file = os.path.join(_MODEL_CONFIG_DIR, f"{model_id.replace('/', '--')}_config.json")
    if not os.path.exists(config_file):
        return model_id

    import tempfile

    with open(config_file) as f:
        config = json.load(f)

    # Normalise model_type so sglang's AutoConfig recognises it.
    if config.get("model_type") in ("deepseek_v32", "glm_moe_dsa"):
        config["architectures"] = ["DeepseekV3ForCausalLM"]
        config["model_type"] = "deepseek_v3"

    # Strip auto_map to prevent remote code download.
    config.pop("auto_map", None)

    tmp_dir = os.path.join(
        tempfile.gettempdir(),
        f"aic_sglang_mlp_config_{model_id.replace('/', '_')}_{os.getpid()}",
    )
    os.makedirs(tmp_dir, exist_ok=True)
    with open(os.path.join(tmp_dir, "config.json"), "w") as f:
        json.dump(config, f)

    return tmp_dir


# ═══════════════════════════════════════════════════════════════════════
# Test Case Generation
# ═══════════════════════════════════════════════════════════════════════


def get_wideep_mlp_context_test_cases():
    """collect.py entrypoint for wideep MLP context (prefill) collection.

    Returns list of [num_tokens, hidden_size, intermediate_size, perf_filename].
    """
    cases = []
    for num_tokens in _NUM_TOKENS:
        cases.append([num_tokens, DEFAULT_HIDDEN_SIZE, DEFAULT_INTERMEDIATE_SIZE, "wideep_context_mlp_perf.txt"])
    return cases


def get_wideep_mlp_generation_test_cases():
    """collect.py entrypoint for wideep MLP generation (decode) collection.

    Returns list of [num_tokens, hidden_size, intermediate_size, perf_filename].
    """
    cases = []
    for num_tokens in _NUM_TOKENS:
        cases.append([num_tokens, DEFAULT_HIDDEN_SIZE, DEFAULT_INTERMEDIATE_SIZE, "wideep_generation_mlp_perf.txt"])
    return cases


# ═══════════════════════════════════════════════════════════════════════
# Model Loading
# ═══════════════════════════════════════════════════════════════════════


def _load_model_runner(model_path: str, device: str = "cuda:0"):
    """Load SGLang ModelRunner with dummy weights for MLP benchmarking."""
    import random

    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.utils import suppress_other_loggers

    suppress_other_loggers()

    device_str = str(device)
    gpu_id = int(device_str.split(":")[-1]) if ":" in device_str else 0

    num_layers = int(os.environ.get("SGLANG_TEST_NUM_LAYERS", "2"))
    load_format = os.environ.get("SGLANG_LOAD_FORMAT", "dummy")

    local_model_path = _resolve_local_model_path(model_path)

    server_args = ServerArgs(
        model_path=local_model_path,
        dtype="auto",
        device="cuda",
        load_format=load_format,
        tp_size=1,
        trust_remote_code=True,
        mem_fraction_static=0.5,
        disable_radix_cache=True,
        disable_cuda_graph=True,
    )

    # Disable auto-quantization for dummy weights
    server_args.quantization = None
    server_args.enable_piecewise_cuda_graph = False

    if num_layers > 0 and load_format == "dummy":
        override_args = {"num_hidden_layers": num_layers}
        server_args.json_model_override_args = json.dumps(override_args)

    _set_envs_and_config(server_args)

    nccl_port = 29500 + random.randint(0, 10000) + gpu_id * 100

    model_config = ModelConfig.from_server_args(server_args)
    model_runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=0.5,
        gpu_id=gpu_id,
        tp_rank=gpu_id,
        tp_size=server_args.tp_size,
        pp_rank=0,
        pp_size=1,
        moe_ep_rank=0,
        moe_ep_size=1,
        nccl_port=nccl_port,
        server_args=server_args,
    )

    return model_runner


def _get_shared_experts(model_runner, test_layer: int = 0):
    """Extract the shared_experts MLP module from the model.

    DeepSeek V2/V3 MoE layers have a `shared_experts` attribute that is a
    DeepseekV2MLP instance. Falls back to the full MLP if shared_experts
    is not found (non-MoE layers).
    """
    layer = model_runner.model.model.layers[test_layer]
    mlp = layer.mlp

    # MoE layers have shared_experts
    if hasattr(mlp, "shared_experts") and mlp.shared_experts is not None:
        return mlp.shared_experts

    # For non-MoE layers or direct MLP layers
    return mlp


# ═══════════════════════════════════════════════════════════════════════
# Core Benchmarking
# ═══════════════════════════════════════════════════════════════════════


def _benchmark_mlp_prefill(
    shared_experts,
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_warmup: int,
    num_iterations: int,
    device: str,
    output_path: str | None,
):
    """Benchmark MLP in prefill (context) mode — direct execution without CUDA Graph."""
    print(f"\nPrefill: num_tokens={num_tokens}")

    try:
        hidden_states = torch.randn(
            num_tokens,
            hidden_size,
            dtype=torch.bfloat16,
            device=device,
        )

        # Warmup
        for _ in range(num_warmup):
            with torch.no_grad():
                _ = shared_experts(hidden_states)

        # Timed runs
        cuda_times = []
        for i in range(num_iterations):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            with torch.no_grad():
                _ = shared_experts(hidden_states)
            end_event.record()
            torch.cuda.synchronize()
            if i > 1:  # Skip first 2 for stability
                cuda_times.append(start_event.elapsed_time(end_event))

        avg_time_ms = np.mean(cuda_times)

        # Log perf
        try:
            version = get_version("sglang")
            device_name = torch.cuda.get_device_name(device)
            perf_filename = _resolve_perf_path(output_path, "wideep_context_mlp_perf.txt")
            log_perf(
                item_list=[
                    {
                        "quant_type": "fp8_block",
                        "num_token": num_tokens,
                        "hidden_size": hidden_size,
                        "intermediate_size": intermediate_size,
                        "avg_ms": f"{avg_time_ms:.4f}",
                    }
                ],
                framework="SGLang",
                version=version,
                device_name=device_name,
                op_name="mlp_context",
                kernel_source="shared_expert",
                perf_filename=perf_filename,
            )
        except Exception as e:
            print(f"  Warning: failed to log prefill MLP metrics: {e}")

        print(
            f"  Prefill: {avg_time_ms:.3f} ms "
            f"(min: {np.min(cuda_times):.3f}, max: {np.max(cuda_times):.3f}, "
            f"std: {np.std(cuda_times):.3f})"
        )

    except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError):
        print(f"  OOM: num_tokens={num_tokens} — skipping")
        torch.cuda.empty_cache()
        return
    except Exception as e:
        traceback.print_exc()
        error_str = str(e).lower()
        if "out of memory" in error_str:
            print(f"  OOM: num_tokens={num_tokens} — skipping")
            torch.cuda.empty_cache()
            return
        if "cuda" in error_str and "illegal" in error_str:
            print("  CUDA illegal access detected — stopping to prevent cascading failures")
            raise
        print("  Skipping this configuration...")
        return
    finally:
        torch.cuda.empty_cache()


def _benchmark_mlp_decode(
    shared_experts,
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_warmup: int,
    num_iterations: int,
    device: str,
    output_path: str | None,
):
    """Benchmark MLP in decode (generation) mode — with CUDA Graph via benchmark_with_power."""
    print(f"\nDecode: num_tokens={num_tokens}")

    try:
        hidden_states = torch.randn(
            num_tokens,
            hidden_size,
            dtype=torch.bfloat16,
            device=device,
        )

        def kernel_func():
            _ = shared_experts(hidden_states)

        with benchmark_with_power(
            device=device,
            kernel_func=kernel_func,
            num_warmups=num_warmup,
            num_runs=num_iterations,
            repeat_n=1,
        ) as results:
            pass

        avg_time_ms = results["latency_ms"]
        power_stats = results["power_stats"]

        # Log perf
        try:
            version = get_version("sglang")
            device_name = torch.cuda.get_device_name(device)
            perf_filename = _resolve_perf_path(output_path, "wideep_generation_mlp_perf.txt")
            log_perf(
                item_list=[
                    {
                        "quant_type": "fp8_block",
                        "num_token": num_tokens,
                        "hidden_size": hidden_size,
                        "intermediate_size": intermediate_size,
                        "avg_ms": f"{avg_time_ms:.4f}",
                    }
                ],
                framework="SGLang",
                version=version,
                device_name=device_name,
                op_name="mlp_generation",
                kernel_source="shared_expert",
                perf_filename=perf_filename,
                power_stats=power_stats,
            )
        except Exception as e:
            print(f"  Warning: failed to log decode MLP metrics: {e}")

        print(f"  Decode: {avg_time_ms:.3f} ms")

    except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError):
        print(f"  OOM: num_tokens={num_tokens} — skipping")
        torch.cuda.empty_cache()
        return
    except Exception as e:
        traceback.print_exc()
        error_str = str(e).lower()
        if "out of memory" in error_str:
            print(f"  OOM: num_tokens={num_tokens} — skipping")
            torch.cuda.empty_cache()
            return
        if "cuda" in error_str and "illegal" in error_str:
            print("  CUDA illegal access detected — stopping to prevent cascading failures")
            raise
        print("  Skipping this configuration...")
        return
    finally:
        torch.cuda.empty_cache()


def _resolve_perf_path(output_path: str | None, filename: str) -> str:
    """Resolve the full path for a perf output file."""
    if output_path is not None:
        return os.path.join(output_path, filename)
    collector_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(collector_dir, filename)


# ═══════════════════════════════════════════════════════════════════════
# Orchestration
# ═══════════════════════════════════════════════════════════════════════


def run_mlp_benchmark(
    model_path: str,
    is_prefill: bool,
    gpu_id: int,
    output_path: str | None = None,
):
    """Run MLP benchmark — called inside a subprocess.

    Loads the model, extracts shared_experts, and benchmarks all num_tokens configs.
    """
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)

    phase = "Context" if is_prefill else "Generation"
    print(f"\n{'=' * 60}")
    print(f"MLP {phase}: model={model_path}, GPU={gpu_id}")
    print(f"{'=' * 60}")

    try:
        model_runner = _load_model_runner(model_path=model_path, device=device)
        shared_experts = _get_shared_experts(model_runner, test_layer=0)

        # Get actual dimensions from the loaded model
        hf_config = model_runner.model_config.hf_config
        hidden_size = hf_config.hidden_size
        intermediate_size = getattr(hf_config, "moe_intermediate_size", DEFAULT_INTERMEDIATE_SIZE)
        n_shared_experts = getattr(hf_config, "n_shared_experts", 1)
        effective_intermediate_size = intermediate_size * n_shared_experts

        print(f"  hidden_size={hidden_size}, intermediate_size={effective_intermediate_size}")

        benchmark_func = _benchmark_mlp_prefill if is_prefill else _benchmark_mlp_decode

        for num_tokens in _NUM_TOKENS:
            benchmark_func(
                shared_experts=shared_experts,
                num_tokens=num_tokens,
                hidden_size=hidden_size,
                intermediate_size=effective_intermediate_size,
                num_warmup=3,
                num_iterations=10,
                device=device,
                output_path=output_path,
            )

    finally:
        torch.cuda.empty_cache()
        gc.collect()


def _run_mlp_subprocess(
    model_path: str,
    is_prefill: bool,
    gpu_id: int,
    output_path: str | None = None,
):
    """Run MLP benchmark in a subprocess with CUDA_VISIBLE_DEVICES isolation."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    output_repr = f'"{output_path}"' if output_path else "None"
    code = (
        f'import sys; sys.path.insert(0, "{os.path.dirname(os.path.abspath(__file__))}")\n'
        f"from collect_wideep_mlp import run_mlp_benchmark\n"
        f'run_mlp_benchmark("{model_path}", {is_prefill}, 0, {output_repr})\n'
    )

    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )

    try:
        stdout, _ = proc.communicate(timeout=1800)  # 30 min timeout
        if stdout:
            print(stdout.decode("utf-8", errors="replace"))
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

    if proc.returncode != 0:
        phase = "context" if is_prefill else "generation"
        raise RuntimeError(f"MLP {phase} subprocess failed (exit code {proc.returncode})")


def run_wideep_mlp(num_tokens, hidden_size, intermediate_size, perf_filename, device="cuda:0"):
    """Worker-compatible positional wrapper used by collector/collect.py.

    Each call runs ALL num_tokens combos for the specified phase in a subprocess.
    The individual num_tokens, hidden_size, and intermediate_size args are ignored
    because the subprocess sweeps all combos internally. This is called once per
    test case but deduplicates via subprocess — only the first call triggers work.
    """
    device_str = str(device) if not isinstance(device, str) else device
    gpu_id = int(device_str.split(":")[-1]) if ":" in device_str else 0
    is_prefill = "context" in perf_filename

    phase = "Context" if is_prefill else "Generation"
    print(f"\n{'=' * 60}")
    print(
        f"MLP {phase}: num_tokens={num_tokens}, hidden_size={hidden_size}, "
        f"intermediate_size={intermediate_size}, GPU={gpu_id}"
    )
    print(f"{'=' * 60}")

    # Resolve output directory
    output_path = os.path.dirname(perf_filename) or os.getcwd()

    model_path = os.environ.get("DEEPSEEK_MODEL_PATH", DEFAULT_MODEL_PATH)

    _run_mlp_subprocess(
        model_path=model_path,
        is_prefill=is_prefill,
        gpu_id=gpu_id,
        output_path=output_path,
    )


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="SGLang Wideep MLP (Shared Expert) Benchmark")
    parser.add_argument("--device", default="cuda:0", help="CUDA device")
    parser.add_argument("--output-path", default=None, help="Output directory for perf files")
    args = parser.parse_args()

    model_path = os.environ.get("DEEPSEEK_MODEL_PATH", DEFAULT_MODEL_PATH)
    gpu_id = int(args.device.split(":")[-1]) if ":" in args.device else 0

    print(f"Model path: {model_path}")

    # Prefill phase
    print(f"\n{'=' * 60}")
    print("PREFILL PHASE")
    print(f"{'=' * 60}")
    _run_mlp_subprocess(
        model_path=model_path,
        is_prefill=True,
        gpu_id=gpu_id,
        output_path=args.output_path,
    )

    # Decode phase
    print(f"\n{'=' * 60}")
    print("DECODE PHASE")
    print(f"{'=' * 60}")
    _run_mlp_subprocess(
        model_path=model_path,
        is_prefill=False,
        gpu_id=gpu_id,
        output_path=args.output_path,
    )

    print(f"\n{'=' * 60}")
    print("ALL TESTS COMPLETED")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
