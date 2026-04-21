## Learnings

- `collector/sglang/collect_gemm.py`: `from common_test_cases import ...` fails when module is imported as `collector.sglang.collect_gemm` because `collector/` isn't on sys.path. Fixed with try/except + sys.path.append pattern (same as collect_moe.py). The `from helper import ...` on a later line is also affected but works after the sys.path fix in the except branch.

- `collector/sglang/collect_attn.py`: `from helper import ...` fails with `ModuleNotFoundError: No module named 'helper'` when imported as `collector.sglang.collect_attn`. Fixed with the standard try/except + sys.path.append pattern. Note: `os` was already imported so no extra import needed in the except branch (only `sys`).

- `collector/sglang/collect_mla.py`: `from helper import ...` fails with `ModuleNotFoundError: No module named 'helper'` when imported as `collector.sglang.collect_mla`. Fixed with the standard try/except + sys.path.append pattern. `os` was already imported so only `sys` needed in the except branch.

- `collector/sglang/collect_wideep_mlp.py`: This file was missing entirely — the README documented it and the campaign expected `wideep_mlp_context`/`wideep_mlp_generation` ops but neither the module nor registry entries existed. Created `collect_wideep_mlp.py` following the same patterns as `collect_wideep_deepep_moe.py` and `collect_mla_module.py`: subprocess isolation via `CUDA_VISIBLE_DEVICES`, dummy weight loading via `_resolve_local_model_path()`, the standard try/except `from helper import ...` pattern. The collector benchmarks the `shared_experts` MLP layer (DeepseekV2MLP) from DeepSeek V2/V3 MoE layers. Added two `OpEntry` entries to `registry.py` mapping `wideep_mlp_context` and `wideep_mlp_generation` to the new module's `get_wideep_mlp_context_test_cases`/`get_wideep_mlp_generation_test_cases` and `run_wideep_mlp` functions.

- `wideep_mlp_context` and `wideep_mlp_generation` ops: Re-added to `collector/sglang/registry.py` (attempt 4). The module `collect_wideep_mlp.py` already existed with `get_wideep_mlp_context_test_cases`, `get_wideep_mlp_generation_test_cases`, and `run_wideep_mlp` functions. The registry just needed OpEntry entries pointing to them.

## User preferences

