## Learnings

- `collector/sglang/collect_gemm.py`: `from common_test_cases import ...` fails when module is imported as `collector.sglang.collect_gemm` because `collector/` isn't on sys.path. Fixed with try/except + sys.path.append pattern (same as collect_moe.py). The `from helper import ...` on a later line is also affected but works after the sys.path fix in the except branch.

- `collector/sglang/collect_attn.py`: `from helper import ...` fails with `ModuleNotFoundError: No module named 'helper'` when imported as `collector.sglang.collect_attn`. Fixed with the standard try/except + sys.path.append pattern. Note: `os` was already imported so no extra import needed in the except branch (only `sys`).

- `collector/sglang/collect_mla.py`: `from helper import ...` fails with `ModuleNotFoundError: No module named 'helper'` when imported as `collector.sglang.collect_mla`. Fixed with the standard try/except + sys.path.append pattern. `os` was already imported so only `sys` needed in the except branch.

## User preferences
