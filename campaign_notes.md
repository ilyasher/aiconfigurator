## Learnings

- `collector/sglang/collect_gemm.py`: `from common_test_cases import ...` fails when module is imported as `collector.sglang.collect_gemm` because `collector/` isn't on sys.path. Fixed with try/except + sys.path.append pattern (same as collect_moe.py). The `from helper import ...` on a later line is also affected but works after the sys.path fix in the except branch.

## User preferences
