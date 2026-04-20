## Learnings

### collect_attn.py: `from helper import ...` fails when imported as `collector.sglang.collect_attn`
- **Root cause**: `helper.py` lives at `collector/helper.py`. When the module is imported via `importlib.import_module('collector.sglang.collect_attn')`, Python can't find `helper` as a top-level module because `collector/` isn't on `sys.path`.
- **Fix**: Wrap the import in a try/except and add `sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))` on failure, matching the pattern used in `collect_mla_module.py`, `collect_moe.py`, and other sglang collectors.
- **Note**: Several other sglang collectors (`collect_gemm.py`, `collect_mla.py`, `collect_mla_bmm.py`) have the same bare `from helper import` pattern and may also fail under the same import conditions.

## User preferences
