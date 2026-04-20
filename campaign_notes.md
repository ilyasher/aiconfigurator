## Learnings

### collect_attn.py: `from helper import ...` fails when imported as `collector.sglang.collect_attn`
- **Root cause**: `helper.py` lives at `collector/helper.py`. When the module is imported via `importlib.import_module('collector.sglang.collect_attn')`, Python can't find `helper` as a top-level module because `collector/` isn't on `sys.path`.
- **Fix**: Wrap the import in a try/except and add `sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))` on failure, matching the pattern used in `collect_mla_module.py`, `collect_moe.py`, and other sglang collectors.
- **Note**: Several other sglang collectors (`collect_gemm.py`, `collect_mla.py`, `collect_mla_bmm.py`) have the same bare `from helper import` pattern and may also fail under the same import conditions.

### collect_attn.py: `MockServerArgs` missing `disable_piecewise_cuda_graph` attribute
- **Root cause**: In sglang 0.5.10, `flashinfer_backend.py` (line 252) accesses `server_args.disable_piecewise_cuda_graph`. The collector's `MockServerArgs` had the old/wrong attribute name `enable_piecewise_cuda_graph` (inverted naming convention).
- **Fix**: Renamed `enable_piecewise_cuda_graph` → `disable_piecewise_cuda_graph` in `MockServerArgs.__init__()` (line 87). Value stays `False` (meaning piecewise cuda graph is enabled, matching the framework's default).
- **Note**: Python's "Did you mean" suggestion was correct in this case, but the fix was verified against `sglang/srt/server_args.py` (line 637: `disable_piecewise_cuda_graph: bool = False`).

## User preferences
