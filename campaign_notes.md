## Initial Review

### attention_context / attention_generation (collect_attn.py, compat: sglang>=0.5.5)

**Framework changes (v0.5.5 → v0.5.10):**
- Only one commit touched the attention/dp_attention/forward_batch area: `1519acf [Hotfix] Fix router gemm on sm103`.
- `dp_attention.py`: Removed module-level `_ATTN_TP_GROUP`, `_ATTN_TP_RANK`, `_ATTN_TP_SIZE` variables. Functions `get_attention_tp_rank()` and `get_attention_tp_size()` now delegate to distributed module (`get_attn_tensor_model_parallel_rank()` etc.) instead of reading module vars. Our lambda monkeypatches still work since they replace the function on the module object before the backend modules import it.
- `dp_attention.py`: Added context parallelism functions: `get_attention_cp_size()`, `get_attention_cp_rank()`, `get_attention_cp_group()`.
- `FlashAttentionBackend.__init__`: Now accesses `model_runner.attn_cp_size` (context parallelism size). **This was missing from MockModelRunner and would cause AttributeError.**
- `FlashAttentionBackend.__init__`: Flash attention imports (`flash_attn_varlen_func`, `flash_attn_with_kvcache`) moved from module level into `__init__` (conditional on `fa_impl_ver`). No impact on collector.
- `server_args.py`: Many new attributes added (`attn_cp_size`, `moe_dp_size`, `fastapi_root_path`, SSL options, etc.). The mock only needs `attn_cp_size` since FlashAttentionBackend accesses it.

**Fix applied:**
1. Added `self.attn_cp_size = 1` to `MockModelRunner` (backwards-compatible; harmless on older versions).
2. Added conditional monkeypatches for `dp_attention.get_attention_cp_size` and `get_attention_cp_rank` (guarded by `hasattr` for backward compat).

**No semantic changes detected:** The attention backends still use the same kernels (FlashAttention v3, Triton, TRTLLM MHA) with the same selection logic. No default backend changes.

### mla_bmm_gen_pre / mla_bmm_gen_post (collect_mla_bmm.py, no compat string)

**Framework changes (v0.5.5 → v0.5.10):**
- `fp8_kernel.py`: Internal refactoring (custom op registration changed from `direct_register_custom_op` to `@register_custom_op` decorator, `sgl_per_tensor_quant_fp8` import path changed internally). The public API `per_tensor_quant_mla_fp8` function is unchanged — still exists at the same import path with the same signature.
- `sgl_kernel.bmm_fp8`: This is from the `sgl_kernel` package, not sglang itself. No changes in sglang's use of it.

**No fix needed.** The collector imports `per_tensor_quant_mla_fp8` and `bmm_fp8` which are both stable. No mock objects are used. No semantic changes to the benchmarked kernels.

### attention_context - MockServerArgs missing disable_piecewise_cuda_graph

**Root cause:** sglang 0.5.10's `flashinfer_backend.py` accesses `server_args.disable_piecewise_cuda_graph` (negative form), but MockServerArgs only had `enable_piecewise_cuda_graph` (positive form). The attribute was renamed/added upstream.

**Fix:** Added `self.disable_piecewise_cuda_graph = True` to MockServerArgs alongside the existing `enable_piecewise_cuda_graph = False` for backward compatibility. Both attributes are semantically consistent (disabled = True, enabled = False).

### mla_bmm_gen_pre / mla_bmm_gen_post - FP8 not supported on A100 (SM 80)

**Root cause:** The `get_mla_gen_pre_test_cases()` and `get_mla_gen_post_test_cases()` functions unconditionally generate FP8 test cases. On A100 (SM 80), the Triton compiler rejects `fp8e4nv` because native FP8 requires SM ≥ 89 (Ada Lovelace / Hopper). This caused 50 CompilationError failures on a100_sxm.

**Fix:** Added SM version check using `get_sm_version()` from helper — only include `"fp8"` in dtype_list when `sm_version >= 89`. Applied to both test case generator functions.

## Learnings

## User preferences
