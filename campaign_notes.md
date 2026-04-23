## Learnings

## Initial Review

### Review scope
Reviewed all 9 ops for vllm 0.19.1 compatibility: gemm, attention_context,
attention_generation, moe, mla_context_module, mla_generation_module,
dsa_context_module, dsa_generation_module, gdn.

### Framework source: vllm v0.19.1 (commit b1388b1)

### Op-by-op findings

#### gemm (collect_gemm.py, compat=0.14.0)
- **Fp8Config** constructor: signature unchanged (is_checkpoint_fp8_serialized, activation_scheme, ignored_layers, weight_block_size). Compatible.
- **RowParallelLinear** constructor: unchanged. Compatible.
- **per_block_cast_to_fp8**: signature unchanged (x, block_size, use_ue8m0). Compatible.
- **maybe_post_process_fp8_weight_block**: now takes only 1 arg (layer). Collector already has try/except for old 2-arg signature. Compatible.
- **CompressedTensorsConfig.from_config**: unchanged. Compatible.
- No semantic changes to kernel selection or default paths.
- **Status: OK, no changes needed.**

#### attention_context / attention_generation (collect_attn.py, compat=0.11.0)
- **get_attn_backend_cls**: signature is now `(selected_backend, attn_selector_config, num_heads=None)`. Collector's first 3 tries (old positional signatures) will fail with TypeError; 4th try uses AttentionSelectorConfig — works correctly.
- **AttentionSelectorConfig**: added `use_mm_prefix`, `use_per_head_quant_scales`, `attn_type` fields (all have defaults). Collector creates config without these — compatible.
- **Backend priorities on SM100+**: FLASHINFER now prioritized over FLASH_ATTN. Collector lets vLLM choose the backend, so it correctly benchmarks production code path.
- **FLEX_ATTENTION_SLOW**: still handled by collector's special-case code.
- **set_kv_cache_layout**: still present at same path.
- **CommonAttentionMetadata**: `seq_lens_cpu` and `num_computed_tokens_cpu` are now deprecated properties wrapping `_seq_lens_cpu` / `_num_computed_tokens_cpu`. utils.py creates the metadata with `_seq_lens_cpu` (first try/except branch). Access via property still works.
- **Status: OK, no changes needed.** All compatibility guards function correctly.

#### moe (collect_moe_v2.py active for >=0.17.0, compat=0.17.0)
- **determine_expert_map**: added optional `expert_placement_strategy`, `num_fused_shared_experts`, `return_expert_mask` params — all have defaults. Collector's 3-positional-arg call `(moe_ep_size, 0, num_experts)` still works.
- **fused_experts**: added `activation`, `apply_router_weight_on_input` params — both have defaults. Collector doesn't pass these; default `SILU` is correct for standard MoE. MXFP4 path uses FusedMoE module which handles activation internally.
- **fp8_w8a8_moe_quant_config**: added optional `w1_bias`, `w2_bias`, `per_act_token_quant`, `per_out_ch_quant`, `a1_gscale`, `a2_gscale`, `g1_alphas`, `g2_alphas` — all optional. Collector's existing call is compatible.
- (collect_moe_v1.py only used for vllm < 0.17.0, not active for this campaign)
- **Status: OK, no changes needed.**

#### mla_context_module / mla_generation_module (collect_mla_module_v2.py active for >=0.17.0, compat=0.17.0)
- **DeepseekV2MLAAttention**: added optional `input_size: int | None = None` param. Collector doesn't pass it — defaults to None → hidden_size internally. Compatible.
- **MLAAttention.process_weights_after_loading(act_dtype)**: signature unchanged.
- **MLAAttentionSpec**: unchanged (`cache_dtype_str` still present).
- **DeepseekV32IndexerBackend**: still at same import path.
- **_get_attention_backend**: uses `AttentionSelectorConfig` which is compatible with 0.19.1.
- **Weight init**: v2 already uses `fill_()` instead of `normal_()` to avoid CUDA graph RNG crash. Correct.
- **auto_map stripping**: v2 already strips `auto_map` from config.json. Correct.
- (collect_mla_module_v1.py only used for vllm < 0.17.0, not active)
- **Status: OK, no changes needed.**

#### dsa_context_module / dsa_generation_module
- Same collector as MLA (collect_mla_module_v2.py), just filtered by attn_type="dsa".
- All DSA-specific code paths (fp8_ds_mla cache, indexer, topk_indices_buffer) use APIs unchanged in 0.19.1.
- **Status: OK, no changes needed.**

#### gdn (collect_gdn.py, compat=0.0.0)
- Stub implementation, raises NotImplementedError.
- **Status: Stub, not implemented.**

### Summary
All active collector scripts (v2 variants used for vllm >=0.17.0) are fully compatible
with vllm 0.19.1 without any changes. The framework's API evolution has been
backward-compatible for all functions/classes the collectors use. No semantic changes
detected that would affect benchmark accuracy.

## User preferences
