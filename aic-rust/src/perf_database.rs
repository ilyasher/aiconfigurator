use crate::system::SystemSpec;
use crate::types::{
    BackendName, CommQuantMode, DatabaseMode, FmhaQuantMode, GemmQuantMode, KvCacheQuantMode,
    MoeQuantMode, PerformanceResult,
};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

type KMap = BTreeMap<i64, PerformanceResult>;
type NMap = BTreeMap<i64, KMap>;
type MMap = BTreeMap<i64, NMap>;
type ContextAttentionData = BTreeMap<
    FmhaQuantMode,
    BTreeMap<KvCacheQuantMode, BTreeMap<i64, BTreeMap<i64, BTreeMap<i64, MMap>>>>,
>;
type GenerationAttentionData =
    BTreeMap<KvCacheQuantMode, BTreeMap<i64, BTreeMap<i64, BTreeMap<i64, MMap>>>>;
type ContextMlaData = BTreeMap<FmhaQuantMode, BTreeMap<KvCacheQuantMode, MMap>>;
type GenerationMlaData = BTreeMap<KvCacheQuantMode, MMap>;
type ContextMlaModuleData =
    BTreeMap<FmhaQuantMode, BTreeMap<KvCacheQuantMode, BTreeMap<GemmQuantMode, MMap>>>;
type GenerationMlaModuleData =
    BTreeMap<FmhaQuantMode, BTreeMap<KvCacheQuantMode, BTreeMap<GemmQuantMode, MMap>>>;
type CommSizeMap = BTreeMap<i64, PerformanceResult>;
type CustomAllreduceData = BTreeMap<CommQuantMode, BTreeMap<u32, BTreeMap<String, CommSizeMap>>>;
type NcclData = BTreeMap<CommQuantMode, BTreeMap<String, BTreeMap<u32, CommSizeMap>>>;
type MoeTokenData = BTreeMap<i64, PerformanceResult>;
type MoeData = BTreeMap<
    MoeQuantMode,
    BTreeMap<
        String,
        BTreeMap<
            i64,
            BTreeMap<i64, BTreeMap<i64, BTreeMap<i64, BTreeMap<u32, BTreeMap<u32, MoeTokenData>>>>>,
        >,
    >,
>;

#[derive(Debug, Clone)]
pub struct PerfDatabase {
    pub system: String,
    pub backend: BackendName,
    pub version: String,
    pub systems_root: PathBuf,
    pub system_spec: SystemSpec,
    gemm_data: BTreeMap<GemmQuantMode, MMap>,
    context_attention_data: ContextAttentionData,
    generation_attention_data: GenerationAttentionData,
    context_mla_data: ContextMlaData,
    generation_mla_data: GenerationMlaData,
    context_mla_module_data: ContextMlaModuleData,
    generation_mla_module_data: GenerationMlaModuleData,
    custom_allreduce_data: CustomAllreduceData,
    nccl_data: NcclData,
    moe_data: MoeData,
    moe_low_latency_data: MoeData,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GemmRow {
    gemm_dtype: String,
    m: i64,
    n: i64,
    k: i64,
    latency: f64,
    #[serde(default)]
    power: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ContextAttentionRow {
    batch_size: i64,
    isl: i64,
    num_heads: i64,
    num_key_value_heads: i64,
    head_dim: i64,
    #[serde(default)]
    window_size: Option<i64>,
    attn_dtype: String,
    kv_cache_dtype: String,
    latency: f64,
    #[serde(default)]
    power: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GenerationAttentionRow {
    batch_size: i64,
    isl: i64,
    #[serde(rename = "step")]
    s: i64,
    num_heads: i64,
    num_key_value_heads: i64,
    head_dim: i64,
    #[serde(default)]
    window_size: Option<i64>,
    kv_cache_dtype: String,
    latency: f64,
    #[serde(default)]
    power: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ContextMlaRow {
    #[serde(default)]
    mla_dtype: String,
    kv_cache_dtype: String,
    #[serde(default)]
    num_heads: Option<i64>,
    #[serde(default)]
    tp_size: Option<i64>,
    batch_size: i64,
    isl: i64,
    latency: f64,
    #[serde(default)]
    power: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GenerationMlaRow {
    #[serde(default)]
    mla_dtype: String,
    kv_cache_dtype: String,
    #[serde(default)]
    num_heads: Option<i64>,
    #[serde(default)]
    tp_size: Option<i64>,
    batch_size: i64,
    isl: i64,
    #[serde(rename = "step")]
    s: i64,
    latency: f64,
    #[serde(default)]
    power: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MlaModuleRow {
    mla_dtype: String,
    kv_cache_dtype: String,
    gemm_type: String,
    num_heads: i64,
    batch_size: i64,
    isl: i64,
    #[serde(rename = "step")]
    s: i64,
    latency: f64,
    #[serde(default)]
    power: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MoeRow {
    #[serde(default)]
    kernel_source: Option<String>,
    moe_dtype: String,
    num_tokens: i64,
    hidden_size: i64,
    inter_size: i64,
    topk: i64,
    num_experts: i64,
    moe_tp_size: u32,
    moe_ep_size: u32,
    distribution: String,
    latency: f64,
    #[serde(default)]
    power: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CustomAllreduceRow {
    allreduce_dtype: String,
    num_gpus: u32,
    message_size: i64,
    latency: f64,
    #[serde(default)]
    power: Option<f64>,
    #[serde(default)]
    kernel_source: Option<String>,
    #[serde(default)]
    backend: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct NcclRow {
    nccl_dtype: String,
    op_name: String,
    num_gpus: u32,
    message_size: i64,
    latency: f64,
    #[serde(default)]
    power: Option<f64>,
}

impl PerfDatabase {
    pub fn load(
        systems_root: impl AsRef<Path>,
        system: &str,
        backend: BackendName,
        version: &str,
    ) -> Result<Self> {
        let systems_root = systems_root.as_ref().to_path_buf();
        let system_spec = SystemSpec::load(&systems_root, system)?;
        let data_path = system_spec.data_path(&systems_root, backend, version);
        if !data_path.is_dir() {
            bail!("database path not found: {}", data_path.display());
        }
        let gemm_data = load_gemm_data(data_path.join("gemm_perf.txt"), &system_spec)?;
        let context_attention_data =
            load_context_attention_data(data_path.join("context_attention_perf.txt"))?;
        let generation_attention_data = load_generation_attention_data(
            data_path.join("generation_attention_perf.txt"),
            &system_spec,
        )?;
        let context_mla_data = load_context_mla_data(data_path.join("context_mla_perf.txt"))?;
        let generation_mla_data =
            load_generation_mla_data(data_path.join("generation_mla_perf.txt"))?;
        let context_mla_module_data =
            load_context_mla_module_data(data_path.join("mla_context_module_perf.txt"))?;
        let generation_mla_module_data =
            load_generation_mla_module_data(data_path.join("mla_generation_module_perf.txt"))?;
        let custom_allreduce_data =
            load_custom_allreduce_data(data_path.join("custom_allreduce_perf.txt"))?;
        let (moe_data, moe_low_latency_data) = load_moe_data(data_path.join("moe_perf.txt"))?;
        let nccl_data = match system_spec
            .misc
            .as_ref()
            .and_then(|misc| misc.nccl_version.as_ref())
        {
            Some(version) => load_nccl_data(
                systems_root
                    .join(&system_spec.data_dir)
                    .join("nccl")
                    .join(version)
                    .join("nccl_perf.txt"),
            )?,
            None => BTreeMap::new(),
        };
        Ok(Self {
            system: system.to_string(),
            backend,
            version: version.to_string(),
            systems_root,
            system_spec,
            gemm_data,
            context_attention_data,
            generation_attention_data,
            context_mla_data,
            generation_mla_data,
            context_mla_module_data,
            generation_mla_module_data,
            custom_allreduce_data,
            nccl_data,
            moe_data,
            moe_low_latency_data,
        })
    }

    pub fn query_gemm(
        &self,
        m: i64,
        n: i64,
        k: i64,
        quant_mode: GemmQuantMode,
        database_mode: DatabaseMode,
    ) -> Result<PerformanceResult> {
        let table_quant_mode = normalize_gemm_quant_mode_for_table(quant_mode);
        match database_mode {
            DatabaseMode::SOL => Ok(PerformanceResult::new(
                self.gemm_sol(m, n, k, quant_mode).0,
                0.0,
            )),
            DatabaseMode::SOL_FULL => Ok(PerformanceResult::new(
                self.gemm_sol(m, n, k, quant_mode).0,
                0.0,
            )),
            DatabaseMode::EMPIRICAL => Ok(PerformanceResult::new(
                self.gemm_empirical(m, n, k, quant_mode),
                0.0,
            )),
            DatabaseMode::SILICON => self.query_gemm_silicon(m, n, k, table_quant_mode),
            DatabaseMode::HYBRID => {
                self.query_gemm_silicon(m, n, k, table_quant_mode)
                    .or_else(|_| {
                        Ok(PerformanceResult::new(
                            self.gemm_empirical(m, n, k, quant_mode),
                            0.0,
                        ))
                    })
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn query_context_attention(
        &self,
        b: i64,
        s: i64,
        prefix: i64,
        n: i64,
        n_kv: i64,
        kvcache_quant_mode: KvCacheQuantMode,
        fmha_quant_mode: FmhaQuantMode,
        database_mode: DatabaseMode,
        window_size: i64,
        head_size: i64,
    ) -> Result<PerformanceResult> {
        match database_mode {
            DatabaseMode::SOL => Ok(PerformanceResult::new(
                self.context_attention_sol(
                    b,
                    s,
                    prefix,
                    n,
                    n_kv,
                    head_size,
                    window_size,
                    kvcache_quant_mode,
                    fmha_quant_mode,
                )
                .0,
                0.0,
            )),
            DatabaseMode::SOL_FULL => Ok(PerformanceResult::new(
                self.context_attention_sol(
                    b,
                    s,
                    prefix,
                    n,
                    n_kv,
                    head_size,
                    window_size,
                    kvcache_quant_mode,
                    fmha_quant_mode,
                )
                .0,
                0.0,
            )),
            DatabaseMode::EMPIRICAL => Ok(PerformanceResult::new(
                self.context_attention_empirical(
                    b,
                    s,
                    prefix,
                    n,
                    n_kv,
                    head_size,
                    window_size,
                    kvcache_quant_mode,
                    fmha_quant_mode,
                ),
                0.0,
            )),
            DatabaseMode::SILICON => self.query_context_attention_silicon(
                b,
                s,
                prefix,
                n,
                n_kv,
                kvcache_quant_mode,
                fmha_quant_mode,
                window_size,
                head_size,
            ),
            DatabaseMode::HYBRID => self
                .query_context_attention_silicon(
                    b,
                    s,
                    prefix,
                    n,
                    n_kv,
                    kvcache_quant_mode,
                    fmha_quant_mode,
                    window_size,
                    head_size,
                )
                .or_else(|_| {
                    Ok(PerformanceResult::new(
                        self.context_attention_empirical(
                            b,
                            s,
                            prefix,
                            n,
                            n_kv,
                            head_size,
                            window_size,
                            kvcache_quant_mode,
                            fmha_quant_mode,
                        ),
                        0.0,
                    ))
                }),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn query_generation_attention(
        &self,
        b: i64,
        s: i64,
        n: i64,
        n_kv: i64,
        kvcache_quant_mode: KvCacheQuantMode,
        database_mode: DatabaseMode,
        window_size: i64,
        head_size: i64,
    ) -> Result<PerformanceResult> {
        match database_mode {
            DatabaseMode::SOL | DatabaseMode::SOL_FULL => Ok(PerformanceResult::new(
                self.generation_attention_sol(
                    b,
                    s,
                    n,
                    n_kv,
                    head_size,
                    window_size,
                    kvcache_quant_mode,
                )
                .0,
                0.0,
            )),
            DatabaseMode::EMPIRICAL => Ok(PerformanceResult::new(
                self.generation_attention_empirical(
                    b,
                    s,
                    n,
                    n_kv,
                    head_size,
                    window_size,
                    kvcache_quant_mode,
                ),
                0.0,
            )),
            DatabaseMode::SILICON => self.query_generation_attention_silicon(
                b,
                s,
                n,
                n_kv,
                kvcache_quant_mode,
                window_size,
                head_size,
            ),
            DatabaseMode::HYBRID => self
                .query_generation_attention_silicon(
                    b,
                    s,
                    n,
                    n_kv,
                    kvcache_quant_mode,
                    window_size,
                    head_size,
                )
                .or_else(|_| {
                    Ok(PerformanceResult::new(
                        self.generation_attention_empirical(
                            b,
                            s,
                            n,
                            n_kv,
                            head_size,
                            window_size,
                            kvcache_quant_mode,
                        ),
                        0.0,
                    ))
                }),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn query_context_mla(
        &self,
        b: i64,
        s: i64,
        prefix: i64,
        num_heads: i64,
        kvcache_quant_mode: KvCacheQuantMode,
        fmha_quant_mode: FmhaQuantMode,
        database_mode: DatabaseMode,
    ) -> Result<PerformanceResult> {
        match database_mode {
            DatabaseMode::SOL | DatabaseMode::SOL_FULL => Ok(PerformanceResult::new(
                self.context_mla_sol(b, s, prefix, num_heads, kvcache_quant_mode, fmha_quant_mode)
                    .0,
                0.0,
            )),
            DatabaseMode::EMPIRICAL => Ok(PerformanceResult::new(
                self.context_mla_empirical(
                    b,
                    s,
                    prefix,
                    num_heads,
                    kvcache_quant_mode,
                    fmha_quant_mode,
                ),
                0.0,
            )),
            DatabaseMode::SILICON => self.query_context_mla_silicon(
                b,
                s,
                prefix,
                num_heads,
                kvcache_quant_mode,
                fmha_quant_mode,
            ),
            DatabaseMode::HYBRID => self
                .query_context_mla_silicon(
                    b,
                    s,
                    prefix,
                    num_heads,
                    kvcache_quant_mode,
                    fmha_quant_mode,
                )
                .or_else(|_| {
                    Ok(PerformanceResult::new(
                        self.context_mla_empirical(
                            b,
                            s,
                            prefix,
                            num_heads,
                            kvcache_quant_mode,
                            fmha_quant_mode,
                        ),
                        0.0,
                    ))
                }),
        }
    }

    pub fn query_generation_mla(
        &self,
        b: i64,
        s: i64,
        num_heads: i64,
        kvcache_quant_mode: KvCacheQuantMode,
        database_mode: DatabaseMode,
    ) -> Result<PerformanceResult> {
        match database_mode {
            DatabaseMode::SOL | DatabaseMode::SOL_FULL => Ok(PerformanceResult::new(
                self.generation_mla_sol(b, s, num_heads, kvcache_quant_mode)
                    .0,
                0.0,
            )),
            DatabaseMode::EMPIRICAL => Ok(PerformanceResult::new(
                self.generation_mla_empirical(b, s, num_heads, kvcache_quant_mode),
                0.0,
            )),
            DatabaseMode::SILICON => {
                self.query_generation_mla_silicon(b, s, num_heads, kvcache_quant_mode)
            }
            DatabaseMode::HYBRID => self
                .query_generation_mla_silicon(b, s, num_heads, kvcache_quant_mode)
                .or_else(|_| {
                    Ok(PerformanceResult::new(
                        self.generation_mla_empirical(b, s, num_heads, kvcache_quant_mode),
                        0.0,
                    ))
                }),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn query_context_mla_module(
        &self,
        b: i64,
        s: i64,
        prefix: i64,
        num_heads: i64,
        kvcache_quant_mode: KvCacheQuantMode,
        fmha_quant_mode: FmhaQuantMode,
        gemm_quant_mode: GemmQuantMode,
        database_mode: DatabaseMode,
    ) -> Result<PerformanceResult> {
        match database_mode {
            DatabaseMode::SOL | DatabaseMode::SOL_FULL => Ok(PerformanceResult::new(
                self.context_mla_sol(b, s, prefix, num_heads, kvcache_quant_mode, fmha_quant_mode)
                    .0,
                0.0,
            )),
            DatabaseMode::EMPIRICAL => Ok(PerformanceResult::new(
                self.context_mla_empirical(
                    b,
                    s,
                    prefix,
                    num_heads,
                    kvcache_quant_mode,
                    fmha_quant_mode,
                ),
                0.0,
            )),
            DatabaseMode::SILICON => self.query_context_mla_module_silicon(
                b,
                s,
                prefix,
                num_heads,
                kvcache_quant_mode,
                fmha_quant_mode,
                normalize_gemm_quant_mode_for_table(gemm_quant_mode),
            ),
            DatabaseMode::HYBRID => self
                .query_context_mla_module_silicon(
                    b,
                    s,
                    prefix,
                    num_heads,
                    kvcache_quant_mode,
                    fmha_quant_mode,
                    normalize_gemm_quant_mode_for_table(gemm_quant_mode),
                )
                .or_else(|_| {
                    Ok(PerformanceResult::new(
                        self.context_mla_empirical(
                            b,
                            s,
                            prefix,
                            num_heads,
                            kvcache_quant_mode,
                            fmha_quant_mode,
                        ),
                        0.0,
                    ))
                }),
        }
    }

    pub fn query_generation_mla_module(
        &self,
        b: i64,
        s: i64,
        num_heads: i64,
        kvcache_quant_mode: KvCacheQuantMode,
        fmha_quant_mode: FmhaQuantMode,
        gemm_quant_mode: GemmQuantMode,
        database_mode: DatabaseMode,
    ) -> Result<PerformanceResult> {
        match database_mode {
            DatabaseMode::SOL | DatabaseMode::SOL_FULL => Ok(PerformanceResult::new(
                self.generation_mla_module_sol(
                    b,
                    s,
                    num_heads,
                    kvcache_quant_mode,
                    gemm_quant_mode,
                )
                .0,
                0.0,
            )),
            DatabaseMode::EMPIRICAL => Ok(PerformanceResult::new(
                self.generation_mla_module_empirical(
                    b,
                    s,
                    num_heads,
                    kvcache_quant_mode,
                    gemm_quant_mode,
                ),
                0.0,
            )),
            DatabaseMode::SILICON => self.query_generation_mla_module_silicon(
                b,
                s,
                num_heads,
                kvcache_quant_mode,
                fmha_quant_mode,
                normalize_gemm_quant_mode_for_table(gemm_quant_mode),
            ),
            DatabaseMode::HYBRID => self
                .query_generation_mla_module_silicon(
                    b,
                    s,
                    num_heads,
                    kvcache_quant_mode,
                    fmha_quant_mode,
                    normalize_gemm_quant_mode_for_table(gemm_quant_mode),
                )
                .or_else(|_| {
                    Ok(PerformanceResult::new(
                        self.generation_mla_module_empirical(
                            b,
                            s,
                            num_heads,
                            kvcache_quant_mode,
                            gemm_quant_mode,
                        ),
                        0.0,
                    ))
                }),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn query_moe(
        &self,
        num_tokens: i64,
        hidden_size: i64,
        inter_size: i64,
        topk: i64,
        num_experts: i64,
        moe_tp_size: u32,
        moe_ep_size: u32,
        quant_mode: MoeQuantMode,
        workload_distribution: &str,
        is_context: bool,
        moe_backend: Option<&str>,
        database_mode: DatabaseMode,
        is_gated: bool,
        enable_eplb: bool,
    ) -> Result<PerformanceResult> {
        let num_tokens = if self.backend == BackendName::sglang && enable_eplb && is_context {
            (num_tokens as f64 * 0.8) as i64
        } else {
            num_tokens
        };
        let sol = || {
            self.moe_sol(
                num_tokens,
                hidden_size,
                inter_size,
                topk,
                num_experts,
                moe_tp_size,
                moe_ep_size,
                quant_mode,
                is_gated,
            )
        };
        let empirical = || PerformanceResult::new(sol().0 / 0.4, 0.0);
        match database_mode {
            DatabaseMode::SOL | DatabaseMode::SOL_FULL => Ok(PerformanceResult::new(sol().0, 0.0)),
            DatabaseMode::EMPIRICAL => Ok(empirical()),
            DatabaseMode::SILICON => self.query_moe_silicon(
                num_tokens,
                hidden_size,
                inter_size,
                topk,
                num_experts,
                moe_tp_size,
                moe_ep_size,
                quant_mode,
                workload_distribution,
                moe_backend,
                is_gated,
            ),
            DatabaseMode::HYBRID => self
                .query_moe_silicon(
                    num_tokens,
                    hidden_size,
                    inter_size,
                    topk,
                    num_experts,
                    moe_tp_size,
                    moe_ep_size,
                    quant_mode,
                    workload_distribution,
                    moe_backend,
                    is_gated,
                )
                .or_else(|_| Ok(empirical())),
        }
    }

    pub fn query_mem_op(&self, mem_bytes: f64, database_mode: DatabaseMode) -> PerformanceResult {
        let gpu = &self.system_spec.gpu;
        let sol = mem_bytes / gpu.mem_bw * 1000.0;
        match database_mode {
            DatabaseMode::SOL | DatabaseMode::SOL_FULL => PerformanceResult::new(sol, 0.0),
            DatabaseMode::EMPIRICAL | DatabaseMode::SILICON | DatabaseMode::HYBRID => {
                let scale = gpu.mem_bw_empirical_scaling_factor.unwrap_or(0.8);
                let constant = gpu.mem_empirical_constant_latency.unwrap_or(0.0);
                PerformanceResult::new((mem_bytes / (gpu.mem_bw * scale) + constant) * 1000.0, 0.0)
            }
        }
    }

    pub fn query_p2p(
        &self,
        message_bytes: i64,
        database_mode: DatabaseMode,
    ) -> Result<PerformanceResult> {
        let inter_node_bw = self
            .system_spec
            .node
            .inter_node_bw
            .context("system node.inter_node_bw is required for p2p query")?;
        let sol = message_bytes as f64 / inter_node_bw * 1000.0;
        let empirical = || {
            let latency = self.system_spec.node.p2p_latency.unwrap_or(0.0);
            PerformanceResult::new(
                (message_bytes as f64 / inter_node_bw + latency) * 1000.0,
                0.0,
            )
        };
        Ok(match database_mode {
            DatabaseMode::SOL | DatabaseMode::SOL_FULL => PerformanceResult::new(sol, 0.0),
            DatabaseMode::EMPIRICAL | DatabaseMode::SILICON | DatabaseMode::HYBRID => empirical(),
        })
    }

    pub fn query_custom_allreduce(
        &self,
        quant_mode: CommQuantMode,
        tp_size: u32,
        size: i64,
        database_mode: DatabaseMode,
    ) -> Result<PerformanceResult> {
        let sol = self.custom_allreduce_sol(tp_size, size)?;
        let empirical = || PerformanceResult::new(sol.latency / 0.8, 0.0);
        match database_mode {
            DatabaseMode::SOL | DatabaseMode::SOL_FULL => Ok(sol),
            DatabaseMode::EMPIRICAL => Ok(empirical()),
            DatabaseMode::SILICON => self.query_custom_allreduce_silicon(quant_mode, tp_size, size),
            DatabaseMode::HYBRID => self
                .query_custom_allreduce_silicon(quant_mode, tp_size, size)
                .or_else(|_| Ok(empirical())),
        }
    }

    pub fn query_nccl(
        &self,
        quant_mode: CommQuantMode,
        num_gpus: u32,
        operation: &str,
        message_size: i64,
        database_mode: DatabaseMode,
    ) -> Result<PerformanceResult> {
        let sol = self.nccl_sol(quant_mode, num_gpus, operation, message_size)?;
        let empirical = || PerformanceResult::new(sol.latency / 0.8, 0.0);
        match database_mode {
            DatabaseMode::SOL | DatabaseMode::SOL_FULL => Ok(sol),
            DatabaseMode::EMPIRICAL => Ok(empirical()),
            DatabaseMode::SILICON => {
                self.query_nccl_silicon(quant_mode, num_gpus, operation, message_size)
            }
            DatabaseMode::HYBRID => self
                .query_nccl_silicon(quant_mode, num_gpus, operation, message_size)
                .or_else(|_| Ok(empirical())),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn query_moe_silicon(
        &self,
        num_tokens: i64,
        hidden_size: i64,
        inter_size: i64,
        topk: i64,
        num_experts: i64,
        moe_tp_size: u32,
        moe_ep_size: u32,
        quant_mode: MoeQuantMode,
        workload_distribution: &str,
        moe_backend: Option<&str>,
        is_gated: bool,
    ) -> Result<PerformanceResult> {
        if self.backend == BackendName::sglang && moe_backend == Some("deepep_moe") {
            bail!("sglang deepep MoE tables are not loaded by the native database yet");
        }

        let use_low_latency = self.backend == BackendName::trtllm
            && num_tokens <= 128
            && quant_mode == MoeQuantMode::nvfp4
            && is_gated;
        let data = if use_low_latency && !self.moe_low_latency_data.is_empty() {
            &self.moe_low_latency_data
        } else {
            &self.moe_data
        };
        let token_map = self
            .lookup_moe_token_map(
                data,
                hidden_size,
                inter_size,
                topk,
                num_experts,
                moe_tp_size,
                moe_ep_size,
                quant_mode,
                workload_distribution,
            )
            .or_else(|_| {
                if std::ptr::eq(data, &self.moe_low_latency_data) {
                    self.lookup_moe_token_map(
                        &self.moe_data,
                        hidden_size,
                        inter_size,
                        topk,
                        num_experts,
                        moe_tp_size,
                        moe_ep_size,
                        quant_mode,
                        workload_distribution,
                    )
                } else {
                    bail!("moe data not found")
                }
            })?;

        let token_points = token_map.keys().copied().collect::<Vec<_>>();
        if token_points.is_empty() {
            bail!("moe token map is empty");
        }
        let max_token = *token_points.iter().max().unwrap();
        if num_tokens > max_token {
            return Ok(self.estimate_moe_overflow(
                num_tokens,
                token_map,
                hidden_size,
                inter_size,
                topk,
                num_experts,
                moe_tp_size,
                moe_ep_size,
                quant_mode,
                is_gated,
            ));
        }
        let (left, right) = nearest_1d(num_tokens, &token_points, false)?;
        Ok(interp_result_1d(
            left,
            right,
            token_map[&left],
            token_map[&right],
            num_tokens,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn lookup_moe_token_map<'a>(
        &'a self,
        data: &'a MoeData,
        hidden_size: i64,
        inter_size: i64,
        topk: i64,
        num_experts: i64,
        moe_tp_size: u32,
        moe_ep_size: u32,
        quant_mode: MoeQuantMode,
        workload_distribution: &str,
    ) -> Result<&'a MoeTokenData> {
        let quant_data = data
            .get(&quant_mode)
            .with_context(|| format!("moe data not available for quant mode {quant_mode}"))?;
        let distribution = if quant_data.contains_key(workload_distribution) {
            workload_distribution
        } else {
            "uniform"
        };
        quant_data
            .get(distribution)
            .and_then(|m| m.get(&topk))
            .and_then(|m| m.get(&num_experts))
            .and_then(|m| m.get(&hidden_size))
            .and_then(|m| m.get(&inter_size))
            .and_then(|m| m.get(&moe_tp_size))
            .and_then(|m| m.get(&moe_ep_size))
            .with_context(|| {
                format!(
                    "moe data not available for distribution={distribution}, topk={topk}, experts={num_experts}, hidden={hidden_size}, inter={inter_size}, moe_tp={moe_tp_size}, moe_ep={moe_ep_size}"
                )
            })
    }

    #[allow(clippy::too_many_arguments)]
    fn estimate_moe_overflow(
        &self,
        query_tokens: i64,
        token_map: &MoeTokenData,
        hidden_size: i64,
        inter_size: i64,
        topk: i64,
        num_experts: i64,
        moe_tp_size: u32,
        moe_ep_size: u32,
        quant_mode: MoeQuantMode,
        is_gated: bool,
    ) -> PerformanceResult {
        let last_token = *token_map.keys().max().unwrap();
        let last = token_map[&last_token];
        let sol_last = self
            .moe_sol(
                last_token,
                hidden_size,
                inter_size,
                topk,
                num_experts,
                moe_tp_size,
                moe_ep_size,
                quant_mode,
                is_gated,
            )
            .0;
        let sol_query = self
            .moe_sol(
                query_tokens,
                hidden_size,
                inter_size,
                topk,
                num_experts,
                moe_tp_size,
                moe_ep_size,
                quant_mode,
                is_gated,
            )
            .0;
        let util = (sol_last / last.latency).clamp(1e-8, 1.0);
        let latency = sol_query / util;
        let energy = if last.energy > 0.0 && last.latency > 0.0 {
            last.energy * (latency / last.latency)
        } else {
            0.0
        };
        PerformanceResult::new(latency, energy)
    }

    #[allow(clippy::too_many_arguments)]
    fn moe_sol(
        &self,
        num_tokens: i64,
        hidden_size: i64,
        inter_size: i64,
        topk: i64,
        num_experts: i64,
        moe_tp_size: u32,
        moe_ep_size: u32,
        quant_mode: MoeQuantMode,
        is_gated: bool,
    ) -> (f64, f64, f64) {
        let num_gemms = if is_gated { 3.0 } else { 2.0 };
        let total_tokens = (num_tokens * topk) as f64;
        let moe_tp = moe_tp_size as f64;
        let moe_ep = moe_ep_size as f64;
        let ops = total_tokens * hidden_size as f64 * inter_size as f64 * num_gemms * 2.0
            / moe_ep
            / moe_tp;
        let routed_tokens = (total_tokens / moe_ep).floor();
        let active_experts = ((num_experts as f64 / moe_ep).floor()).min(routed_tokens);
        let mem_bytes = quant_mode.mapping().memory
            * (routed_tokens * hidden_size as f64 * 2.0
                + routed_tokens * inter_size as f64 * num_gemms / moe_tp
                + hidden_size as f64 * inter_size as f64 * num_gemms / moe_tp * active_experts);
        let bf16_flops = self.system_spec.gpu.bfloat16_tc_flops.unwrap_or(0.0);
        let sol_math = ops / (bf16_flops * quant_mode.mapping().compute) * 1000.0;
        let sol_mem = mem_bytes / self.system_spec.gpu.mem_bw * 1000.0;
        (sol_math.max(sol_mem), sol_math, sol_mem)
    }

    #[allow(clippy::too_many_arguments)]
    fn query_context_attention_silicon(
        &self,
        b: i64,
        s: i64,
        prefix: i64,
        n: i64,
        n_kv: i64,
        kvcache_quant_mode: KvCacheQuantMode,
        fmha_quant_mode: FmhaQuantMode,
        window_size: i64,
        head_size: i64,
    ) -> Result<PerformanceResult> {
        let full_s = s + prefix;
        let prefix_correction =
            ((full_s * full_s - prefix * prefix) as f64) / ((full_s * full_s) as f64);
        let n_kv_lookup = if n == n_kv { 0 } else { n_kv };
        let attention_dict = self
            .context_attention_data
            .get(&fmha_quant_mode)
            .and_then(|m| m.get(&kvcache_quant_mode))
            .and_then(|m| m.get(&n_kv_lookup))
            .and_then(|m| m.get(&head_size))
            .and_then(|m| m.get(&window_size))
            .with_context(|| {
                format!(
                    "context attention data not available for fmha={fmha_quant_mode}, kv={kvcache_quant_mode}, n_kv={n_kv_lookup}, head_size={head_size}, window_size={window_size}"
                )
            })?;
        let result = interp_3d_2d1d_result(n, full_s, b, attention_dict, InterpMethod::Cubic)?;
        Ok(result * prefix_correction)
    }

    fn query_generation_attention_silicon(
        &self,
        b: i64,
        s: i64,
        n: i64,
        n_kv: i64,
        kvcache_quant_mode: KvCacheQuantMode,
        window_size: i64,
        head_size: i64,
    ) -> Result<PerformanceResult> {
        let n_kv_lookup = if n == n_kv { 0 } else { n_kv };
        let attention_dict = self
            .generation_attention_data
            .get(&kvcache_quant_mode)
            .and_then(|m| m.get(&n_kv_lookup))
            .and_then(|m| m.get(&head_size))
            .and_then(|m| m.get(&window_size))
            .with_context(|| {
                format!(
                    "generation attention data not available for kv={kvcache_quant_mode}, n_kv={n_kv_lookup}, head_size={head_size}, window_size={window_size}"
                )
            })?;

        let s_min = 1_i64.max((s as f64 * 0.9) as i64);
        let s_max = s_min.max((s as f64 * 1.1) as i64);
        let mut total = PerformanceResult::new(0.0, 0.0);
        for i in 0..5 {
            let s_i = s_min + (s_max - s_min) * i / 4;
            let sample = interp_3d_2d1d_result(n, b, s_i, attention_dict, InterpMethod::Bilinear)?;
            total = total + sample;
        }
        Ok(total / 5.0)
    }

    #[allow(clippy::too_many_arguments)]
    fn query_context_mla_silicon(
        &self,
        b: i64,
        s: i64,
        prefix: i64,
        num_heads: i64,
        kvcache_quant_mode: KvCacheQuantMode,
        fmha_quant_mode: FmhaQuantMode,
    ) -> Result<PerformanceResult> {
        let full_s = s + prefix;
        let prefix_correction =
            ((full_s * full_s - prefix * prefix) as f64) / ((full_s * full_s) as f64);
        let mla_dict = self
            .context_mla_data
            .get(&fmha_quant_mode)
            .and_then(|m| m.get(&kvcache_quant_mode))
            .with_context(|| {
                format!("context MLA data not available for fmha={fmha_quant_mode}, kv={kvcache_quant_mode}")
            })?;
        let result = interp_3d_2d1d_result(num_heads, full_s, b, mla_dict, InterpMethod::Cubic)?;
        Ok(result * prefix_correction)
    }

    fn query_generation_mla_silicon(
        &self,
        b: i64,
        s: i64,
        num_heads: i64,
        kvcache_quant_mode: KvCacheQuantMode,
    ) -> Result<PerformanceResult> {
        let mla_dict = self
            .generation_mla_data
            .get(&kvcache_quant_mode)
            .with_context(|| {
                format!("generation MLA data not available for kv={kvcache_quant_mode}")
            })?;
        interp_3d_2d1d_result(num_heads, b, s, mla_dict, InterpMethod::Bilinear)
    }

    #[allow(clippy::too_many_arguments)]
    fn query_context_mla_module_silicon(
        &self,
        b: i64,
        s: i64,
        prefix: i64,
        num_heads: i64,
        kvcache_quant_mode: KvCacheQuantMode,
        fmha_quant_mode: FmhaQuantMode,
        gemm_quant_mode: GemmQuantMode,
    ) -> Result<PerformanceResult> {
        let full_s = s + prefix;
        let prefix_correction =
            ((full_s * full_s - prefix * prefix) as f64) / ((full_s * full_s) as f64);
        let mla_dict = self
            .context_mla_module_data
            .get(&fmha_quant_mode)
            .and_then(|m| m.get(&kvcache_quant_mode))
            .and_then(|m| m.get(&gemm_quant_mode))
            .with_context(|| {
                format!(
                    "context MLA module data not available for fmha={fmha_quant_mode}, kv={kvcache_quant_mode}, gemm={gemm_quant_mode}"
                )
            })?;
        let result = interp_3d_2d1d_result(num_heads, full_s, b, mla_dict, InterpMethod::Cubic)?;
        Ok(result * prefix_correction)
    }

    fn query_generation_mla_module_silicon(
        &self,
        b: i64,
        s: i64,
        num_heads: i64,
        kvcache_quant_mode: KvCacheQuantMode,
        fmha_quant_mode: FmhaQuantMode,
        gemm_quant_mode: GemmQuantMode,
    ) -> Result<PerformanceResult> {
        let mla_dict = self
            .generation_mla_module_data
            .get(&fmha_quant_mode)
            .and_then(|m| m.get(&kvcache_quant_mode))
            .and_then(|m| m.get(&gemm_quant_mode))
            .with_context(|| {
                format!(
                    "generation MLA module data not available for fmha={fmha_quant_mode}, kv={kvcache_quant_mode}, gemm={gemm_quant_mode}"
                )
            })?;
        interp_3d_2d1d_result(num_heads, b, s, mla_dict, InterpMethod::Cubic)
    }

    #[allow(clippy::too_many_arguments)]
    fn context_mla_sol(
        &self,
        b: i64,
        s: i64,
        prefix: i64,
        num_heads: i64,
        kvcache_quant_mode: KvCacheQuantMode,
        fmha_quant_mode: FmhaQuantMode,
    ) -> (f64, f64, f64) {
        let full_s = s + prefix;
        let ops = b as f64 * num_heads as f64 * 2.0 / 2.0
            * (192.0 + 128.0)
            * ((full_s * full_s - prefix * prefix) as f64);
        let mem_bytes = b as f64
            * num_heads as f64
            * (kvcache_quant_mode.mapping().memory * full_s as f64 * (192.0 + 128.0)
                + 2.0 * s as f64 * (192.0 + 128.0));
        let bf16_flops = self.system_spec.gpu.bfloat16_tc_flops.unwrap_or(0.0);
        let sol_math = ops / bf16_flops * 1000.0 / fmha_quant_mode.mapping().compute;
        let sol_mem = mem_bytes / self.system_spec.gpu.mem_bw * 1000.0;
        (sol_math.max(sol_mem), sol_math, sol_mem)
    }

    fn context_mla_empirical(
        &self,
        b: i64,
        s: i64,
        prefix: i64,
        num_heads: i64,
        kvcache_quant_mode: KvCacheQuantMode,
        fmha_quant_mode: FmhaQuantMode,
    ) -> f64 {
        self.context_mla_sol(b, s, prefix, num_heads, kvcache_quant_mode, fmha_quant_mode)
            .0
            / 0.6
    }

    fn generation_mla_sol(
        &self,
        b: i64,
        s: i64,
        num_heads: i64,
        kvcache_quant_mode: KvCacheQuantMode,
    ) -> (f64, f64, f64) {
        let quant_compute = if kvcache_quant_mode == KvCacheQuantMode::fp8 {
            FmhaQuantMode::fp8.mapping().compute
        } else {
            FmhaQuantMode::bfloat16.mapping().compute
        };
        let ops = 2.0 * b as f64 * num_heads as f64 * 1088.0 * s as f64;
        let mem_bytes = b as f64
            * (num_heads as f64 * 1088.0 * 2.0
                + (s - 1) as f64 * 576.0 * kvcache_quant_mode.mapping().memory);
        let bf16_flops = self.system_spec.gpu.bfloat16_tc_flops.unwrap_or(0.0);
        let sol_math = ops / bf16_flops * 1000.0 / quant_compute;
        let sol_mem = mem_bytes / self.system_spec.gpu.mem_bw * 1000.0;
        (sol_math.max(sol_mem), sol_math, sol_mem)
    }

    fn generation_mla_empirical(
        &self,
        b: i64,
        s: i64,
        num_heads: i64,
        kvcache_quant_mode: KvCacheQuantMode,
    ) -> f64 {
        self.generation_mla_sol(b, s, num_heads, kvcache_quant_mode)
            .0
            / 0.8
    }

    fn generation_mla_module_sol(
        &self,
        b: i64,
        s: i64,
        num_heads: i64,
        kvcache_quant_mode: KvCacheQuantMode,
        gemm_quant_mode: GemmQuantMode,
    ) -> (f64, f64, f64) {
        let (mut sol_time, mut sol_math, mut sol_mem) =
            self.generation_mla_sol(b, s, num_heads, kvcache_quant_mode);
        let bmm_ops = 2.0 * 2.0 * b as f64 * num_heads as f64 * 128.0 * 512.0;
        let bmm_mem = 2.0
            * num_heads as f64
            * (b as f64 * 640.0 + 128.0 * 512.0)
            * gemm_quant_mode.mapping().memory;
        let bf16_flops = self.system_spec.gpu.bfloat16_tc_flops.unwrap_or(0.0);
        sol_math += bmm_ops / (bf16_flops * gemm_quant_mode.mapping().compute) * 1000.0;
        sol_mem += bmm_mem / self.system_spec.gpu.mem_bw * 1000.0;
        sol_time = sol_math.max(sol_mem).max(sol_time);
        (sol_time, sol_math, sol_mem)
    }

    fn generation_mla_module_empirical(
        &self,
        b: i64,
        s: i64,
        num_heads: i64,
        kvcache_quant_mode: KvCacheQuantMode,
        gemm_quant_mode: GemmQuantMode,
    ) -> f64 {
        self.generation_mla_module_sol(b, s, num_heads, kvcache_quant_mode, gemm_quant_mode)
            .0
            / 0.5
    }

    fn generation_attention_sol(
        &self,
        b: i64,
        s: i64,
        n: i64,
        n_kv: i64,
        h: i64,
        w: i64,
        kvcache_quant_mode: KvCacheQuantMode,
    ) -> (f64, f64, f64) {
        let quant_compute = if kvcache_quant_mode == KvCacheQuantMode::fp8 {
            FmhaQuantMode::fp8.mapping().compute
        } else {
            FmhaQuantMode::bfloat16.mapping().compute
        };
        let kv_len = if w > 0 { (s - 1).min(w) } else { s - 1 };
        let ops = 2.0 * b as f64 * n as f64 * h as f64 * 2.0 * kv_len as f64;
        let mem_bytes = b as f64
            * (n as f64 * h as f64 * 2.0
                + 2.0
                    * n_kv as f64
                    * kv_len as f64
                    * h as f64
                    * kvcache_quant_mode.mapping().memory
                + n as f64 * h as f64 * 2.0);
        let bf16_flops = self.system_spec.gpu.bfloat16_tc_flops.unwrap_or(0.0);
        let sol_math = ops / bf16_flops * 1000.0 / quant_compute;
        let sol_mem = mem_bytes / self.system_spec.gpu.mem_bw * 1000.0;
        (sol_math.max(sol_mem), sol_math, sol_mem)
    }

    fn generation_attention_empirical(
        &self,
        b: i64,
        s: i64,
        n: i64,
        n_kv: i64,
        h: i64,
        w: i64,
        kvcache_quant_mode: KvCacheQuantMode,
    ) -> f64 {
        self.generation_attention_sol(b, s, n, n_kv, h, w, kvcache_quant_mode)
            .0
            / 0.8
    }

    #[allow(clippy::too_many_arguments)]
    fn context_attention_sol(
        &self,
        b: i64,
        s: i64,
        prefix: i64,
        n: i64,
        n_kv: i64,
        h: i64,
        w: i64,
        kvcache_quant_mode: KvCacheQuantMode,
        fmha_quant_mode: FmhaQuantMode,
    ) -> (f64, f64, f64) {
        let full_s = s + prefix;
        let ops = if w > 0 && full_s > w {
            2.0 * b as f64 * (full_s - prefix) as f64 * w as f64 * n as f64 * h as f64 * 2.0
        } else {
            2.0 * b as f64
                * ((full_s * full_s - prefix * prefix) as f64)
                * n as f64
                * h as f64
                * 2.0
                / 2.0
        };
        let mem_bytes = 2.0
            * b as f64
            * (n as f64 * (full_s - prefix) as f64 * h as f64
                + n as f64 * (full_s - prefix) as f64 * h as f64)
            + kvcache_quant_mode.mapping().memory
                * b as f64
                * (2.0 * n_kv as f64 * full_s as f64 * h as f64);
        let bf16_flops = self.system_spec.gpu.bfloat16_tc_flops.unwrap_or(0.0);
        let sol_math = ops / bf16_flops * 1000.0 / fmha_quant_mode.mapping().compute;
        let sol_mem = mem_bytes / self.system_spec.gpu.mem_bw * 1000.0;
        (sol_math.max(sol_mem), sol_math, sol_mem)
    }

    #[allow(clippy::too_many_arguments)]
    fn context_attention_empirical(
        &self,
        b: i64,
        s: i64,
        prefix: i64,
        n: i64,
        n_kv: i64,
        h: i64,
        w: i64,
        kvcache_quant_mode: KvCacheQuantMode,
        fmha_quant_mode: FmhaQuantMode,
    ) -> f64 {
        self.context_attention_sol(
            b,
            s,
            prefix,
            n,
            n_kv,
            h,
            w,
            kvcache_quant_mode,
            fmha_quant_mode,
        )
        .0 / 0.6
    }

    fn query_gemm_silicon(
        &self,
        m: i64,
        n: i64,
        k: i64,
        table_quant_mode: GemmQuantMode,
    ) -> Result<PerformanceResult> {
        let data = self.gemm_data.get(&table_quant_mode).with_context(|| {
            format!("gemm data not available for quant mode {table_quant_mode}")
        })?;

        if let Some(result) = data
            .get(&m)
            .and_then(|n_map| n_map.get(&n))
            .and_then(|k_map| k_map.get(&k))
            .copied()
        {
            return Ok(result);
        }

        let m_values = data
            .iter()
            .filter_map(|(&m_key, n_map)| {
                n_map.get(&n).and_then(|k_map| k_map.get(&k)).map(|_| m_key)
            })
            .collect::<Vec<_>>();
        if m_values.len() >= 2 {
            let (left, right) = nearest_1d(m, &m_values, false)?;
            let left_value = data[&left][&n][&k];
            let right_value = data[&right][&n][&k];
            return Ok(interp_result_1d(left, right, left_value, right_value, m));
        }

        interp_3d_2d1d_result(m, n, k, data, InterpMethod::Cubic)
    }

    fn gemm_sol(&self, m: i64, n: i64, k: i64, quant_mode: GemmQuantMode) -> (f64, f64, f64) {
        gemm_sol_from_spec(&self.system_spec, m, n, k, quant_mode)
    }

    fn gemm_empirical(&self, m: i64, n: i64, k: i64, quant_mode: GemmQuantMode) -> f64 {
        self.gemm_sol(m, n, k, quant_mode).0 / 0.8
    }

    fn p2p_bandwidth(&self, num_gpus: u32) -> Result<f64> {
        let node = &self.system_spec.node;
        if num_gpus <= node.num_gpus_per_node {
            node.intra_node_bw
                .or(node.inter_node_bw)
                .context("system node.intra_node_bw or inter_node_bw is required")
        } else if node
            .num_gpus_per_rack
            .map_or(true, |per_rack| num_gpus <= per_rack)
        {
            node.inter_node_bw
                .or(node.intra_node_bw)
                .context("system node.inter_node_bw or intra_node_bw is required")
        } else {
            node.inter_rack_bw
                .or(node.inter_node_bw)
                .or(node.intra_node_bw)
                .context("system node bandwidth is required")
        }
    }

    fn custom_allreduce_sol(&self, tp_size: u32, size: i64) -> Result<PerformanceResult> {
        if tp_size == 1 {
            return Ok(PerformanceResult::new(0.0, 0.0));
        }
        let p2p_bw = self.p2p_bandwidth(tp_size)?;
        let latency =
            2.0 * size as f64 * 2.0 / tp_size as f64 * (tp_size - 1) as f64 / p2p_bw * 1000.0;
        Ok(PerformanceResult::new(latency, 0.0))
    }

    fn query_custom_allreduce_silicon(
        &self,
        quant_mode: CommQuantMode,
        tp_size: u32,
        size: i64,
    ) -> Result<PerformanceResult> {
        if tp_size == 1 {
            return Ok(PerformanceResult::new(0.0, 0.0));
        }
        if self.system_spec.node.num_gpus_per_node == 72 && tp_size > 4 {
            return self.query_nccl(
                quant_mode,
                tp_size,
                "all_reduce",
                size,
                DatabaseMode::SILICON,
            );
        }
        let effective_tp = tp_size.min(self.system_spec.node.num_gpus_per_node);
        let comm_dict = self
            .custom_allreduce_data
            .get(&quant_mode)
            .and_then(|m| m.get(&effective_tp))
            .and_then(|m| m.get("AUTO"))
            .with_context(|| {
                format!(
                    "custom allreduce data not available for quant={quant_mode}, tp={effective_tp}"
                )
            })?;
        let size_keys = comm_dict.keys().copied().collect::<Vec<_>>();
        let (left, right) = nearest_1d(size, &size_keys, false)?;
        let mut result = interp_result_1d(left, right, comm_dict[&left], comm_dict[&right], size);

        if tp_size > self.system_spec.node.num_gpus_per_node {
            let base_bw = self.p2p_bandwidth(self.system_spec.node.num_gpus_per_node)?;
            let target_bw = self.p2p_bandwidth(tp_size)?;
            let base = self.system_spec.node.num_gpus_per_node as f64;
            let target = tp_size as f64;
            let scale_factor = (target - 1.0) / target * base / (base - 1.0) * base_bw / target_bw;
            result = result * scale_factor;
        }
        Ok(result)
    }

    fn nccl_sol(
        &self,
        quant_mode: CommQuantMode,
        num_gpus: u32,
        operation: &str,
        message_size: i64,
    ) -> Result<PerformanceResult> {
        if num_gpus <= 1 {
            return Ok(PerformanceResult::new(0.0, 0.0));
        }
        let p2p_bw = self.p2p_bandwidth(num_gpus)?;
        let mem = quant_mode.mapping().memory;
        let gpus = num_gpus as f64;
        let multiplier = match operation {
            "all_gather" | "alltoall" | "reduce_scatter" => 1.0,
            "all_reduce" => 2.0,
            _ => 0.0,
        };
        let latency =
            multiplier * mem * message_size as f64 * (gpus - 1.0) / gpus / p2p_bw * 1000.0;
        Ok(PerformanceResult::new(latency, 0.0))
    }

    fn query_nccl_silicon(
        &self,
        quant_mode: CommQuantMode,
        num_gpus: u32,
        operation: &str,
        message_size: i64,
    ) -> Result<PerformanceResult> {
        if num_gpus == 1 {
            return Ok(PerformanceResult::new(0.0, 0.0));
        }
        let by_gpu = self
            .nccl_data
            .get(&quant_mode)
            .and_then(|m| m.get(operation))
            .with_context(|| {
                format!("nccl data not available for quant={quant_mode}, op={operation}")
            })?;
        let max_num_gpus = by_gpu
            .keys()
            .copied()
            .max()
            .context("nccl data has no gpu counts")?;
        let lookup_gpus = num_gpus.min(max_num_gpus);
        let comm_dict = by_gpu
            .get(&lookup_gpus)
            .with_context(|| format!("nccl data missing gpu count {lookup_gpus}"))?;
        let size_keys = comm_dict.keys().copied().collect::<Vec<_>>();
        let (left, right) = nearest_1d(message_size, &size_keys, false)?;
        let mut result = interp_result_1d(
            left,
            right,
            comm_dict[&left],
            comm_dict[&right],
            message_size,
        );
        if num_gpus > max_num_gpus {
            let max_bw = self.p2p_bandwidth(max_num_gpus)?;
            let target_bw = self.p2p_bandwidth(num_gpus)?;
            let max_g = max_num_gpus as f64;
            let target = num_gpus as f64;
            let scaling_formula =
                (target - 1.0) / target * max_g / (max_g - 1.0) * max_bw / target_bw;
            result = result * scaling_formula;
        }
        Ok(result)
    }
}

fn load_gemm_data(
    path: impl AsRef<Path>,
    system_spec: &SystemSpec,
) -> Result<BTreeMap<GemmQuantMode, MMap>> {
    let path = path.as_ref();
    let mut result: BTreeMap<GemmQuantMode, MMap> = BTreeMap::new();
    if !path.exists() {
        return Ok(result);
    }
    let mut reader = csv::Reader::from_path(path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    for row in reader.deserialize::<GemmRow>() {
        let row = row.with_context(|| format!("failed to parse {}", path.display()))?;
        if row.gemm_dtype == "awq" || row.gemm_dtype == "gptq" {
            continue;
        }
        let quant_mode = row.gemm_dtype.parse::<GemmQuantMode>()?;
        let power = row.power.unwrap_or(0.0);
        result
            .entry(quant_mode)
            .or_default()
            .entry(row.m)
            .or_default()
            .entry(row.n)
            .or_default()
            .entry(row.k)
            .or_insert_with(|| PerformanceResult::new(row.latency, power * row.latency));
    }
    correct_gemm_data(&mut result, system_spec);
    let target_x_list = vec![
        1,
        2,
        4,
        8,
        16,
        32,
        48,
        64,
        80,
        96,
        128,
        160,
        192,
        224,
        256,
        320,
        384,
        448,
        512,
        640,
        768,
        896,
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
        131072,
        524288,
        1048576,
        2097152 * 8,
    ];
    let target_y_list = vec![
        32, 64, 128, 256, 512, 768, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 5120, 6144, 7168,
        8192, 10240, 12288, 14336, 16384, 20480, 24576, 28672, 32768, 40960, 49152, 57344, 65536,
        131072, 262144,
    ];
    let target_z_list = target_y_list.clone();
    for data in result.values_mut() {
        extrapolate_data_grid(data, &target_x_list, &target_y_list, &target_z_list, false);
    }
    correct_gemm_data(&mut result, system_spec);
    Ok(result)
}

fn correct_gemm_data(data: &mut BTreeMap<GemmQuantMode, MMap>, system_spec: &SystemSpec) {
    for (quant_mode, m_data) in data.iter_mut() {
        for (m, n_data) in m_data.iter_mut() {
            for (n, k_data) in n_data.iter_mut() {
                for (k, value) in k_data.iter_mut() {
                    let sol = gemm_sol_from_spec(system_spec, *m, *n, *k, *quant_mode).0;
                    if sol > value.latency {
                        value.latency = sol;
                    }
                }
            }
        }
    }
}

fn load_context_attention_data(path: impl AsRef<Path>) -> Result<ContextAttentionData> {
    let path = path.as_ref();
    let mut result: ContextAttentionData = BTreeMap::new();
    if !path.exists() {
        return Ok(result);
    }
    let mut reader = csv::Reader::from_path(path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    for row in reader.deserialize::<ContextAttentionRow>() {
        let row = row.with_context(|| format!("failed to parse {}", path.display()))?;
        let fmha = row.attn_dtype.parse::<FmhaQuantMode>()?;
        let kv = row.kv_cache_dtype.parse::<KvCacheQuantMode>()?;
        let kv_lookup = if row.num_heads == row.num_key_value_heads {
            0
        } else {
            row.num_key_value_heads
        };
        let window_size = row.window_size.unwrap_or(0);
        let power = row.power.unwrap_or(0.0);
        result
            .entry(fmha)
            .or_default()
            .entry(kv)
            .or_default()
            .entry(kv_lookup)
            .or_default()
            .entry(row.head_dim)
            .or_default()
            .entry(window_size)
            .or_default()
            .entry(row.num_heads)
            .or_default()
            .entry(row.isl)
            .or_default()
            .entry(row.batch_size)
            .or_insert_with(|| PerformanceResult::new(row.latency, power * row.latency));
    }
    let target_x_list = vec![
        1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 48, 56, 72, 96, 128,
    ];
    let mut target_y_list = vec![1, 16, 32, 64, 128, 256, 512, 1024, 2048];
    target_y_list.extend((0..14).map(|i| 4096 + i * 2048));
    target_y_list.extend((0..6).map(|i| 32768 + i * 16384));
    target_y_list.extend((0..12).map(|i| 131072 + i * 32768));
    target_y_list.extend((0..9).map(|i| 524288 + i * 65536));
    let target_z_list = vec![1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 384, 1024, 2048];
    for fmha_data in result.values_mut() {
        for kv_data in fmha_data.values_mut() {
            for n_kv_data in kv_data.values_mut() {
                for head_data in n_kv_data.values_mut() {
                    for data in head_data.values_mut() {
                        let min_x = data.keys().next().copied().unwrap_or(0);
                        let filtered_x = target_x_list
                            .iter()
                            .copied()
                            .filter(|x| *x >= min_x)
                            .collect::<Vec<_>>();
                        extrapolate_data_grid(
                            data,
                            &filtered_x,
                            &target_y_list,
                            &target_z_list,
                            true,
                        );
                    }
                }
            }
        }
    }
    Ok(result)
}

fn load_generation_attention_data(
    path: impl AsRef<Path>,
    system_spec: &SystemSpec,
) -> Result<GenerationAttentionData> {
    let path = path.as_ref();
    let mut result: GenerationAttentionData = BTreeMap::new();
    if !path.exists() {
        return Ok(result);
    }
    let mut reader = csv::Reader::from_path(path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    for row in reader.deserialize::<GenerationAttentionRow>() {
        let row = row.with_context(|| format!("failed to parse {}", path.display()))?;
        let kv = row.kv_cache_dtype.parse::<KvCacheQuantMode>()?;
        let kv_lookup = if row.num_heads == row.num_key_value_heads {
            0
        } else {
            row.num_key_value_heads
        };
        let window_size = row.window_size.unwrap_or(0);
        let power = row.power.unwrap_or(0.0);
        let seq_len = row.isl + row.s;
        result
            .entry(kv)
            .or_default()
            .entry(kv_lookup)
            .or_default()
            .entry(row.head_dim)
            .or_default()
            .entry(window_size)
            .or_default()
            .entry(row.num_heads)
            .or_default()
            .entry(row.batch_size)
            .or_default()
            .entry(seq_len)
            .or_insert_with(|| PerformanceResult::new(row.latency, power * row.latency));
    }

    correct_generation_attention_data(&mut result, system_spec);

    let target_x_list = vec![
        1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 48, 56, 72, 96, 128,
    ];
    let target_y_list = vec![1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 1024, 2048, 8192];
    let target_z_list = vec![
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
        65536,
        131072,
        262144,
        2097152 * 8,
    ];
    for kv_data in result.values_mut() {
        for n_kv_data in kv_data.values_mut() {
            for head_data in n_kv_data.values_mut() {
                for data in head_data.values_mut() {
                    let min_x = data.keys().next().copied().unwrap_or(0);
                    let filtered_x = target_x_list
                        .iter()
                        .copied()
                        .filter(|x| *x >= min_x)
                        .collect::<Vec<_>>();
                    extrapolate_data_grid(data, &filtered_x, &target_y_list, &target_z_list, false);
                }
            }
        }
    }
    correct_generation_attention_data(&mut result, system_spec);
    Ok(result)
}

fn correct_generation_attention_data(data: &mut GenerationAttentionData, system_spec: &SystemSpec) {
    for (kv_mode, kv_data) in data.iter_mut() {
        for (n_kv_lookup, n_kv_data) in kv_data.iter_mut() {
            for (head_size, head_data) in n_kv_data.iter_mut() {
                for (window_size, attention_data) in head_data.iter_mut() {
                    for (n, batch_data) in attention_data.iter_mut() {
                        let n_kv = if *n_kv_lookup == 0 { *n } else { *n_kv_lookup };
                        for (b, seq_data) in batch_data.iter_mut() {
                            for (s, value) in seq_data.iter_mut() {
                                let sol = generation_attention_sol_from_spec(
                                    system_spec,
                                    *b,
                                    *s,
                                    *n,
                                    n_kv,
                                    *head_size,
                                    *window_size,
                                    *kv_mode,
                                );
                                if sol > value.latency {
                                    value.latency = sol;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

fn load_context_mla_data(path: impl AsRef<Path>) -> Result<ContextMlaData> {
    let path = path.as_ref();
    let mut result: ContextMlaData = BTreeMap::new();
    if !path.exists() {
        return Ok(result);
    }
    let mut reader = csv::Reader::from_path(path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    for row in reader.deserialize::<ContextMlaRow>() {
        let row = row.with_context(|| format!("failed to parse {}", path.display()))?;
        let fmha = row.mla_dtype.parse::<FmhaQuantMode>()?;
        let kv = row.kv_cache_dtype.parse::<KvCacheQuantMode>()?;
        let num_heads = row
            .num_heads
            .or_else(|| row.tp_size.map(|tp| 128 / tp))
            .context("context MLA row missing num_heads and tp_size")?;
        let power = row.power.unwrap_or(0.0);
        result
            .entry(fmha)
            .or_default()
            .entry(kv)
            .or_default()
            .entry(num_heads)
            .or_default()
            .entry(row.isl)
            .or_default()
            .entry(row.batch_size)
            .or_insert_with(|| PerformanceResult::new(row.latency, power * row.latency));
    }

    let mut target_y_list = vec![1, 16, 32, 64, 128, 256, 512, 1024, 2048];
    target_y_list.extend((0..14).map(|i| 4096 + i * 2048));
    target_y_list.extend((0..6).map(|i| 32768 + i * 16384));
    target_y_list.extend((0..12).map(|i| 131072 + i * 32768));
    target_y_list.extend((0..9).map(|i| 524288 + i * 65536));
    let target_z_list = vec![1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 1024, 2048];
    for fmha_data in result.values_mut() {
        for data in fmha_data.values_mut() {
            let target_x_list = data.keys().copied().collect::<Vec<_>>();
            extrapolate_data_grid(data, &target_x_list, &target_y_list, &target_z_list, true);
        }
    }
    Ok(result)
}

fn load_generation_mla_data(path: impl AsRef<Path>) -> Result<GenerationMlaData> {
    let path = path.as_ref();
    let mut result: GenerationMlaData = BTreeMap::new();
    if !path.exists() {
        return Ok(result);
    }
    let mut reader = csv::Reader::from_path(path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    for row in reader.deserialize::<GenerationMlaRow>() {
        let row = row.with_context(|| format!("failed to parse {}", path.display()))?;
        let kv = row.kv_cache_dtype.parse::<KvCacheQuantMode>()?;
        let num_heads = row
            .num_heads
            .or_else(|| row.tp_size.map(|tp| 128 / tp))
            .context("generation MLA row missing num_heads and tp_size")?;
        let power = row.power.unwrap_or(0.0);
        let seq_len = row.isl + row.s;
        result
            .entry(kv)
            .or_default()
            .entry(num_heads)
            .or_default()
            .entry(row.batch_size)
            .or_default()
            .entry(seq_len)
            .or_insert_with(|| PerformanceResult::new(row.latency, power * row.latency));
    }

    let target_y_list = vec![1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 1024, 2048, 8192];
    let target_z_list = vec![
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
        65536,
        131072,
        262144,
        2097152 * 8,
    ];
    for data in result.values_mut() {
        let target_x_list = data.keys().copied().collect::<Vec<_>>();
        extrapolate_data_grid(data, &target_x_list, &target_y_list, &target_z_list, false);
    }
    Ok(result)
}

fn load_context_mla_module_data(path: impl AsRef<Path>) -> Result<ContextMlaModuleData> {
    let path = path.as_ref();
    let mut result: ContextMlaModuleData = BTreeMap::new();
    if !path.exists() {
        return Ok(result);
    }
    let mut reader = csv::Reader::from_path(path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    for row in reader.deserialize::<MlaModuleRow>() {
        let row = row.with_context(|| format!("failed to parse {}", path.display()))?;
        let fmha = row.mla_dtype.parse::<FmhaQuantMode>()?;
        let kv = row.kv_cache_dtype.parse::<KvCacheQuantMode>()?;
        let gemm = row.gemm_type.parse::<GemmQuantMode>()?;
        let power = row.power.unwrap_or(0.0);
        result
            .entry(fmha)
            .or_default()
            .entry(kv)
            .or_default()
            .entry(gemm)
            .or_default()
            .entry(row.num_heads)
            .or_default()
            .entry(row.isl)
            .or_default()
            .entry(row.batch_size)
            .or_insert_with(|| PerformanceResult::new(row.latency, power * row.latency));
    }

    let mut target_y_list = vec![1, 16, 32, 64, 128, 256, 512, 1024, 2048];
    target_y_list.extend((0..14).map(|i| 4096 + i * 2048));
    target_y_list.extend((0..6).map(|i| 32768 + i * 16384));
    target_y_list.extend((0..12).map(|i| 131072 + i * 32768));
    target_y_list.extend((0..9).map(|i| 524288 + i * 65536));
    let target_z_list = vec![1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 1024, 2048];
    for fmha_data in result.values_mut() {
        for kv_data in fmha_data.values_mut() {
            for data in kv_data.values_mut() {
                let target_x_list = data.keys().copied().collect::<Vec<_>>();
                extrapolate_data_grid(data, &target_x_list, &target_y_list, &target_z_list, false);
            }
        }
    }
    Ok(result)
}

fn load_generation_mla_module_data(path: impl AsRef<Path>) -> Result<GenerationMlaModuleData> {
    let path = path.as_ref();
    let mut result: GenerationMlaModuleData = BTreeMap::new();
    if !path.exists() {
        return Ok(result);
    }
    let mut reader = csv::Reader::from_path(path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    for row in reader.deserialize::<MlaModuleRow>() {
        let row = row.with_context(|| format!("failed to parse {}", path.display()))?;
        let fmha = row.mla_dtype.parse::<FmhaQuantMode>()?;
        let kv = row.kv_cache_dtype.parse::<KvCacheQuantMode>()?;
        let gemm = row.gemm_type.parse::<GemmQuantMode>()?;
        let power = row.power.unwrap_or(0.0);
        let seq_len = row.isl + row.s;
        result
            .entry(fmha)
            .or_default()
            .entry(kv)
            .or_default()
            .entry(gemm)
            .or_default()
            .entry(row.num_heads)
            .or_default()
            .entry(row.batch_size)
            .or_default()
            .entry(seq_len)
            .or_insert_with(|| PerformanceResult::new(row.latency, power * row.latency));
    }

    let target_y_list = vec![1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 1024, 2048, 8192];
    let target_z_list = vec![
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
        65536,
        131072,
        262144,
        2097152 * 8,
    ];
    for fmha_data in result.values_mut() {
        for kv_data in fmha_data.values_mut() {
            for data in kv_data.values_mut() {
                let target_x_list = data.keys().copied().collect::<Vec<_>>();
                extrapolate_data_grid(data, &target_x_list, &target_y_list, &target_z_list, false);
            }
        }
    }
    Ok(result)
}

fn load_moe_data(path: impl AsRef<Path>) -> Result<(MoeData, MoeData)> {
    let path = path.as_ref();
    let mut default_data: MoeData = BTreeMap::new();
    let mut low_latency_data: MoeData = BTreeMap::new();
    if !path.exists() {
        return Ok((default_data, low_latency_data));
    }
    let mut reader = csv::Reader::from_path(path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    for row in reader.deserialize::<MoeRow>() {
        let row = row.with_context(|| format!("failed to parse {}", path.display()))?;
        let quant_mode = row.moe_dtype.parse::<MoeQuantMode>()?;
        let power = row.power.unwrap_or(0.0);
        let target = if row.kernel_source.as_deref() == Some("moe_torch_flow_min_latency") {
            &mut low_latency_data
        } else {
            &mut default_data
        };
        target
            .entry(quant_mode)
            .or_default()
            .entry(row.distribution)
            .or_default()
            .entry(row.topk)
            .or_default()
            .entry(row.num_experts)
            .or_default()
            .entry(row.hidden_size)
            .or_default()
            .entry(row.inter_size)
            .or_default()
            .entry(row.moe_tp_size)
            .or_default()
            .entry(row.moe_ep_size)
            .or_default()
            .entry(row.num_tokens)
            .or_insert_with(|| PerformanceResult::new(row.latency, power * row.latency));
    }
    Ok((default_data, low_latency_data))
}

#[allow(clippy::too_many_arguments)]
fn gemm_sol_from_spec(
    system_spec: &SystemSpec,
    m: i64,
    n: i64,
    k: i64,
    quant_mode: GemmQuantMode,
) -> (f64, f64, f64) {
    let gpu = &system_spec.gpu;
    let tc_flops = match quant_mode.mapping().compute as i64 {
        1 => gpu.bfloat16_tc_flops.unwrap_or(0.0),
        2 => gpu
            .fp8_tc_flops
            .unwrap_or_else(|| gpu.bfloat16_tc_flops.unwrap_or(0.0) * 2.0),
        4 => gpu
            .fp4_tc_flops
            .unwrap_or_else(|| gpu.bfloat16_tc_flops.unwrap_or(0.0) * 4.0),
        _ => gpu.bfloat16_tc_flops.unwrap_or(0.0),
    };
    let mapping = quant_mode.mapping();
    let m = m as f64;
    let n = n as f64;
    let k = k as f64;
    let sol_math = 2.0 * m * n * k / tc_flops * 1000.0;
    let sol_mem = mapping.memory * (m * n + m * k + n * k) / system_spec.gpu.mem_bw * 1000.0;
    (sol_math.max(sol_mem), sol_math, sol_mem)
}

#[allow(clippy::too_many_arguments)]
fn generation_attention_sol_from_spec(
    system_spec: &SystemSpec,
    b: i64,
    s: i64,
    n: i64,
    n_kv: i64,
    h: i64,
    w: i64,
    kvcache_quant_mode: KvCacheQuantMode,
) -> f64 {
    let quant_compute = if kvcache_quant_mode == KvCacheQuantMode::fp8 {
        FmhaQuantMode::fp8.mapping().compute
    } else {
        FmhaQuantMode::bfloat16.mapping().compute
    };
    let kv_len = if w > 0 { (s - 1).min(w) } else { s - 1 };
    let ops = 2.0 * b as f64 * n as f64 * h as f64 * 2.0 * kv_len as f64;
    let mem_bytes = b as f64
        * (n as f64 * h as f64 * 2.0
            + 2.0 * n_kv as f64 * kv_len as f64 * h as f64 * kvcache_quant_mode.mapping().memory
            + n as f64 * h as f64 * 2.0);
    let bf16_flops = system_spec.gpu.bfloat16_tc_flops.unwrap_or(0.0);
    let sol_math = ops / bf16_flops * 1000.0 / quant_compute;
    let sol_mem = mem_bytes / system_spec.gpu.mem_bw * 1000.0;
    sol_math.max(sol_mem)
}

fn load_custom_allreduce_data(path: impl AsRef<Path>) -> Result<CustomAllreduceData> {
    let path = path.as_ref();
    let mut result: CustomAllreduceData = BTreeMap::new();
    if !path.exists() {
        return Ok(result);
    }
    let mut reader = csv::Reader::from_path(path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    for row in reader.deserialize::<CustomAllreduceRow>() {
        let row = row.with_context(|| format!("failed to parse {}", path.display()))?;
        let kernel_source = row.kernel_source.as_deref().unwrap_or_default();
        let backend = row.backend.as_deref().unwrap_or_default();
        if (kernel_source.ends_with("_eager") || backend.ends_with("_eager"))
            && !path.to_string_lossy().contains("b60")
        {
            continue;
        }
        // Python currently normalizes this table to half regardless of the CSV dtype.
        let quant_mode = CommQuantMode::half;
        let power = row.power.unwrap_or(0.0);
        result
            .entry(quant_mode)
            .or_default()
            .entry(row.num_gpus)
            .or_default()
            .entry("AUTO".to_string())
            .or_default()
            .entry(row.message_size)
            .or_insert_with(|| PerformanceResult::new(row.latency, power * row.latency));
    }
    Ok(result)
}

fn load_nccl_data(path: impl AsRef<Path>) -> Result<NcclData> {
    let path = path.as_ref();
    let mut result: NcclData = BTreeMap::new();
    if !path.exists() {
        return Ok(result);
    }
    let mut reader = csv::Reader::from_path(path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    for row in reader.deserialize::<NcclRow>() {
        let row = row.with_context(|| format!("failed to parse {}", path.display()))?;
        let quant_mode = row.nccl_dtype.parse::<CommQuantMode>()?;
        let power = row.power.unwrap_or(0.0);
        result
            .entry(quant_mode)
            .or_default()
            .entry(row.op_name)
            .or_default()
            .entry(row.num_gpus)
            .or_default()
            .entry(row.message_size)
            .or_insert_with(|| PerformanceResult::new(row.latency, power * row.latency));
    }
    Ok(result)
}

fn extrapolate_data_grid(
    data: &mut MMap,
    target_x_list: &[i64],
    target_y_list: &[i64],
    target_z_list: &[i64],
    sqrt_y_value: bool,
) {
    let x_list = data.keys().copied().collect::<Vec<_>>();
    for x in x_list {
        let y_existing = data
            .get(&x)
            .map(|m| m.keys().copied().collect::<Vec<_>>())
            .unwrap_or_default();
        for y in y_existing {
            if data[&x][&y].len() <= 1 {
                continue;
            }
            for &z in target_z_list {
                if data[&x][&y].contains_key(&z) {
                    continue;
                }
                let z_keys = data[&x][&y].keys().copied().collect::<Vec<_>>();
                if let Ok((z_left, z_right)) = nearest_1d(z, &z_keys, false) {
                    if let (Some(left), Some(right)) =
                        (data[&x][&y].get(&z_left), data[&x][&y].get(&z_right))
                    {
                        let value = interp_result_1d(z_left, z_right, *left, *right, z);
                        data.get_mut(&x)
                            .unwrap()
                            .get_mut(&y)
                            .unwrap()
                            .insert(z, value);
                    }
                }
            }
        }

        for &y in target_y_list {
            if data[&x].contains_key(&y) {
                continue;
            }
            let y_keys = data[&x].keys().copied().collect::<Vec<_>>();
            if y_keys.len() < 2 {
                break;
            }
            let Ok((y_left, y_right)) = nearest_1d(y, &y_keys, false) else {
                continue;
            };
            if !data[&x].contains_key(&y_left) || !data[&x].contains_key(&y_right) {
                continue;
            }
            let z_list = data[&x][&y_left].keys().copied().collect::<Vec<_>>();
            for z in z_list {
                let (Some(left), Some(right)) =
                    (data[&x][&y_left].get(&z), data[&x][&y_right].get(&z))
                else {
                    continue;
                };
                let value = if sqrt_y_value {
                    interp_result_1d_sqrt(y_left, y_right, *left, *right, y)
                } else {
                    interp_result_1d(y_left, y_right, *left, *right, y)
                };
                data.entry(x)
                    .or_default()
                    .entry(y)
                    .or_default()
                    .insert(z, value);
            }
        }
    }

    let x_keys = data.keys().copied().collect::<Vec<_>>();
    for &x in target_x_list {
        if data.contains_key(&x) {
            continue;
        }
        if x_keys.len() < 2 {
            break;
        }
        let Ok((x_left, x_right)) = nearest_1d(x, &x_keys, false) else {
            continue;
        };
        if !data.contains_key(&x_left) || !data.contains_key(&x_right) {
            continue;
        }
        let y_list = data[&x_left].keys().copied().collect::<Vec<_>>();
        for y in y_list {
            if !data[&x_right].contains_key(&y) {
                continue;
            }
            let z_list = data[&x_left][&y].keys().copied().collect::<Vec<_>>();
            for z in z_list {
                let (Some(left), Some(right)) =
                    (data[&x_left][&y].get(&z), data[&x_right][&y].get(&z))
                else {
                    continue;
                };
                let value = interp_result_1d(x_left, x_right, *left, *right, x);
                data.entry(x)
                    .or_default()
                    .entry(y)
                    .or_default()
                    .insert(z, value);
            }
        }
    }
}

fn interp_result_1d_sqrt(
    x0: i64,
    x1: i64,
    y0: PerformanceResult,
    y1: PerformanceResult,
    value: i64,
) -> PerformanceResult {
    let latency = interp_scalar_1d(x0, x1, y0.latency.sqrt(), y1.latency.sqrt(), value);
    let energy = if y0.energy > 0.0 && y1.energy > 0.0 {
        interp_scalar_1d(x0, x1, y0.energy.sqrt(), y1.energy.sqrt(), value)
    } else {
        0.0
    };
    PerformanceResult::new(latency * latency, energy * energy)
}

fn normalize_gemm_quant_mode_for_table(quant_mode: GemmQuantMode) -> GemmQuantMode {
    if quant_mode == GemmQuantMode::fp8_static {
        GemmQuantMode::fp8
    } else {
        quant_mode
    }
}

fn nearest_1d(x: i64, values: &[i64], inner_only: bool) -> Result<(i64, i64)> {
    if values.is_empty() {
        bail!("values is empty");
    }
    if values.len() == 1 {
        if inner_only && x != values[0] {
            bail!("x is not equal to the only value");
        }
        return Ok((values[0], values[0]));
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    if x < sorted[0] {
        if inner_only {
            bail!("x is less than the smallest value");
        }
        return Ok((sorted[0], sorted[1]));
    }
    if x > *sorted.last().unwrap() {
        if inner_only {
            bail!("x is greater than the largest value");
        }
        return Ok((sorted[sorted.len() - 2], sorted[sorted.len() - 1]));
    }
    for i in 0..sorted.len() {
        let value = sorted[i];
        if x >= value && i != sorted.len() - 1 {
            continue;
        }
        return Ok((sorted[i - 1], value));
    }
    bail!("failed to find nearest points")
}

fn interp_scalar_1d(x0: i64, x1: i64, y0: f64, mut y1: f64, value: i64) -> f64 {
    if (x0 - x1) as f64 * (y0 - y1) < 0.0 && (value - x0) * (value - x1) > 0 {
        y1 = y0;
    }
    if (y0 - y1).abs() < f64::EPSILON {
        y0
    } else {
        y0 + (y1 - y0) / (x1 - x0) as f64 * (value - x0) as f64
    }
}

fn interp_result_1d(
    x0: i64,
    x1: i64,
    y0: PerformanceResult,
    y1: PerformanceResult,
    value: i64,
) -> PerformanceResult {
    PerformanceResult::new(
        interp_scalar_1d(x0, x1, y0.latency, y1.latency, value),
        interp_scalar_1d(x0, x1, y0.energy, y1.energy, value),
    )
}

fn bilinear(
    x1: i64,
    x2: i64,
    y1: i64,
    y2: i64,
    q11: f64,
    q12: f64,
    q21: f64,
    q22: f64,
    x: i64,
    y: i64,
) -> f64 {
    if x1 == x2 && y1 == y2 {
        return q11;
    }
    if x1 == x2 {
        return interp_scalar_1d(y1, y2, q11, q12, y);
    }
    if y1 == y2 {
        return interp_scalar_1d(x1, x2, q11, q21, x);
    }
    let numerator = q11 * (x2 - x) as f64 * (y2 - y) as f64
        + q12 * (x2 - x) as f64 * (y - y1) as f64
        + q21 * (x - x1) as f64 * (y2 - y) as f64
        + q22 * (x - x1) as f64 * (y - y1) as f64;
    numerator / ((x2 - x1) * (y2 - y1)) as f64
}

#[derive(Debug, Clone, Copy)]
enum InterpMethod {
    Bilinear,
    Cubic,
}

fn clough_tocher_rect(
    x1: i64,
    x2: i64,
    y1: i64,
    y2: i64,
    q11: f64,
    q12: f64,
    q21: f64,
    q22: f64,
    x: i64,
    y: i64,
) -> f64 {
    if x1 == x2 || y1 == y2 {
        return bilinear(x1, x2, y1, y2, q11, q12, q21, q22, x, y);
    }

    let points = [
        [x1 as f64, y1 as f64],
        [x1 as f64, y2 as f64],
        [x2 as f64, y1 as f64],
        [x2 as f64, y2 as f64],
    ];
    let values = [q11, q12, q21, q22];
    let gradients = estimate_clough_tocher_gradients(&points, &values);

    let u = (x - x1) as f64 / (x2 - x1) as f64;
    let v = (y - y1) as f64 / (y2 - y1) as f64;
    if v >= u {
        clough_tocher_triangle(
            &points,
            &[3, 1, 0],
            [u, v - u, 1.0 - v],
            &values,
            &gradients,
        )
    } else {
        clough_tocher_triangle(
            &points,
            &[2, 3, 0],
            [u - v, v, 1.0 - u],
            &values,
            &gradients,
        )
    }
}

fn estimate_clough_tocher_gradients(points: &[[f64; 2]; 4], values: &[f64; 4]) -> [[f64; 2]; 4] {
    let neighbors: [&[usize]; 4] = [&[3, 1, 2], &[3, 0], &[3, 0], &[1, 0, 2]];
    let mut gradients = [[0.0; 2]; 4];

    for _ in 0..400 {
        let mut err = 0.0_f64;
        for ipoint in 0..4 {
            let mut q00 = 0.0;
            let mut q01 = 0.0;
            let mut q11 = 0.0;
            let mut s0 = 0.0;
            let mut s1 = 0.0;

            for &ipoint2 in neighbors[ipoint] {
                let ex = points[ipoint2][0] - points[ipoint][0];
                let ey = points[ipoint2][1] - points[ipoint][1];
                let len = (ex * ex + ey * ey).sqrt();
                let len3 = len * len * len;
                let f1 = values[ipoint];
                let f2 = values[ipoint2];
                let df2 = -ex * gradients[ipoint2][0] - ey * gradients[ipoint2][1];
                let edge_scale = (6.0 * (f1 - f2) - 2.0 * df2) / len3;

                q00 += 4.0 * ex * ex / len3;
                q01 += 4.0 * ex * ey / len3;
                q11 += 4.0 * ey * ey / len3;
                s0 += edge_scale * ex;
                s1 += edge_scale * ey;
            }

            let det = q00 * q11 - q01 * q01;
            let r0 = (q11 * s0 - q01 * s1) / det;
            let r1 = (-q01 * s0 + q00 * s1) / det;
            let change = (gradients[ipoint][0] + r0)
                .abs()
                .max((gradients[ipoint][1] + r1).abs());

            gradients[ipoint][0] = -r0;
            gradients[ipoint][1] = -r1;

            let change = change / 1.0_f64.max(r0.abs().max(r1.abs()));
            err = err.max(change);
        }

        if err < 1e-6 {
            break;
        }
    }

    gradients
}

fn clough_tocher_triangle(
    points: &[[f64; 2]; 4],
    simplex: &[usize; 3],
    b: [f64; 3],
    values: &[f64; 4],
    gradients: &[[f64; 2]; 4],
) -> f64 {
    let p0 = simplex[0];
    let p1 = simplex[1];
    let p2 = simplex[2];

    let e12x = points[p1][0] - points[p0][0];
    let e12y = points[p1][1] - points[p0][1];
    let e23x = points[p2][0] - points[p1][0];
    let e23y = points[p2][1] - points[p1][1];
    let e31x = points[p0][0] - points[p2][0];
    let e31y = points[p0][1] - points[p2][1];

    let f1 = values[p0];
    let f2 = values[p1];
    let f3 = values[p2];

    let df12 = gradients[p0][0] * e12x + gradients[p0][1] * e12y;
    let df21 = -(gradients[p1][0] * e12x + gradients[p1][1] * e12y);
    let df23 = gradients[p1][0] * e23x + gradients[p1][1] * e23y;
    let df32 = -(gradients[p2][0] * e23x + gradients[p2][1] * e23y);
    let df31 = gradients[p2][0] * e31x + gradients[p2][1] * e31y;
    let df13 = -(gradients[p0][0] * e31x + gradients[p0][1] * e31y);

    let c3000 = f1;
    let c2100 = (df12 + 3.0 * c3000) / 3.0;
    let c2010 = (df13 + 3.0 * c3000) / 3.0;
    let c0300 = f2;
    let c1200 = (df21 + 3.0 * c0300) / 3.0;
    let c0210 = (df23 + 3.0 * c0300) / 3.0;
    let c0030 = f3;
    let c1020 = (df31 + 3.0 * c0030) / 3.0;
    let c0120 = (df32 + 3.0 * c0030) / 3.0;

    let c2001 = (c2100 + c2010 + c3000) / 3.0;
    let c0201 = (c1200 + c0300 + c0210) / 3.0;
    let c0021 = (c1020 + c0120 + c0030) / 3.0;

    let g = [-0.5_f64, -0.5, -0.5];
    let c0111 = (g[0] * (-c0300 + 3.0 * c0210 - 3.0 * c0120 + c0030)
        + (-c0300 + 2.0 * c0210 - c0120 + c0021 + c0201))
        / 2.0;
    let c1011 = (g[1] * (-c0030 + 3.0 * c1020 - 3.0 * c2010 + c3000)
        + (-c0030 + 2.0 * c1020 - c2010 + c2001 + c0021))
        / 2.0;
    let c1101 = (g[2] * (-c3000 + 3.0 * c2100 - 3.0 * c1200 + c0300)
        + (-c3000 + 2.0 * c2100 - c1200 + c2001 + c0201))
        / 2.0;

    let c1002 = (c1101 + c1011 + c2001) / 3.0;
    let c0102 = (c1101 + c0111 + c0201) / 3.0;
    let c0012 = (c1011 + c0111 + c0021) / 3.0;
    let c0003 = (c1002 + c0102 + c0012) / 3.0;

    let minval = b[0].min(b[1]).min(b[2]);
    let b1 = b[0] - minval;
    let b2 = b[1] - minval;
    let b3 = b[2] - minval;
    let b4 = 3.0 * minval;

    b1.powi(3) * c3000
        + 3.0 * b1.powi(2) * b2 * c2100
        + 3.0 * b1.powi(2) * b3 * c2010
        + 3.0 * b1.powi(2) * b4 * c2001
        + 3.0 * b1 * b2.powi(2) * c1200
        + 6.0 * b1 * b2 * b4 * c1101
        + 3.0 * b1 * b3.powi(2) * c1020
        + 6.0 * b1 * b3 * b4 * c1011
        + 3.0 * b1 * b4.powi(2) * c1002
        + b2.powi(3) * c0300
        + 3.0 * b2.powi(2) * b3 * c0210
        + 3.0 * b2.powi(2) * b4 * c0201
        + 3.0 * b2 * b3.powi(2) * c0120
        + 6.0 * b2 * b3 * b4 * c0111
        + 3.0 * b2 * b4.powi(2) * c0102
        + b3.powi(3) * c0030
        + 3.0 * b3.powi(2) * b4 * c0021
        + 3.0 * b3 * b4.powi(2) * c0012
        + b4.powi(3) * c0003
}

fn interp_2d_value(
    x1: i64,
    x2: i64,
    y1: i64,
    y2: i64,
    q11: f64,
    q12: f64,
    q21: f64,
    q22: f64,
    x: i64,
    y: i64,
    method: InterpMethod,
) -> f64 {
    match method {
        InterpMethod::Bilinear => bilinear(x1, x2, y1, y2, q11, q12, q21, q22, x, y),
        InterpMethod::Cubic => clough_tocher_rect(x1, x2, y1, y2, q11, q12, q21, q22, x, y),
    }
}

fn interp_3d_2d1d_result(
    x: i64,
    y: i64,
    z: i64,
    data: &MMap,
    method: InterpMethod,
) -> Result<PerformanceResult> {
    let x_keys = data.keys().copied().collect::<Vec<_>>();
    let (x_left, x_right) = nearest_1d(x, &x_keys, true)?;
    let mut x_values = Vec::new();
    for xi in [x_left, x_right] {
        let y_keys = data[&xi].keys().copied().collect::<Vec<_>>();
        let (y_left, y_right) = nearest_1d(y, &y_keys, true)?;
        let z_keys = match method {
            // Python's bilinear path leaves z_left/z_right set from the final
            // y-neighbor loop iteration, i.e. y_right.
            InterpMethod::Bilinear => data[&xi][&y_right].keys().copied().collect::<Vec<_>>(),
            InterpMethod::Cubic => data[&xi][&y_left].keys().copied().collect::<Vec<_>>(),
        };
        let (z_left, z_right) = nearest_1d(z, &z_keys, true)?;
        let q11 = data[&xi][&y_left][&z_left];
        let q12 = data[&xi][&y_left][&z_right];
        let q21 = data[&xi][&y_right][&z_left];
        let q22 = data[&xi][&y_right][&z_right];
        let latency = interp_2d_value(
            y_left,
            y_right,
            z_left,
            z_right,
            q11.latency,
            q12.latency,
            q21.latency,
            q22.latency,
            y,
            z,
            method,
        );
        let energy = interp_2d_value(
            y_left, y_right, z_left, z_right, q11.energy, q12.energy, q21.energy, q22.energy, y, z,
            method,
        );
        x_values.push(PerformanceResult::new(latency, energy));
    }
    Ok(interp_result_1d(
        x_left,
        x_right,
        x_values[0],
        x_values[1],
        x,
    ))
}

#[cfg(test)]
mod tests {
    use super::nearest_1d;

    #[test]
    fn nearest_matches_python_boundary_behavior() {
        assert_eq!(nearest_1d(4, &[1, 4, 8], true).unwrap(), (4, 8));
        assert_eq!(nearest_1d(9, &[1, 4, 8], false).unwrap(), (4, 8));
        assert!(nearest_1d(9, &[1, 4, 8], true).is_err());
    }
}
