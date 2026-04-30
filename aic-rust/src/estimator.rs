use crate::model::{load_model_info, ModelInfo};
use crate::paths::default_systems_root;
use crate::perf_database::PerfDatabase;
use crate::search::{enumerate_parallel_config, ParallelConfig, ParallelSearch};
use crate::system::latest_database_version;
use crate::types::{
    BackendName, CommQuantMode, DatabaseMode, FmhaQuantMode, GemmQuantMode, KvCacheQuantMode,
    MoeQuantMode, PerformanceResult,
};
use anyhow::{bail, Context, Result};
use serde::Serialize;
use serde_json::Value;
use std::cmp::Ordering;
use std::collections::BTreeMap;

const TRTLLM_DEFAULT_FREE_GPU_MEMORY_FRACTION: f64 = 0.9;
const TRTLLM_DEFAULT_MAX_NUM_TOKENS: i64 = 8192;
const KV_CACHE_MEMORY_RESERVED_FRACTION: f64 = 0.015;
const KV_CACHE_MEMORY_TOLERANCE: f64 = 0.02;

#[derive(Debug, Clone)]
pub struct NativeDefaultRequest {
    pub model_path: String,
    pub total_gpus: u32,
    pub system: String,
    pub backend: BackendName,
    pub backend_version: Option<String>,
    pub database_mode: DatabaseMode,
    pub isl: i64,
    pub osl: i64,
    pub ttft: f64,
    pub tpot: f64,
    pub request_latency: Option<f64>,
    pub prefix: i64,
    pub top_n: usize,
    pub free_gpu_memory_fraction: Option<f64>,
    pub max_seq_len: Option<i64>,
    pub enable_chunked_prefill: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct NativeDefaultReport {
    pub chosen_exp: String,
    pub best_throughput: f64,
    pub disagg_ratio: f64,
    pub best: NativeDefaultResult,
    pub evaluated_configs: usize,
    pub backend_version: String,
    pub total_gpus: u32,
    #[serde(skip_serializing)]
    pub model_path: String,
    #[serde(skip_serializing)]
    pub is_moe: bool,
    #[serde(skip_serializing)]
    pub pareto_x_axis: String,
    #[serde(skip_serializing)]
    pub agg_pareto_front: Vec<ParetoPoint>,
    #[serde(skip_serializing)]
    pub disagg_pareto_front: Vec<ParetoPoint>,
    #[serde(skip_serializing)]
    pub top_agg_configs: Vec<AggResult>,
    #[serde(skip_serializing)]
    pub top_disagg_configs: Vec<DisaggResult>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "mode", content = "row")]
pub enum NativeDefaultResult {
    Agg(AggResult),
    Disagg(DisaggResult),
}

#[derive(Debug, Clone)]
pub struct ParetoPoint {
    pub x: f64,
    pub y: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct AggResult {
    pub model: String,
    pub isl: i64,
    pub osl: i64,
    pub prefix: i64,
    pub concurrency: f64,
    pub request_rate: f64,
    pub bs: i64,
    pub global_bs: i64,
    pub ttft: f64,
    pub tpot: f64,
    pub seq_s: f64,
    pub seq_s_gpu: f64,
    pub tokens_s: f64,
    pub tokens_s_gpu: f64,
    pub tokens_s_user: f64,
    pub request_latency: f64,
    pub num_total_gpus: u32,
    pub tp: u32,
    pub pp: u32,
    pub dp: u32,
    pub moe_tp: u32,
    pub moe_ep: u32,
    pub parallel: String,
    pub gemm: String,
    pub kvcache: String,
    pub fmha: String,
    pub moe: String,
    pub comm: String,
    pub memory: f64,
    pub balance_score: f64,
    pub num_ctx_reqs: f64,
    pub num_gen_reqs: f64,
    pub num_tokens: f64,
    pub ctx_tokens: i64,
    pub gen_tokens: f64,
    pub backend: String,
    pub version: String,
    pub system: String,
    pub power_w: f64,
    pub tokens_s_gpu_cluster: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct DisaggResult {
    pub model: String,
    pub isl: i64,
    pub osl: i64,
    pub prefix: i64,
    pub concurrency: f64,
    pub request_rate: f64,
    pub prefill_bs: i64,
    pub prefill_global_bs: i64,
    pub prefill_workers: i64,
    pub decode_bs: i64,
    pub decode_global_bs: i64,
    pub decode_workers: i64,
    pub ttft: f64,
    pub tpot: f64,
    pub request_latency: f64,
    pub seq_s: f64,
    pub seq_s_gpu: f64,
    pub tokens_s: f64,
    pub tokens_s_gpu: f64,
    pub tokens_s_user: f64,
    pub prefill_seq_s_worker: f64,
    pub decode_seq_s_worker: f64,
    pub num_total_gpus: u32,
    pub prefill_tp: u32,
    pub prefill_pp: u32,
    pub prefill_dp: u32,
    pub prefill_parallel: String,
    pub prefill_memory: f64,
    pub decode_tp: u32,
    pub decode_pp: u32,
    pub decode_dp: u32,
    pub decode_parallel: String,
    pub decode_memory: f64,
    pub backend: String,
    pub version: String,
    pub system: String,
    pub power_w: f64,
    pub tokens_s_gpu_cluster: f64,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct StaticRow {
    model: String,
    isl: i64,
    osl: i64,
    prefix: i64,
    concurrency: f64,
    bs: i64,
    global_bs: i64,
    ttft: f64,
    tpot: f64,
    seq_s: f64,
    seq_s_gpu: f64,
    tokens_s: f64,
    tokens_s_gpu: f64,
    tokens_s_user: f64,
    request_latency: f64,
    num_total_gpus: u32,
    tp: u32,
    pp: u32,
    dp: u32,
    moe_tp: u32,
    moe_ep: u32,
    parallel: String,
    gemm: String,
    kvcache: String,
    fmha: String,
    moe: String,
    comm: String,
    memory: f64,
    backend: String,
    version: String,
    system: String,
    power_w: f64,
}

#[derive(Debug, Clone)]
struct RuntimeConfig {
    batch_size: i64,
    beam_width: i64,
    isl: i64,
    osl: i64,
    prefix: i64,
    database_mode: DatabaseMode,
    seq_imbalance_correction_scale: f64,
    gen_seq_imbalance_correction_scale: f64,
}

#[derive(Debug, Clone, Copy)]
struct ModelConfig {
    tp_size: u32,
    pp_size: u32,
    attention_dp_size: u32,
    moe_tp_size: u32,
    moe_ep_size: u32,
    gemm_quant_mode: GemmQuantMode,
    moe_quant_mode: MoeQuantMode,
    kvcache_quant_mode: KvCacheQuantMode,
    fmha_quant_mode: FmhaQuantMode,
    comm_quant_mode: CommQuantMode,
    nextn: u32,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct DenseModel {
    model_path: String,
    model_family: String,
    architecture: String,
    config: ModelConfig,
    num_layers: f64,
    num_heads: i64,
    num_kv_heads: i64,
    head_size: i64,
    hidden_size: i64,
    inter_size: i64,
    vocab_size: i64,
    kvcache_elements_per_token_override: Option<f64>,
    context_ops: Vec<Op>,
    generation_ops: Vec<Op>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct StaticSummary {
    context_latency: BTreeMap<String, f64>,
    context_energy: BTreeMap<String, f64>,
    generation_latency: BTreeMap<String, f64>,
    generation_energy: BTreeMap<String, f64>,
    memory: MemoryUsage,
    oom: bool,
    kv_cache_oom: bool,
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
struct MemoryUsage {
    total: f64,
    weights: f64,
    activations: f64,
    kvcache: f64,
    nccl: f64,
    others: f64,
}

#[derive(Debug, Clone)]
struct Op {
    name: String,
    scale_factor: f64,
    kind: OpKind,
}

#[derive(Debug, Clone)]
enum OpKind {
    Embedding {
        row_size: i64,
        column_size: i64,
    },
    ElementWise {
        dim_in: i64,
        dim_out: i64,
        scale_num_tokens: i64,
    },
    Gemm {
        n: i64,
        k: i64,
        quant_mode: GemmQuantMode,
    },
    ContextAttention {
        n: i64,
        n_kv: i64,
        kvcache_quant_mode: KvCacheQuantMode,
        fmha_quant_mode: FmhaQuantMode,
        window_size: i64,
        head_size: i64,
        use_qk_norm: bool,
    },
    GenerationAttention {
        n: i64,
        n_kv: i64,
        kvcache_quant_mode: KvCacheQuantMode,
        window_size: i64,
        head_size: i64,
    },
    MlaModule {
        is_context: bool,
        num_heads: i64,
        kvcache_quant_mode: KvCacheQuantMode,
        fmha_quant_mode: FmhaQuantMode,
        gemm_quant_mode: GemmQuantMode,
        weight_bytes: f64,
    },
    Moe {
        hidden_size: i64,
        inter_size: i64,
        topk: i64,
        num_experts: i64,
        moe_tp_size: u32,
        moe_ep_size: u32,
        quant_mode: MoeQuantMode,
        workload_distribution: String,
        attention_dp_size: u32,
        is_gated: bool,
    },
    MoeDispatch {
        hidden_size: i64,
        moe_tp_size: u32,
        moe_ep_size: u32,
        attention_dp_size: u32,
        pre_dispatch: bool,
    },
    CustomAllReduce {
        h: i64,
        tp_size: u32,
    },
    P2P {
        h: i64,
        pp_size: u32,
    },
    Overlap {
        group_a: Vec<Op>,
        group_b: Vec<Op>,
    },
}

pub fn run_native_default(req: &NativeDefaultRequest) -> Result<NativeDefaultReport> {
    let model_info = load_model_info(&req.model_path)?;
    let is_moe = is_moe_model(&model_info);
    let version = match &req.backend_version {
        Some(v) => v.clone(),
        None => latest_database_version(default_systems_root(), &req.system, req.backend)?
            .with_context(|| format!("no database for {}/{}", req.system, req.backend))?,
    };
    let db = PerfDatabase::load(default_systems_root(), &req.system, req.backend, &version)?;
    let base_quant = infer_quant_modes(&model_info, req.backend)?;
    let parallel_config_list = if is_moe {
        enumerate_parallel_config(&ParallelSearch {
            num_gpu_list: vec![1, 2, 4, 8]
                .into_iter()
                .filter(|v| *v <= req.total_gpus)
                .collect(),
            tp_list: vec![1, 2, 4, 8],
            pp_list: vec![1],
            dp_list: vec![1, 2, 4, 8],
            moe_tp_list: vec![1, 2, 4, 8],
            moe_ep_list: vec![1, 2, 4, 8],
            is_moe: true,
            backend: req.backend,
            ..ParallelSearch::default()
        })
    } else {
        enumerate_parallel_config(&ParallelSearch {
            num_gpu_list: vec![1, 2, 4, 8]
                .into_iter()
                .filter(|v| *v <= req.total_gpus)
                .collect(),
            tp_list: vec![1, 2, 4, 8],
            pp_list: vec![1],
            backend: req.backend,
            ..ParallelSearch::default()
        })
    };

    let mut pareto_rows = Vec::new();
    for parallel in parallel_config_list {
        let mut model_config = base_quant;
        model_config.tp_size = parallel.tp;
        model_config.pp_size = parallel.pp;
        model_config.attention_dp_size = parallel.dp;
        model_config.moe_tp_size = parallel.moe_tp;
        model_config.moe_ep_size = parallel.moe_ep;
        let Ok(model) = build_model(&req.model_path, &model_info, model_config) else {
            continue;
        };
        let rows = find_best_agg_for_parallel(req, &db, &model, parallel)?;
        pareto_rows.extend(rows);
    }

    if pareto_rows.is_empty() {
        bail!("native default found no valid agg configurations");
    }

    for row in &mut pareto_rows {
        let replicas = req.total_gpus / row.num_total_gpus;
        row.tokens_s_gpu_cluster =
            row.tokens_s_gpu * replicas as f64 * row.num_total_gpus as f64 / req.total_gpus as f64;
    }

    let pareto_x_axis = pareto_x_axis(req).to_string();
    let agg_pareto_front = agg_pareto_front(req, &pareto_rows);
    let top_agg_configs = top_agg_configs(req, &pareto_rows);
    let best_agg = pick_best_agg(req, &pareto_rows).context("no best agg config")?;
    let best_agg_throughput = best_agg.tokens_s_gpu_cluster;
    let mut best = NativeDefaultResult::Agg(best_agg.clone());
    let mut chosen_exp = "agg".to_string();
    let mut best_throughput = best_agg_throughput;
    let mut evaluated_configs = pareto_rows.len();
    let mut best_disagg_throughput = None;
    let mut disagg_front = Vec::new();
    let mut top_disagg = Vec::new();

    if req.total_gpus >= 2 {
        let mut disagg_rows = run_disagg_search(req, &db, &model_info, base_quant, is_moe)?;
        for row in &mut disagg_rows {
            let replicas = req.total_gpus / row.num_total_gpus;
            row.tokens_s_gpu_cluster =
                row.tokens_s_gpu * replicas as f64 * row.num_total_gpus as f64
                    / req.total_gpus as f64;
        }
        evaluated_configs += disagg_rows.len();
        disagg_front = disagg_pareto_front(req, &disagg_rows);
        top_disagg = top_disagg_configs(req, &disagg_rows);
        if let Some(best_disagg) = pick_best_disagg(req, &disagg_rows) {
            best_disagg_throughput = Some(best_disagg.tokens_s_gpu_cluster);
            if best_disagg.tokens_s_gpu_cluster > best_throughput {
                best_throughput = best_disagg.tokens_s_gpu_cluster;
                chosen_exp = "disagg".to_string();
                best = NativeDefaultResult::Disagg(best_disagg);
            }
        }
    }

    Ok(NativeDefaultReport {
        chosen_exp,
        best_throughput,
        disagg_ratio: best_disagg_throughput
            .map(|throughput| throughput / best_agg_throughput)
            .unwrap_or(0.0),
        best,
        evaluated_configs,
        backend_version: version,
        total_gpus: req.total_gpus,
        model_path: req.model_path.clone(),
        is_moe,
        pareto_x_axis,
        agg_pareto_front,
        disagg_pareto_front: disagg_front,
        top_agg_configs,
        top_disagg_configs: top_disagg,
    })
}

fn pareto_x_axis(req: &NativeDefaultRequest) -> &'static str {
    if req.request_latency.is_some_and(|v| v > 0.0) {
        "request_latency"
    } else {
        "tokens/s/user"
    }
}

fn agg_pareto_front(req: &NativeDefaultRequest, rows: &[AggResult]) -> Vec<ParetoPoint> {
    let use_request_latency = req.request_latency.is_some_and(|v| v > 0.0);
    let points = rows
        .iter()
        .map(|row| ParetoPoint {
            x: if use_request_latency {
                row.request_latency
            } else {
                row.tokens_s_user
            },
            y: row.tokens_s_gpu_cluster,
        })
        .collect::<Vec<_>>();
    pareto_front(points, !use_request_latency)
}

fn disagg_pareto_front(req: &NativeDefaultRequest, rows: &[DisaggResult]) -> Vec<ParetoPoint> {
    let use_request_latency = req.request_latency.is_some_and(|v| v > 0.0);
    let points = rows
        .iter()
        .map(|row| ParetoPoint {
            x: if use_request_latency {
                row.request_latency
            } else {
                row.tokens_s_user
            },
            y: row.tokens_s_gpu_cluster,
        })
        .collect::<Vec<_>>();
    pareto_front(points, !use_request_latency)
}

fn top_agg_configs(req: &NativeDefaultRequest, rows: &[AggResult]) -> Vec<AggResult> {
    let mut candidates = rows
        .iter()
        .filter(|row| row_meets_sla_agg(req, row))
        .cloned()
        .collect::<Vec<_>>();
    let all_exceed_sla = candidates.is_empty();
    if all_exceed_sla {
        candidates = rows.to_vec();
    }
    candidates = best_agg_by_parallel(candidates);
    if all_exceed_sla {
        if req.request_latency.is_some_and(|v| v > 0.0) {
            candidates.sort_by(|a, b| a.request_latency.total_cmp(&b.request_latency));
        } else {
            candidates.sort_by(|a, b| a.tpot.total_cmp(&b.tpot));
        }
    } else {
        sort_best_agg_candidates(req, &mut candidates);
    }
    candidates.truncate(req.top_n);
    candidates
}

fn top_disagg_configs(req: &NativeDefaultRequest, rows: &[DisaggResult]) -> Vec<DisaggResult> {
    let mut candidates = rows
        .iter()
        .filter(|row| row_meets_sla_disagg(req, row))
        .cloned()
        .collect::<Vec<_>>();
    let all_exceed_sla = candidates.is_empty();
    if all_exceed_sla {
        candidates = rows.to_vec();
    }
    candidates = best_disagg_by_decode_parallel_for_top(req, candidates);
    if all_exceed_sla {
        if req.request_latency.is_some_and(|v| v > 0.0) {
            candidates.sort_by(|a, b| a.request_latency.total_cmp(&b.request_latency));
        } else {
            candidates.sort_by(|a, b| a.tpot.total_cmp(&b.tpot));
        }
    } else {
        sort_best_disagg_candidates(req, &mut candidates);
    }
    candidates.truncate(req.top_n);
    candidates
}

fn best_disagg_by_decode_parallel_for_top(
    req: &NativeDefaultRequest,
    rows: Vec<DisaggResult>,
) -> Vec<DisaggResult> {
    let mut by_parallel: BTreeMap<String, DisaggResult> = BTreeMap::new();
    let use_request_latency = req.request_latency.is_some_and(|v| v > 0.0);
    for row in rows {
        match by_parallel.get(&row.decode_parallel) {
            Some(current) if disagg_top_tie_keeps_current(use_request_latency, current, &row) => {
                continue;
            }
            _ => {
                by_parallel.insert(row.decode_parallel.clone(), row);
            }
        }
    }
    by_parallel.into_values().collect()
}

fn disagg_top_tie_keeps_current(
    use_request_latency: bool,
    current: &DisaggResult,
    candidate: &DisaggResult,
) -> bool {
    match candidate
        .tokens_s_gpu_cluster
        .total_cmp(&current.tokens_s_gpu_cluster)
    {
        Ordering::Less => true,
        Ordering::Greater => false,
        Ordering::Equal if use_request_latency => {
            candidate.request_latency >= current.request_latency
        }
        Ordering::Equal => candidate.tokens_s_user <= current.tokens_s_user,
    }
}

fn pareto_front(points: Vec<ParetoPoint>, maximize_x: bool) -> Vec<ParetoPoint> {
    let points = points
        .into_iter()
        .filter(|point| point.x.is_finite() && point.y.is_finite())
        .collect::<Vec<_>>();
    let mut out = Vec::new();

    'candidate: for (idx, point) in points.iter().enumerate() {
        for (other_idx, other) in points.iter().enumerate() {
            if idx == other_idx {
                continue;
            }
            let x_better_or_equal = if maximize_x {
                other.x >= point.x
            } else {
                other.x <= point.x
            };
            let x_strictly_better = if maximize_x {
                other.x > point.x
            } else {
                other.x < point.x
            };
            if x_better_or_equal && other.y >= point.y && (x_strictly_better || other.y > point.y) {
                continue 'candidate;
            }
        }
        out.push(point.clone());
    }

    out.sort_by(|a, b| a.x.total_cmp(&b.x).then_with(|| a.y.total_cmp(&b.y)));
    out
}

fn pick_best_agg(req: &NativeDefaultRequest, rows: &[AggResult]) -> Option<AggResult> {
    let mut candidates = rows
        .iter()
        .filter(|row| row_meets_sla_agg(req, row))
        .cloned()
        .collect::<Vec<_>>();
    if candidates.is_empty() {
        candidates = best_agg_by_parallel(rows.to_vec());
        if req.request_latency.is_some_and(|v| v > 0.0) {
            candidates.sort_by(|a, b| a.request_latency.total_cmp(&b.request_latency));
        } else {
            candidates.sort_by(|a, b| a.tpot.total_cmp(&b.tpot));
        }
    } else {
        candidates = best_agg_by_parallel(candidates);
        sort_best_agg_candidates(req, &mut candidates);
    }
    candidates.into_iter().next()
}

fn pick_best_disagg(req: &NativeDefaultRequest, rows: &[DisaggResult]) -> Option<DisaggResult> {
    let mut candidates = rows
        .iter()
        .filter(|row| row_meets_sla_disagg(req, row))
        .cloned()
        .collect::<Vec<_>>();
    if candidates.is_empty() {
        candidates = best_disagg_by_decode_parallel(rows.to_vec());
        if req.request_latency.is_some_and(|v| v > 0.0) {
            candidates.sort_by(|a, b| a.request_latency.total_cmp(&b.request_latency));
        } else {
            candidates.sort_by(|a, b| a.tpot.total_cmp(&b.tpot));
        }
    } else {
        candidates = best_disagg_by_decode_parallel(candidates);
        sort_best_disagg_candidates(req, &mut candidates);
    }
    candidates.into_iter().next()
}

fn best_agg_by_parallel(rows: Vec<AggResult>) -> Vec<AggResult> {
    let mut by_parallel: BTreeMap<String, AggResult> = BTreeMap::new();
    for row in rows {
        match by_parallel.get(&row.parallel) {
            Some(current) if row.tokens_s_gpu_cluster <= current.tokens_s_gpu_cluster => {
                continue;
            }
            _ => {
                by_parallel.insert(row.parallel.clone(), row);
            }
        }
    }
    by_parallel.into_values().collect()
}

fn best_disagg_by_decode_parallel(rows: Vec<DisaggResult>) -> Vec<DisaggResult> {
    let mut by_parallel: BTreeMap<String, DisaggResult> = BTreeMap::new();
    for row in rows {
        match by_parallel.get(&row.decode_parallel) {
            Some(current) if row.tokens_s_gpu_cluster <= current.tokens_s_gpu_cluster => {
                continue;
            }
            _ => {
                by_parallel.insert(row.decode_parallel.clone(), row);
            }
        }
    }
    by_parallel.into_values().collect()
}

fn sort_best_agg_candidates(req: &NativeDefaultRequest, candidates: &mut [AggResult]) {
    if req.request_latency.is_some_and(|v| v > 0.0) {
        candidates.sort_by(|a, b| {
            b.tokens_s_gpu_cluster
                .total_cmp(&a.tokens_s_gpu_cluster)
                .then_with(|| a.request_latency.total_cmp(&b.request_latency))
        });
    } else {
        candidates.sort_by(|a, b| {
            b.tokens_s_gpu_cluster
                .total_cmp(&a.tokens_s_gpu_cluster)
                .then_with(|| b.tokens_s_user.total_cmp(&a.tokens_s_user))
        });
    }
}

fn sort_best_disagg_candidates(req: &NativeDefaultRequest, candidates: &mut [DisaggResult]) {
    if req.request_latency.is_some_and(|v| v > 0.0) {
        candidates.sort_by(|a, b| {
            b.tokens_s_gpu_cluster
                .total_cmp(&a.tokens_s_gpu_cluster)
                .then_with(|| a.request_latency.total_cmp(&b.request_latency))
        });
    } else {
        candidates.sort_by(|a, b| {
            b.tokens_s_gpu_cluster
                .total_cmp(&a.tokens_s_gpu_cluster)
                .then_with(|| b.tokens_s_user.total_cmp(&a.tokens_s_user))
        });
    }
}

fn row_meets_sla_agg(req: &NativeDefaultRequest, row: &AggResult) -> bool {
    if let Some(target) = req.request_latency.filter(|v| *v > 0.0) {
        row.request_latency <= target
    } else {
        row.tpot <= req.tpot
    }
}

fn row_meets_sla_disagg(req: &NativeDefaultRequest, row: &DisaggResult) -> bool {
    if let Some(target) = req.request_latency.filter(|v| *v > 0.0) {
        row.request_latency <= target
    } else {
        row.tpot <= req.tpot
    }
}

fn constraint_pairs(req: &NativeDefaultRequest) -> Vec<(f64, f64)> {
    if let Some(request_latency) = req.request_latency.filter(|v| *v > 0.0) {
        enumerate_ttft_tpot_constraints(req.osl, request_latency, Some(req.ttft))
    } else {
        vec![(req.ttft, req.tpot)]
    }
}

fn enumerate_ttft_tpot_constraints(
    osl: i64,
    request_latency: f64,
    ttft: Option<f64>,
) -> Vec<(f64, f64)> {
    if osl <= 1 {
        return Vec::new();
    }

    let ttft = ttft.unwrap_or(request_latency * 0.95);
    let base_values = [
        300.0, 400.0, 500.0, 600.0, 800.0, 1000.0, 1200.0, 1400.0, 1600.0, 2000.0, 3000.0, 5000.0,
        8000.0,
    ];
    let mut ttft_values = base_values.to_vec();
    for ratio in [0.1, 0.2, 0.3, 0.5, 0.7] {
        let value = request_latency * ratio;
        if value < base_values[0] || value > base_values[base_values.len() - 1] {
            ttft_values.push(value);
        }
    }
    ttft_values.push(ttft);
    ttft_values.sort_by(f64::total_cmp);
    ttft_values.dedup_by(|a, b| (*a - *b).abs() < 1e-9);

    ttft_values
        .into_iter()
        .filter(|value| *value < request_latency)
        .map(|ttft| (ttft, (request_latency - ttft) / (osl - 1) as f64))
        .collect()
}

fn run_disagg_search(
    req: &NativeDefaultRequest,
    db: &PerfDatabase,
    model_info: &ModelInfo,
    base_quant: ModelConfig,
    is_moe: bool,
) -> Result<Vec<DisaggResult>> {
    let prefill_parallel = disagg_parallel_configs(req, is_moe, false);
    let decode_parallel = disagg_parallel_configs(req, is_moe, true);
    let runtime = RuntimeConfig {
        batch_size: 1,
        beam_width: 1,
        isl: req.isl,
        osl: req.osl,
        prefix: req.prefix,
        database_mode: req.database_mode,
        seq_imbalance_correction_scale: 1.0,
        gen_seq_imbalance_correction_scale: 1.0,
    };

    let prefill_rows = static_worker_candidates(
        req,
        db,
        model_info,
        base_quant,
        &prefill_parallel,
        &[1],
        &runtime,
        "static_ctx",
        1.1,
    )?;
    let decode_batches = decode_batch_size_list(512);
    let decode_rows = static_worker_candidates(
        req,
        db,
        model_info,
        base_quant,
        &decode_parallel,
        &decode_batches,
        &runtime,
        "static_gen",
        1.08,
    )?;
    if prefill_rows.is_empty() || decode_rows.is_empty() {
        return Ok(Vec::new());
    }
    Ok(match_dense_disagg(req, &prefill_rows, &decode_rows))
}

fn disagg_parallel_configs(
    req: &NativeDefaultRequest,
    is_moe: bool,
    _decode: bool,
) -> Vec<ParallelConfig> {
    let max_worker_gpus = if matches!(req.system.as_str(), "gb200" | "gb300") {
        16
    } else {
        8
    };
    let base_gpu_list = if is_moe {
        vec![1_u32, 2, 4, 8]
    } else {
        vec![1_u32, 2, 4, 8, 16]
    };
    let num_gpu_list = base_gpu_list
        .into_iter()
        .filter(|v| *v <= req.total_gpus && *v <= max_worker_gpus)
        .collect::<Vec<_>>();
    let tp_list = if is_moe {
        vec![1_u32, 2, 4, 8]
    } else {
        vec![1_u32, 2, 4, 8, 16]
    }
    .into_iter()
    .filter(|v| *v <= max_worker_gpus)
    .collect::<Vec<_>>();
    let parallel_list = vec![1_u32, 2, 4, 8].into_iter().collect::<Vec<_>>();
    enumerate_parallel_config(&ParallelSearch {
        num_gpu_list,
        tp_list,
        pp_list: vec![1],
        dp_list: if is_moe {
            parallel_list.clone()
        } else {
            vec![1]
        },
        moe_tp_list: if is_moe {
            parallel_list.clone()
        } else {
            vec![1]
        },
        moe_ep_list: if is_moe { parallel_list } else { vec![1] },
        is_moe,
        backend: req.backend,
        ..ParallelSearch::default()
    })
}

#[allow(clippy::too_many_arguments)]
fn static_worker_candidates(
    req: &NativeDefaultRequest,
    db: &PerfDatabase,
    model_info: &ModelInfo,
    base_quant: ModelConfig,
    parallel_configs: &[ParallelConfig],
    b_list: &[i64],
    runtime: &RuntimeConfig,
    mode: &str,
    latency_correction_scale: f64,
) -> Result<Vec<StaticRow>> {
    let mut rows = Vec::new();
    for parallel in parallel_configs {
        let mut model_config = base_quant;
        model_config.tp_size = parallel.tp;
        model_config.pp_size = parallel.pp;
        model_config.attention_dp_size = parallel.dp;
        model_config.moe_tp_size = parallel.moe_tp;
        model_config.moe_ep_size = parallel.moe_ep;
        let Ok(model) = build_model(&req.model_path, model_info, model_config) else {
            continue;
        };
        for &batch_size in b_list {
            let mut runtime = runtime.clone();
            runtime.batch_size = batch_size;
            let Ok(summary) = run_static(db, &model, &runtime, mode, 32, latency_correction_scale)
            else {
                continue;
            };
            if summary.oom {
                break;
            }
            rows.push(static_summary_to_row(
                req, db, &model, &runtime, mode, &summary,
            ));
        }
    }
    Ok(rows)
}

fn static_summary_to_row(
    req: &NativeDefaultRequest,
    db: &PerfDatabase,
    model: &DenseModel,
    runtime: &RuntimeConfig,
    mode: &str,
    summary: &StaticSummary,
) -> StaticRow {
    let context_latency: f64 = summary.context_latency.values().sum();
    let context_energy: f64 = summary.context_energy.values().sum();
    let generation_latency: f64 = summary.generation_latency.values().sum();
    let generation_energy: f64 = summary.generation_energy.values().sum();
    let total_latency = context_latency + generation_latency;
    let total_energy = context_energy + generation_energy;
    let ttft = context_latency;
    let tpot = if runtime.osl <= 1 {
        0.0
    } else {
        generation_latency / (runtime.osl - 1) as f64
    };
    let num_generated_tokens = (runtime.osl - 1).max(0) as f64;
    let mut request_latency = ttft + tpot * num_generated_tokens;
    if request_latency == 0.0 {
        request_latency = total_latency;
    }
    let tp = model.config.tp_size;
    let pp = model.config.pp_size;
    let dp = model.config.attention_dp_size;
    let global_bs = runtime.batch_size * dp as i64;
    let concurrency = global_bs as f64;
    let seq_s = if request_latency == 0.0 {
        0.0
    } else {
        global_bs as f64 / request_latency * 1000.0 * pp as f64
    };
    let seq_s_gpu = seq_s / tp as f64 / pp as f64 / dp as f64;
    let mut tokens_s = if mode == "static_gen" {
        seq_s * (runtime.osl - 1) as f64
    } else {
        seq_s * runtime.osl as f64
    };
    if mode == "static_ctx" {
        tokens_s = seq_s;
    }
    let tokens_s_gpu = tokens_s / tp as f64 / pp as f64 / dp as f64;
    let tokens_s_user = if tpot == 0.0 { 0.0 } else { 1000.0 / tpot };
    let power_w = if total_latency > 0.0 {
        total_energy / total_latency
    } else {
        0.0
    };
    StaticRow {
        model: req.model_path.clone(),
        isl: runtime.isl,
        osl: runtime.osl,
        prefix: runtime.prefix,
        concurrency: round3(concurrency),
        bs: runtime.batch_size,
        global_bs,
        ttft: round3(ttft),
        tpot: round3(tpot),
        seq_s: round3(seq_s),
        seq_s_gpu: round3(seq_s_gpu),
        tokens_s: round3(tokens_s),
        tokens_s_gpu: round3(tokens_s_gpu),
        tokens_s_user: round3(tokens_s_user),
        request_latency: round3(request_latency),
        num_total_gpus: tp * pp * dp,
        tp,
        pp,
        dp,
        moe_tp: model.config.moe_tp_size,
        moe_ep: model.config.moe_ep_size,
        parallel: format!(
            "tp{}pp{}dp{}etp{}ep{}",
            tp, pp, dp, model.config.moe_tp_size, model.config.moe_ep_size
        ),
        gemm: model.config.gemm_quant_mode.to_string(),
        kvcache: model.config.kvcache_quant_mode.to_string(),
        fmha: model.config.fmha_quant_mode.to_string(),
        moe: model.config.moe_quant_mode.to_string(),
        comm: model.config.comm_quant_mode.to_string(),
        memory: round3(summary.memory.total),
        backend: db.backend.to_string(),
        version: db.version.clone(),
        system: db.system.clone(),
        power_w: round3(power_w),
    }
}

fn decode_batch_size_list(max_batch_size: i64) -> Vec<i64> {
    let mut out = Vec::new();
    out.extend(1..16);
    out.extend((16..32).step_by(2));
    out.extend((32..128).step_by(4));
    out.extend((128..512).step_by(8));
    out.push(512);
    out.into_iter().filter(|b| *b <= max_batch_size).collect()
}

fn match_dense_disagg(
    req: &NativeDefaultRequest,
    prefill_rows: &[StaticRow],
    decode_rows: &[StaticRow],
) -> Vec<DisaggResult> {
    let mut results = Vec::new();
    for (ttft, tpot) in constraint_pairs(req) {
        results.extend(match_dense_disagg_for_constraint(
            req,
            prefill_rows,
            decode_rows,
            ttft,
            tpot,
        ));
    }
    results
}

fn match_dense_disagg_for_constraint(
    req: &NativeDefaultRequest,
    prefill_rows: &[StaticRow],
    decode_rows: &[StaticRow],
    ttft_constraint: f64,
    tpot_constraint: f64,
) -> Vec<DisaggResult> {
    const PREFILL_DEGRADATION: f64 = 0.9;
    const DECODE_DEGRADATION: f64 = 0.92;
    const TTFT_CORRECTION: f64 = 1.8;
    const MAX_PREFILL_WORKERS: i64 = 32;

    let allowed_gpu_counts = [
        1_u32, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128,
    ]
    .into_iter()
    .filter(|g| *g <= req.total_gpus)
    .collect::<Vec<_>>();

    let mut prefill_candidates = prefill_rows
        .iter()
        .cloned()
        .map(|mut row| {
            row.ttft = round3(row.ttft * TTFT_CORRECTION);
            row.request_latency = round3(row.ttft);
            row
        })
        .filter(|row| row.ttft < ttft_constraint)
        .collect::<Vec<_>>();
    prefill_candidates.sort_by(|a, b| {
        b.seq_s_gpu
            .total_cmp(&a.seq_s_gpu)
            .then_with(|| a.global_bs.cmp(&b.global_bs))
    });
    prefill_candidates.truncate(MAX_PREFILL_WORKERS as usize);

    let mut decode_by_parallel: BTreeMap<String, Vec<StaticRow>> = BTreeMap::new();
    for row in decode_rows {
        if row.tpot < tpot_constraint && row.tpot > 0.0 {
            decode_by_parallel
                .entry(row.parallel.clone())
                .or_default()
                .push(row.clone());
        }
    }

    let mut all_category_results = Vec::new();
    for rows in decode_by_parallel.values_mut() {
        rows.sort_by(|a, b| b.seq_s_gpu.total_cmp(&a.seq_s_gpu));
        let mut category_results = Vec::new();
        for decode in rows.iter() {
            for prefill in &prefill_candidates {
                if let Some((prefill_workers, decode_workers)) = match_worker_counts(
                    req.total_gpus,
                    &allowed_gpu_counts,
                    prefill.seq_s,
                    prefill.num_total_gpus,
                    decode.seq_s,
                    decode.num_total_gpus,
                    PREFILL_DEGRADATION,
                    DECODE_DEGRADATION,
                ) {
                    category_results.push(build_disagg_result(
                        prefill,
                        prefill_workers,
                        decode,
                        decode_workers,
                        PREFILL_DEGRADATION,
                        DECODE_DEGRADATION,
                    ));
                }
            }
        }
        all_category_results.extend(category_results);
    }
    all_category_results.sort_by(|a, b| b.tokens_s_gpu.total_cmp(&a.tokens_s_gpu));
    all_category_results
}

#[allow(clippy::too_many_arguments)]
fn match_worker_counts(
    total_gpus: u32,
    allowed_gpu_counts: &[u32],
    prefill_throughput: f64,
    prefill_gpus: u32,
    decode_throughput: f64,
    decode_gpus: u32,
    prefill_degradation: f64,
    decode_degradation: f64,
) -> Option<(i64, i64)> {
    let mut best = None;
    let mut throughput_per_gpu_max = 0.0;
    for decode_workers in 1..=32_i64 {
        for prefill_workers in 1..=32_i64 {
            let num_gpu =
                prefill_gpus * prefill_workers as u32 + decode_gpus * decode_workers as u32;
            if num_gpu > total_gpus || !allowed_gpu_counts.contains(&num_gpu) {
                continue;
            }
            let prefill_corrected =
                prefill_throughput * prefill_workers as f64 * prefill_degradation;
            let decode_corrected = decode_throughput * decode_workers as f64 * decode_degradation;
            let throughput_per_gpu = prefill_corrected.min(decode_corrected) / num_gpu as f64;
            if throughput_per_gpu > throughput_per_gpu_max {
                throughput_per_gpu_max = throughput_per_gpu;
                best = Some((prefill_workers, decode_workers));
            }
        }
    }
    best
}

fn build_disagg_result(
    prefill: &StaticRow,
    prefill_workers: i64,
    decode: &StaticRow,
    decode_workers: i64,
    prefill_degradation: f64,
    decode_degradation: f64,
) -> DisaggResult {
    let seq_s = (prefill.seq_s * prefill_workers as f64 * prefill_degradation)
        .min(decode.seq_s * decode_workers as f64 * decode_degradation);
    let num_total_gpus = prefill.num_total_gpus * prefill_workers as u32
        + decode.num_total_gpus * decode_workers as u32;
    let seq_s_gpu = if num_total_gpus > 0 {
        seq_s / num_total_gpus as f64
    } else {
        0.0
    };
    let tokens_s = seq_s * prefill.osl as f64;
    let tokens_s_gpu = if num_total_gpus > 0 {
        tokens_s / num_total_gpus as f64
    } else {
        0.0
    };
    let request_latency = prefill.ttft + decode.tpot * (prefill.osl - 1).max(0) as f64;
    let decode_time = decode.tpot * (prefill.osl - 1).max(0) as f64;
    let total_time = prefill.ttft + decode_time;
    let power_w = if total_time > 0.0 {
        (prefill.power_w * prefill.ttft + decode.power_w * decode_time) / total_time
    } else {
        0.0
    };

    DisaggResult {
        model: prefill.model.clone(),
        isl: prefill.isl,
        osl: prefill.osl,
        prefix: prefill.prefix,
        concurrency: round3(decode.concurrency * decode_workers as f64),
        request_rate: round3(seq_s),
        prefill_bs: prefill.bs,
        prefill_global_bs: prefill.global_bs,
        prefill_workers,
        decode_bs: decode.bs,
        decode_global_bs: decode.global_bs,
        decode_workers,
        ttft: round3(prefill.ttft),
        tpot: round3(decode.tpot),
        request_latency: round3(request_latency),
        seq_s: round3(seq_s),
        seq_s_gpu: round3(seq_s_gpu),
        tokens_s: round3(tokens_s),
        tokens_s_gpu: round3(tokens_s_gpu),
        tokens_s_user: decode.tokens_s_user,
        prefill_seq_s_worker: prefill.seq_s,
        decode_seq_s_worker: decode.seq_s,
        num_total_gpus,
        prefill_tp: prefill.tp,
        prefill_pp: prefill.pp,
        prefill_dp: prefill.dp,
        prefill_parallel: prefill.parallel.clone(),
        prefill_memory: prefill.memory,
        decode_tp: decode.tp,
        decode_pp: decode.pp,
        decode_dp: decode.dp,
        decode_parallel: decode.parallel.clone(),
        decode_memory: decode.memory,
        backend: prefill.backend.clone(),
        version: prefill.version.clone(),
        system: prefill.system.clone(),
        power_w: round3(power_w),
        tokens_s_gpu_cluster: 0.0,
    }
}

fn find_best_agg_for_parallel(
    req: &NativeDefaultRequest,
    db: &PerfDatabase,
    model: &DenseModel,
    _parallel: ParallelConfig,
) -> Result<Vec<AggResult>> {
    let b_list_default = [
        (1, 16, 1),
        (16, 32, 4),
        (32, 64, 8),
        (64, 256, 16),
        (256, 512, 32),
        (512, 1024, 256),
    ];
    let mut b_list = Vec::new();
    for (start, stop, step) in b_list_default {
        let mut b = start;
        while b < stop {
            if b <= 512 {
                b_list.push(b);
            }
            b += step;
        }
    }
    b_list.push(1024);
    b_list.retain(|b| *b <= 512);

    let ctx_tokens_list = ctx_tokens_list_for_agg_sweep(req.isl, 512, req.enable_chunked_prefill);
    let constraints = constraint_pairs(req);
    let mut rows = Vec::new();
    let mut capped_b: Vec<i64> = Vec::new();
    for b in b_list {
        for &ctx_tokens in &ctx_tokens_list {
            if b as f64 - ceil_div(ctx_tokens, req.isl) < 0.0 {
                break;
            }
            if b > 1 && b as f64 - ceil_div(ctx_tokens, req.isl) < 1.0 {
                break;
            }
            let balance_score = req.isl as f64 * b as f64 / ctx_tokens as f64 / req.osl as f64;
            if balance_score > 1.0 {
                let gen_tokens = (b as f64 / balance_score).floor() as i64;
                if gen_tokens > 1 && capped_b.contains(&gen_tokens) {
                    continue;
                }
                capped_b.push(gen_tokens);
            }
            let Ok(row) = run_trtllm_agg(req, db, model, b, ctx_tokens) else {
                continue;
            };
            if row.memory >= db.system_spec.gpu.mem_capacity / (1_u64 << 30) as f64
                || (req.backend == BackendName::trtllm && kv_cache_oom(req, &row, db, model, b))
            {
                break;
            }
            let meets_search_sla = if req.request_latency.is_some_and(|v| v > 0.0) {
                constraints
                    .iter()
                    .any(|(ttft, tpot)| row.tpot <= *tpot && row.ttft <= *ttft)
            } else {
                row.ttft <= req.ttft
            };
            if meets_search_sla {
                rows.push(row);
            }
        }
    }
    Ok(rows)
}

fn run_trtllm_agg(
    req: &NativeDefaultRequest,
    db: &PerfDatabase,
    model: &DenseModel,
    b: i64,
    ctx_tokens: i64,
) -> Result<AggResult> {
    let isl = req.isl;
    let osl = req.osl;
    let prefix = req.prefix;
    let balance_score = isl as f64 * b as f64 / ctx_tokens as f64 / osl as f64;
    let max_seq_len = req.max_seq_len.unwrap_or(isl + osl);
    let max_num_tokens = TRTLLM_DEFAULT_MAX_NUM_TOKENS;
    let steps_to_finish_ctx = (isl as f64 * b as f64 / ctx_tokens as f64).ceil();

    let (
        num_mix_steps,
        num_mix_ctx_tokens,
        num_mix_gen_tokens,
        num_genonly_steps,
        num_genonly_tokens,
        num_mix_steps_for_tpot_calc,
    ) = if b > 1 {
        if steps_to_finish_ctx >= osl as f64 {
            (
                steps_to_finish_ctx,
                ctx_tokens,
                (b as f64 / (steps_to_finish_ctx / osl as f64))
                    .floor()
                    .max(1.0),
                0.0,
                0.0,
                steps_to_finish_ctx,
            )
        } else {
            let gen_tokens = b as f64 - (ctx_tokens as f64 / isl as f64).ceil();
            (
                steps_to_finish_ctx,
                ctx_tokens,
                gen_tokens,
                osl as f64 - steps_to_finish_ctx,
                b as f64,
                (steps_to_finish_ctx - 3.0).max(1.0),
            )
        }
    } else {
        (1.0, ctx_tokens, 0.0, (osl - 1) as f64, 1.0, 0.0)
    };

    let (mix_step_latency, mix_step_energy) = mix_step_latency(
        db,
        model,
        num_mix_ctx_tokens,
        num_mix_gen_tokens as i64,
        isl,
        osl,
        prefix,
        req.database_mode,
    )?;
    let (genonly_step_latency, genonly_step_energy) = genonly_step_latency(
        db,
        model,
        num_genonly_tokens as i64,
        isl,
        osl,
        req.database_mode,
    )?;

    let mut ttft = mix_step_latency * (isl as f64 / ctx_tokens as f64).ceil();
    let correction_factor = (2.0 + (steps_to_finish_ctx - 3.0) / 2.0 / 10.0).min(4.0);
    ttft *= correction_factor;
    let tpot = (mix_step_latency * num_mix_steps_for_tpot_calc
        + genonly_step_latency * num_genonly_steps)
        / (num_mix_steps_for_tpot_calc + num_genonly_steps);
    let mut output_throughput = 1000.0
        / (num_mix_steps * mix_step_latency + num_genonly_steps * genonly_step_latency)
        * b as f64
        * (osl - 1) as f64;
    let total_energy = num_mix_steps * mix_step_energy + num_genonly_steps * genonly_step_energy;
    let total_latency = num_mix_steps * mix_step_latency + num_genonly_steps * genonly_step_latency;
    let power_w = if total_latency > 0.0 {
        total_energy / total_latency
    } else {
        0.0
    };

    let mut num_ctx_requests = (ctx_tokens as f64 / isl as f64).ceil();
    let mut num_gen_requests = b as f64 - num_ctx_requests;
    if b == 1 {
        num_ctx_requests = 1.0;
        num_gen_requests = 1.0;
    }
    let scale_factor = model.config.pp_size as f64 * model.config.attention_dp_size as f64;
    output_throughput *= scale_factor;
    let concurrency = b as f64 * scale_factor;
    let request_rate = output_throughput / (osl - 1) as f64;
    let num_tokens = if b > 1 {
        num_gen_requests + ctx_tokens as f64
    } else {
        ctx_tokens as f64
    };
    let memory = get_memory_usage(
        model,
        db,
        b,
        1,
        isl,
        osl,
        max_num_tokens,
        prefix,
        Some(max_seq_len),
    );
    let tp = model.config.tp_size;
    let pp = model.config.pp_size;
    let dp = model.config.attention_dp_size;
    let num_total_gpus = tp * pp * dp;
    let tokens_s_gpu = output_throughput / pp as f64 / tp as f64 / dp as f64;
    let tokens_s_user = 1000.0 / tpot;
    let seq_s = request_rate;
    let seq_s_gpu = seq_s / pp as f64 / tp as f64 / dp as f64;
    let request_latency = ttft + tpot * (osl - 1).max(0) as f64;

    Ok(AggResult {
        model: model.model_path.clone(),
        isl,
        osl,
        prefix,
        concurrency: round3(concurrency),
        request_rate: round3(request_rate),
        bs: b,
        global_bs: b * model.config.attention_dp_size as i64,
        ttft: round3(ttft),
        tpot: round3(tpot),
        seq_s: round3(seq_s),
        seq_s_gpu: round3(seq_s_gpu),
        tokens_s: round3(output_throughput),
        tokens_s_gpu: round3(tokens_s_gpu),
        tokens_s_user: round3(tokens_s_user),
        request_latency: round3(request_latency),
        num_total_gpus,
        tp,
        pp,
        dp,
        moe_tp: model.config.moe_tp_size,
        moe_ep: model.config.moe_ep_size,
        parallel: format!(
            "tp{}pp{}dp{}etp{}ep{}",
            tp, pp, dp, model.config.moe_tp_size, model.config.moe_ep_size
        ),
        gemm: model.config.gemm_quant_mode.to_string(),
        kvcache: model.config.kvcache_quant_mode.to_string(),
        fmha: model.config.fmha_quant_mode.to_string(),
        moe: model.config.moe_quant_mode.to_string(),
        comm: model.config.comm_quant_mode.to_string(),
        memory: round3(memory.total),
        balance_score: round3(balance_score),
        num_ctx_reqs: round3(num_ctx_requests),
        num_gen_reqs: round3(num_gen_requests),
        num_tokens: round3(num_tokens),
        ctx_tokens,
        gen_tokens: round3(num_gen_requests),
        backend: db.backend.to_string(),
        version: db.version.clone(),
        system: db.system.clone(),
        power_w: round3(power_w),
        tokens_s_gpu_cluster: 0.0,
    })
}

fn mix_step_latency(
    db: &PerfDatabase,
    model: &DenseModel,
    ctx_tokens: i64,
    gen_tokens: i64,
    isl: i64,
    osl: i64,
    prefix: i64,
    database_mode: DatabaseMode,
) -> Result<(f64, f64)> {
    let num_tokens = ctx_tokens + gen_tokens;
    let prefix_for_mix = prefix * (ctx_tokens as f64 / isl as f64).floor() as i64;
    let summary = run_static(
        db,
        model,
        &RuntimeConfig {
            batch_size: 1,
            beam_width: 1,
            isl: num_tokens,
            osl: 1,
            prefix: prefix_for_mix,
            database_mode,
            seq_imbalance_correction_scale: 1.0,
            gen_seq_imbalance_correction_scale: 1.0,
        },
        "static_ctx",
        32,
        1.0,
    )?;
    let mut non_attention_latency = 0.0;
    let mut non_attention_energy = 0.0;
    for (name, latency) in &summary.context_latency {
        if !is_context_attention_op(name) {
            non_attention_latency += latency;
            non_attention_energy += summary.context_energy.get(name).copied().unwrap_or(0.0);
        }
    }

    let batch_size = (ctx_tokens as f64 / isl as f64).ceil() as i64;
    let ctx_summary = run_static(
        db,
        model,
        &RuntimeConfig {
            batch_size,
            beam_width: 1,
            isl,
            osl: 1,
            prefix,
            database_mode,
            seq_imbalance_correction_scale: 1.0,
            gen_seq_imbalance_correction_scale: 1.0,
        },
        "static_ctx",
        32,
        1.0,
    )?;
    let scale_factor = (isl as f64 / ctx_tokens as f64).ceil();
    let (ctx_attention_latency, ctx_attention_energy) =
        if let Some(ctx_attention_name) = context_attention_op_name(&ctx_summary.context_latency) {
            (
                ctx_summary.context_latency[ctx_attention_name] / scale_factor,
                ctx_summary
                    .context_energy
                    .get(ctx_attention_name)
                    .copied()
                    .unwrap_or(0.0)
                    / scale_factor,
            )
        } else {
            (0.0, 0.0)
        };

    let mut gen_attention_latency = 0.0;
    let mut gen_attention_energy = 0.0;
    if gen_tokens > 0 {
        let gen_summary = run_static(
            db,
            model,
            &RuntimeConfig {
                batch_size: gen_tokens,
                beam_width: 1,
                isl: isl + osl / 2,
                osl: 2,
                prefix: 0,
                database_mode,
                seq_imbalance_correction_scale: 1.0,
                gen_seq_imbalance_correction_scale: 1.0,
            },
            "static_gen",
            32,
            1.0,
        )?;
        if let Some(gen_attention_name) =
            generation_attention_op_name(&gen_summary.generation_latency)
        {
            gen_attention_latency = gen_summary.generation_latency[gen_attention_name];
            gen_attention_energy = gen_summary
                .generation_energy
                .get(gen_attention_name)
                .copied()
                .unwrap_or(0.0);
        }
    }

    Ok((
        non_attention_latency + ctx_attention_latency + gen_attention_latency,
        non_attention_energy + ctx_attention_energy + gen_attention_energy,
    ))
}

fn is_context_attention_op(name: &str) -> bool {
    matches!(name, "context_attention")
}

fn context_attention_op_name(latencies: &BTreeMap<String, f64>) -> Option<&str> {
    latencies
        .contains_key("context_attention")
        .then_some("context_attention")
}

fn generation_attention_op_name(latencies: &BTreeMap<String, f64>) -> Option<&str> {
    latencies
        .contains_key("generation_attention")
        .then_some("generation_attention")
}

fn genonly_step_latency(
    db: &PerfDatabase,
    model: &DenseModel,
    gen_tokens: i64,
    isl: i64,
    osl: i64,
    database_mode: DatabaseMode,
) -> Result<(f64, f64)> {
    if gen_tokens <= 0 {
        return Ok((0.0, 0.0));
    }
    let summary = run_static(
        db,
        model,
        &RuntimeConfig {
            batch_size: gen_tokens,
            beam_width: 1,
            isl: isl + osl / 2,
            osl: 2,
            prefix: 0,
            database_mode,
            seq_imbalance_correction_scale: 1.0,
            gen_seq_imbalance_correction_scale: 1.0,
        },
        "static_gen",
        32,
        1.0,
    )?;
    Ok((
        summary.generation_latency.values().sum(),
        summary.generation_energy.values().sum(),
    ))
}

fn run_static(
    db: &PerfDatabase,
    model: &DenseModel,
    runtime: &RuntimeConfig,
    mode: &str,
    stride: i64,
    latency_correction_scale: f64,
) -> Result<StaticSummary> {
    let mut context_latency = BTreeMap::new();
    let mut context_energy = BTreeMap::new();
    let mut generation_latency = BTreeMap::new();
    let mut generation_energy = BTreeMap::new();

    if mode == "static_ctx" || mode == "static" {
        let effective_isl = runtime.isl - runtime.prefix;
        for op in &model.context_ops {
            let x = if op.name.contains("logits_gemm") {
                runtime.batch_size
            } else {
                runtime.batch_size * effective_isl
            };
            let result = op
                .query_context(db, runtime, x, effective_isl)
                .with_context(|| format!("querying context op {}", op.name))?;
            *context_latency.entry(op.name.clone()).or_insert(0.0) += result.latency;
            *context_energy.entry(op.name.clone()).or_insert(0.0) += result.energy;
        }
    }

    if mode == "static_gen" || mode == "static" {
        let batch_size = runtime.batch_size * (model.config.nextn as i64 + 1);
        let mut i = 0;
        while i < runtime.osl - 1 {
            let mut step_latency = BTreeMap::new();
            let mut step_energy = BTreeMap::new();
            for op in &model.generation_ops {
                let result = op
                    .query_generation(
                        db,
                        runtime,
                        batch_size * runtime.beam_width,
                        runtime.isl + i + 1,
                    )
                    .with_context(|| format!("querying generation op {}", op.name))?;
                *step_latency.entry(op.name.clone()).or_insert(0.0) += result.latency;
                *step_energy.entry(op.name.clone()).or_insert(0.0) += result.energy;
            }
            let repeat_count = stride.min(runtime.osl - 1 - i) as f64;
            for (name, latency) in step_latency {
                *generation_latency.entry(name.clone()).or_insert(0.0) += latency * repeat_count;
                *generation_energy.entry(name.clone()).or_insert(0.0) +=
                    step_energy.get(&name).copied().unwrap_or(0.0) * repeat_count;
            }
            i += stride;
        }
    }

    if latency_correction_scale != 1.0 {
        for value in context_latency.values_mut() {
            *value *= latency_correction_scale;
        }
        for value in context_energy.values_mut() {
            *value *= latency_correction_scale;
        }
        for value in generation_latency.values_mut() {
            *value *= latency_correction_scale;
        }
        for value in generation_energy.values_mut() {
            *value *= latency_correction_scale;
        }
    }

    let memory = match mode {
        "static_ctx" => get_memory_usage(
            model,
            db,
            runtime.batch_size,
            runtime.beam_width,
            runtime.isl,
            1,
            0,
            runtime.prefix,
            None,
        ),
        "static_gen" => get_memory_usage(
            model,
            db,
            runtime.batch_size,
            runtime.beam_width,
            runtime.isl,
            runtime.osl,
            runtime.batch_size * runtime.beam_width,
            runtime.prefix,
            None,
        ),
        _ => get_memory_usage(
            model,
            db,
            runtime.batch_size,
            runtime.beam_width,
            runtime.isl,
            runtime.osl,
            0,
            runtime.prefix,
            None,
        ),
    };
    let capacity_gib = db.system_spec.gpu.mem_capacity / (1_u64 << 30) as f64;
    Ok(StaticSummary {
        context_latency,
        context_energy,
        generation_latency,
        generation_energy,
        memory,
        oom: memory.total >= capacity_gib,
        kv_cache_oom: false,
    })
}

impl Op {
    fn query_context(
        &self,
        db: &PerfDatabase,
        runtime: &RuntimeConfig,
        x: i64,
        effective_isl: i64,
    ) -> Result<PerformanceResult> {
        let result = match self.kind {
            OpKind::Embedding { column_size, .. } => {
                db.query_mem_op((x * column_size * 2) as f64, runtime.database_mode)
            }
            OpKind::ElementWise {
                dim_in,
                dim_out,
                scale_num_tokens,
            } => db.query_mem_op(
                ((x / scale_num_tokens) * (dim_in + dim_out) * 2) as f64,
                runtime.database_mode,
            ),
            OpKind::Gemm { n, k, quant_mode } => {
                db.query_gemm(x, n, k, quant_mode, runtime.database_mode)?
            }
            OpKind::ContextAttention {
                n,
                n_kv,
                kvcache_quant_mode,
                fmha_quant_mode,
                window_size,
                head_size,
                use_qk_norm,
            } => {
                let mut r = db.query_context_attention(
                    runtime.batch_size,
                    effective_isl,
                    runtime.prefix,
                    n,
                    n_kv,
                    kvcache_quant_mode,
                    fmha_quant_mode,
                    runtime.database_mode,
                    window_size,
                    head_size,
                )?;
                let q_num = n * head_size;
                let k_num = n_kv * head_size;
                let v_num = n_kv * head_size;
                let mut extra_latency = 0.0;
                if use_qk_norm {
                    let qk_norm_latency = db
                        .query_mem_op((q_num * 2) as f64, runtime.database_mode)
                        .latency
                        * 2.0
                        + db.query_mem_op((k_num * 2) as f64, runtime.database_mode)
                            .latency
                            * 2.0;
                    extra_latency += qk_norm_latency * 2.0;
                }
                let apply_rope_latency = db
                    .query_mem_op((q_num * 2 + k_num * 2) as f64, runtime.database_mode)
                    .latency
                    * 2.0;
                let kv_write_latency = db
                    .query_mem_op(
                        k_num as f64 * fmha_quant_mode.mapping().memory,
                        runtime.database_mode,
                    )
                    .latency
                    + db.query_mem_op(
                        v_num as f64 * fmha_quant_mode.mapping().memory,
                        runtime.database_mode,
                    )
                    .latency;
                extra_latency += apply_rope_latency + kv_write_latency;
                r = r + PerformanceResult::new(extra_latency, 0.0) * 1.1;
                r
            }
            OpKind::MlaModule {
                is_context: true,
                num_heads,
                kvcache_quant_mode,
                fmha_quant_mode,
                gemm_quant_mode,
                ..
            } => db.query_context_mla_module(
                runtime.batch_size,
                effective_isl,
                runtime.prefix,
                num_heads,
                kvcache_quant_mode,
                fmha_quant_mode,
                gemm_quant_mode,
                runtime.database_mode,
            )?,
            OpKind::Moe {
                hidden_size,
                inter_size,
                topk,
                num_experts,
                moe_tp_size,
                moe_ep_size,
                quant_mode,
                ref workload_distribution,
                attention_dp_size,
                is_gated,
            } => db.query_moe(
                x * attention_dp_size as i64,
                hidden_size,
                inter_size,
                topk,
                num_experts,
                moe_tp_size,
                moe_ep_size,
                quant_mode,
                workload_distribution,
                true,
                None,
                runtime.database_mode,
                is_gated,
                false,
            )?,
            OpKind::MoeDispatch {
                hidden_size,
                moe_tp_size,
                moe_ep_size,
                attention_dp_size,
                pre_dispatch,
            } => query_moe_dispatch(
                db,
                runtime.database_mode,
                x,
                hidden_size,
                moe_tp_size,
                moe_ep_size,
                attention_dp_size,
                pre_dispatch,
            )?,
            OpKind::CustomAllReduce { tp_size, .. } if tp_size == 1 => {
                PerformanceResult::new(0.0, 0.0)
            }
            OpKind::P2P { pp_size, .. } if pp_size == 1 => PerformanceResult::new(0.0, 0.0),
            OpKind::CustomAllReduce { h, tp_size } => db.query_custom_allreduce(
                CommQuantMode::half,
                tp_size,
                x * h,
                runtime.database_mode,
            )?,
            OpKind::P2P { h, .. } => db.query_p2p(x * h * 2, runtime.database_mode)?,
            OpKind::Overlap { .. } => PerformanceResult::new(0.0, 0.0),
            _ => PerformanceResult::new(0.0, 0.0),
        };
        Ok(result * self.scale_factor * runtime.seq_imbalance_correction_scale)
    }

    fn query_generation(
        &self,
        db: &PerfDatabase,
        runtime: &RuntimeConfig,
        x: i64,
        s: i64,
    ) -> Result<PerformanceResult> {
        let result = match self.kind {
            OpKind::Embedding { column_size, .. } => {
                db.query_mem_op((x * column_size * 2) as f64, runtime.database_mode)
            }
            OpKind::ElementWise {
                dim_in,
                dim_out,
                scale_num_tokens,
            } => db.query_mem_op(
                ((x / scale_num_tokens) * (dim_in + dim_out) * 2) as f64,
                runtime.database_mode,
            ),
            OpKind::Gemm { n, k, quant_mode } => {
                db.query_gemm(x, n, k, quant_mode, runtime.database_mode)?
            }
            OpKind::GenerationAttention {
                n,
                n_kv,
                kvcache_quant_mode,
                window_size,
                head_size,
            } => db.query_generation_attention(
                x,
                s,
                n,
                n_kv,
                kvcache_quant_mode,
                runtime.database_mode,
                window_size,
                head_size,
            )?,
            OpKind::MlaModule {
                is_context: false,
                num_heads,
                kvcache_quant_mode,
                fmha_quant_mode,
                gemm_quant_mode,
                ..
            } => db.query_generation_mla_module(
                x,
                s,
                num_heads,
                kvcache_quant_mode,
                fmha_quant_mode,
                gemm_quant_mode,
                runtime.database_mode,
            )?,
            OpKind::Moe {
                hidden_size,
                inter_size,
                topk,
                num_experts,
                moe_tp_size,
                moe_ep_size,
                quant_mode,
                ref workload_distribution,
                attention_dp_size,
                is_gated,
            } => db.query_moe(
                x * attention_dp_size as i64,
                hidden_size,
                inter_size,
                topk,
                num_experts,
                moe_tp_size,
                moe_ep_size,
                quant_mode,
                workload_distribution,
                false,
                None,
                runtime.database_mode,
                is_gated,
                false,
            )?,
            OpKind::MoeDispatch {
                hidden_size,
                moe_tp_size,
                moe_ep_size,
                attention_dp_size,
                pre_dispatch,
            } => query_moe_dispatch(
                db,
                runtime.database_mode,
                x,
                hidden_size,
                moe_tp_size,
                moe_ep_size,
                attention_dp_size,
                pre_dispatch,
            )?,
            OpKind::CustomAllReduce { tp_size, .. } if tp_size == 1 => {
                PerformanceResult::new(0.0, 0.0)
            }
            OpKind::P2P { pp_size, .. } if pp_size == 1 => PerformanceResult::new(0.0, 0.0),
            OpKind::CustomAllReduce { h, tp_size } => db.query_custom_allreduce(
                CommQuantMode::half,
                tp_size,
                x * h,
                runtime.database_mode,
            )?,
            OpKind::P2P { h, .. } => db.query_p2p(x * h * 2, runtime.database_mode)?,
            OpKind::Overlap {
                ref group_a,
                ref group_b,
            } => query_overlap_generation(group_a, group_b, db, runtime, x, s)?,
            _ => PerformanceResult::new(0.0, 0.0),
        };
        Ok(result * self.scale_factor * runtime.gen_seq_imbalance_correction_scale)
    }

    fn weights(&self) -> f64 {
        match self.kind {
            OpKind::Embedding {
                row_size,
                column_size,
            } => (row_size * column_size * 2) as f64 * self.scale_factor,
            OpKind::Gemm { n, k, quant_mode } => {
                n as f64 * k as f64 * quant_mode.mapping().memory * self.scale_factor
            }
            OpKind::Moe {
                hidden_size,
                inter_size,
                num_experts,
                moe_tp_size,
                moe_ep_size,
                quant_mode,
                is_gated,
                ..
            } => {
                let num_gemms = if is_gated { 3.0 } else { 2.0 };
                hidden_size as f64
                    * inter_size as f64
                    * num_experts as f64
                    * quant_mode.mapping().memory
                    * num_gemms
                    / moe_ep_size as f64
                    / moe_tp_size as f64
                    * self.scale_factor
            }
            OpKind::MlaModule { weight_bytes, .. } => weight_bytes * self.scale_factor,
            OpKind::Overlap {
                ref group_a,
                ref group_b,
            } => {
                group_a
                    .iter()
                    .chain(group_b.iter())
                    .map(Op::weights)
                    .sum::<f64>()
                    * self.scale_factor
            }
            _ => 0.0,
        }
    }
}

fn query_moe_dispatch(
    db: &PerfDatabase,
    database_mode: DatabaseMode,
    num_tokens: i64,
    hidden_size: i64,
    moe_tp_size: u32,
    moe_ep_size: u32,
    attention_dp_size: u32,
    pre_dispatch: bool,
) -> Result<PerformanceResult> {
    let num_gpus = moe_tp_size * moe_ep_size;
    let attention_tp_size = (moe_tp_size * moe_ep_size) / attention_dp_size.max(1);
    let volume = num_tokens * hidden_size;
    if db.backend == BackendName::vllm {
        let mut result = PerformanceResult::new(0.0, 0.0);
        if attention_tp_size > 1 {
            result = result
                + db.query_custom_allreduce(CommQuantMode::half, num_gpus, volume, database_mode)?;
        }
        if attention_dp_size > 1 {
            let op = if pre_dispatch {
                "all_gather"
            } else {
                "reduce_scatter"
            };
            result = result
                + db.query_nccl(
                    CommQuantMode::half,
                    num_gpus,
                    op,
                    volume * attention_dp_size as i64,
                    database_mode,
                )?;
        }
        return Ok(result);
    }

    if attention_tp_size > 1 {
        db.query_custom_allreduce(CommQuantMode::half, num_gpus, volume, database_mode)
    } else if attention_dp_size > 1 {
        let op = if pre_dispatch {
            "all_gather"
        } else {
            "reduce_scatter"
        };
        db.query_nccl(
            CommQuantMode::half,
            num_gpus,
            op,
            volume * attention_dp_size as i64,
            database_mode,
        )
    } else {
        Ok(PerformanceResult::new(0.0, 0.0))
    }
}

fn query_overlap_generation(
    group_a: &[Op],
    group_b: &[Op],
    db: &PerfDatabase,
    runtime: &RuntimeConfig,
    x: i64,
    s: i64,
) -> Result<PerformanceResult> {
    let mut latency_a = 0.0;
    let mut energy_a = 0.0;
    for op in group_a {
        let result = op
            .query_generation(db, runtime, x, s)
            .with_context(|| format!("querying overlapped generation op {}", op.name))?;
        latency_a += result.latency;
        energy_a += result.energy;
    }

    let mut latency_b = 0.0;
    let mut energy_b = 0.0;
    for op in group_b {
        let result = op
            .query_generation(db, runtime, x, s)
            .with_context(|| format!("querying overlapped generation op {}", op.name))?;
        latency_b += result.latency;
        energy_b += result.energy;
    }

    Ok(PerformanceResult::new(
        latency_a.max(latency_b),
        energy_a + energy_b,
    ))
}

fn build_model(model_path: &str, info: &ModelInfo, config: ModelConfig) -> Result<DenseModel> {
    if info.architecture == "DeepseekV3ForCausalLM" {
        build_deepseek_v3_model(model_path, info, config)
    } else if is_moe_model(info) {
        build_moe_model(model_path, info, config)
    } else {
        build_dense_model(model_path, info, config)
    }
}

fn build_dense_model(
    model_path: &str,
    info: &ModelInfo,
    config: ModelConfig,
) -> Result<DenseModel> {
    let family = architecture_to_family(&info.architecture);
    if family != "LLAMA" && family != "GPT" {
        bail!(
            "native dense estimator only supports GPT/LLAMA-style models, got {}",
            info.architecture
        );
    }
    let h = info.hidden_size as i64;
    let use_qk_norm = matches!(
        info.architecture.as_str(),
        "Qwen3ForCausalLM" | "Qwen3MoeForCausalLM"
    );
    let tp = config.tp_size as i64;
    let pp = config.pp_size as i64;
    let num_heads = info.n as i64;
    let head_size = info.d as i64;
    let n_kv_total = if info.n_kv == 0 { info.n } else { info.n_kv } as i64;
    let num_kv_heads_per_gpu = (n_kv_total + tp - 1) / tp;
    let layers = info.layers as f64;
    let vocab = info.vocab as i64;
    let inter = info.inter_size as i64;
    let mtp_scale_factor = mtp_scale_factor(config.nextn, layers);

    let mut context_ops = vec![
        op_embedding("context_embedding", 1.0, vocab / tp, h),
        op_elementwise("context_add_norm_1", layers, 2 * h, 2 * h),
        op_gemm(
            "context_qkv_gemm",
            layers,
            num_heads * head_size / tp + head_size * num_kv_heads_per_gpu * 2,
            h,
            config.gemm_quant_mode,
        ),
        op_context_attention(
            "context_attention",
            layers,
            num_heads / tp,
            num_kv_heads_per_gpu,
            config.kvcache_quant_mode,
            config.fmha_quant_mode,
            head_size,
            use_qk_norm,
        ),
        op_gemm(
            "context_proj_gemm",
            layers,
            h,
            num_heads * head_size / tp,
            config.gemm_quant_mode,
        ),
        op_elementwise("context_add_norm_2", layers, 2 * h, 2 * h),
    ];
    if family == "GPT" {
        context_ops.push(op_gemm(
            "context_ffn1_gemm",
            layers,
            inter / tp,
            h,
            config.gemm_quant_mode,
        ));
        context_ops.push(op_elementwise(
            "context_act",
            layers,
            inter / tp,
            inter / tp,
        ));
    } else {
        context_ops.push(op_gemm(
            "context_gate_ffn1_gemm",
            layers,
            2 * inter / tp,
            h,
            config.gemm_quant_mode,
        ));
        context_ops.push(op_elementwise(
            "context_act_gate",
            layers,
            2 * inter / tp,
            inter / tp,
        ));
    }
    context_ops.push(op_gemm(
        "context_ffn2_gemm",
        layers,
        h,
        inter / tp,
        config.gemm_quant_mode,
    ));
    context_ops.push(op_gemm(
        "context_logits_gemm",
        1.0,
        vocab / tp,
        h,
        GemmQuantMode::bfloat16,
    ));
    context_ops.push(op_custom_ar("context_embedding_ar", 1.0, h, config.tp_size));
    context_ops.push(op_custom_ar("context_ar_1", layers, h, config.tp_size));
    context_ops.push(op_custom_ar("context_ar_2", layers, h, config.tp_size));
    context_ops.push(op_p2p("context_p2p", (pp - 1) as f64, h, config.pp_size));

    let mut generation_ops = vec![
        op_embedding("generation_embedding", mtp_scale_factor, vocab / tp, h),
        op_elementwise(
            "generation_add_norm_1",
            layers * mtp_scale_factor,
            2 * h,
            2 * h,
        ),
        op_gemm(
            "generation_qkv_gemm",
            layers * mtp_scale_factor,
            num_heads * head_size / tp + head_size * num_kv_heads_per_gpu * 2,
            h,
            config.gemm_quant_mode,
        ),
        op_generation_attention(
            "generation_attention",
            layers * mtp_scale_factor,
            num_heads / tp,
            num_kv_heads_per_gpu,
            config.kvcache_quant_mode,
            head_size,
        ),
        op_gemm(
            "generation_proj_gemm",
            layers * mtp_scale_factor,
            h,
            num_heads * head_size / tp,
            config.gemm_quant_mode,
        ),
        op_elementwise(
            "generation_add_norm_2",
            layers * mtp_scale_factor,
            2 * h,
            2 * h,
        ),
    ];
    if family == "GPT" {
        generation_ops.push(op_gemm(
            "generation_ffn1_gemm",
            layers * mtp_scale_factor,
            inter / tp,
            h,
            config.gemm_quant_mode,
        ));
        generation_ops.push(op_elementwise(
            "generation_act",
            layers * mtp_scale_factor,
            inter / tp,
            inter / tp,
        ));
    } else {
        generation_ops.push(op_gemm(
            "generation_gate_ffn1_gemm",
            layers * mtp_scale_factor,
            2 * inter / tp,
            h,
            config.gemm_quant_mode,
        ));
        generation_ops.push(op_elementwise(
            "generation_act_gate",
            layers * mtp_scale_factor,
            2 * inter / tp,
            inter / tp,
        ));
    }
    generation_ops.push(op_gemm(
        "generation_ffn2_gemm",
        layers * mtp_scale_factor,
        h,
        inter / tp,
        config.gemm_quant_mode,
    ));
    generation_ops.push(op_gemm(
        "generation_logits_gemm",
        mtp_scale_factor,
        vocab / tp,
        h,
        GemmQuantMode::bfloat16,
    ));
    generation_ops.push(op_custom_ar(
        "generation_embedding_ar",
        mtp_scale_factor,
        h,
        config.tp_size,
    ));
    generation_ops.push(op_custom_ar(
        "generation_ar_1",
        layers * mtp_scale_factor,
        h,
        config.tp_size,
    ));
    generation_ops.push(op_custom_ar(
        "generation_ar_2",
        layers * mtp_scale_factor,
        h,
        config.tp_size,
    ));
    generation_ops.push(op_p2p(
        "generation_p2p",
        (pp - 1) as f64 * mtp_scale_factor,
        h,
        config.pp_size,
    ));

    Ok(DenseModel {
        model_path: model_path.to_string(),
        model_family: family.to_string(),
        architecture: info.architecture.clone(),
        config,
        num_layers: layers,
        num_heads,
        num_kv_heads: n_kv_total,
        head_size,
        hidden_size: h,
        inter_size: inter,
        vocab_size: vocab,
        kvcache_elements_per_token_override: None,
        context_ops,
        generation_ops,
    })
}

fn build_deepseek_v3_model(
    model_path: &str,
    info: &ModelInfo,
    config: ModelConfig,
) -> Result<DenseModel> {
    if config.tp_size * config.attention_dp_size != config.moe_tp_size * config.moe_ep_size {
        bail!(
            "tp_size ({}) * attention_dp_size ({}) must equal moe_tp_size ({}) * moe_ep_size ({})",
            config.tp_size,
            config.attention_dp_size,
            config.moe_tp_size,
            config.moe_ep_size
        );
    }
    if info.num_experts < config.moe_ep_size as u64 {
        bail!("moe_ep_size cannot exceed number of experts");
    }

    let h = info.hidden_size as i64;
    let tp = config.tp_size as i64;
    let pp = config.pp_size as i64;
    let layers = info.layers as f64;
    let vocab = info.vocab as i64;
    let moe_inter = if info.moe_inter_size > 0 {
        info.moe_inter_size as i64
    } else {
        info.inter_size as i64
    };
    let topk = info.topk as i64;
    let num_experts = info.num_experts as i64;
    let mla_heads = 128 / tp;
    let mtp_scale_factor = mtp_scale_factor(config.nextn, layers);
    let workload_distribution = "power_law_1.01";
    let mla_weight_bytes = deepseek_mla_weight_bytes(h, tp, config.gemm_quant_mode);

    let context_ops = vec![
        op_embedding("context_embedding", 1.0, vocab, h),
        op_elementwise("context_add_norm_1", layers, 2 * h, 2 * h),
        op_mla_module(
            "context_mla_module",
            layers,
            true,
            mla_heads,
            config.kvcache_quant_mode,
            config.fmha_quant_mode,
            config.gemm_quant_mode,
            mla_weight_bytes,
        ),
        op_elementwise("context_add_norm_2", layers, 2 * h, 2 * h),
        op_gemm(
            "context_shared_gate_up_gemm",
            layers,
            2 * moe_inter / tp,
            h,
            config.gemm_quant_mode,
        ),
        op_elementwise(
            "context_shared_act_gate",
            layers,
            2 * moe_inter / tp,
            moe_inter / tp,
        ),
        op_gemm(
            "context_shared_ffn2_gemm",
            layers,
            h,
            moe_inter / tp,
            config.gemm_quant_mode,
        ),
        op_gemm(
            "context_router_gemm",
            layers,
            num_experts,
            h,
            GemmQuantMode::bfloat16,
        ),
        op_moe_dispatch(
            "context_moe_pre_dispatch",
            layers,
            h,
            config.moe_tp_size,
            config.moe_ep_size,
            config.attention_dp_size,
            true,
        ),
        op_moe(
            "context_moe",
            layers,
            h,
            moe_inter,
            topk,
            num_experts,
            config.moe_tp_size,
            config.moe_ep_size,
            config.moe_quant_mode,
            workload_distribution,
            config.attention_dp_size,
            true,
        ),
        op_moe_dispatch(
            "context_moe_post_dispatch",
            layers,
            h,
            config.moe_tp_size,
            config.moe_ep_size,
            config.attention_dp_size,
            false,
        ),
        op_gemm(
            "context_logits_gemm",
            1.0,
            vocab / tp,
            h,
            GemmQuantMode::bfloat16,
        ),
        op_p2p("context_p2p", (pp - 1) as f64, h, config.pp_size),
    ];

    let gen_shared_ops = vec![
        op_gemm(
            "generation_shared_gate_up_gemm",
            layers * mtp_scale_factor,
            2 * moe_inter / tp,
            h,
            config.gemm_quant_mode,
        ),
        op_elementwise(
            "generation_shared_act_gate",
            layers * mtp_scale_factor,
            2 * moe_inter / tp,
            moe_inter / tp,
        ),
        op_gemm(
            "generation_shared_ffn2_gemm",
            layers * mtp_scale_factor,
            h,
            moe_inter / tp,
            config.gemm_quant_mode,
        ),
    ];

    let gen_routed_ops = vec![
        op_gemm(
            "generation_router_gemm",
            layers * mtp_scale_factor,
            num_experts,
            h,
            GemmQuantMode::bfloat16,
        ),
        op_moe_dispatch(
            "generation_moe_pre_dispatch",
            layers * mtp_scale_factor,
            h,
            config.moe_tp_size,
            config.moe_ep_size,
            config.attention_dp_size,
            true,
        ),
        op_moe(
            "generation_moe",
            layers * mtp_scale_factor,
            h,
            moe_inter,
            topk,
            num_experts,
            config.moe_tp_size,
            config.moe_ep_size,
            config.moe_quant_mode,
            workload_distribution,
            config.attention_dp_size,
            true,
        ),
        op_moe_dispatch(
            "generation_moe_post_dispatch",
            layers * mtp_scale_factor,
            h,
            config.moe_tp_size,
            config.moe_ep_size,
            config.attention_dp_size,
            false,
        ),
    ];

    let generation_ops = vec![
        op_embedding("generation_embedding", mtp_scale_factor, vocab, h),
        op_elementwise(
            "generation_add_norm_1",
            layers * mtp_scale_factor,
            2 * h,
            2 * h,
        ),
        op_mla_module(
            "generation_mla_module",
            layers * mtp_scale_factor,
            false,
            mla_heads,
            config.kvcache_quant_mode,
            config.fmha_quant_mode,
            config.gemm_quant_mode,
            mla_weight_bytes,
        ),
        op_elementwise(
            "generation_add_norm_2",
            layers * mtp_scale_factor,
            2 * h,
            2 * h,
        ),
        op_overlap("generation_moe_overlap", gen_routed_ops, gen_shared_ops),
        op_gemm(
            "generation_logits_gemm",
            mtp_scale_factor,
            vocab / tp,
            h,
            GemmQuantMode::bfloat16,
        ),
        op_p2p(
            "generation_p2p",
            (pp - 1) as f64 * mtp_scale_factor,
            h,
            config.pp_size,
        ),
    ];

    Ok(DenseModel {
        model_path: model_path.to_string(),
        model_family: "MOE".to_string(),
        architecture: info.architecture.clone(),
        config,
        num_layers: layers,
        num_heads: info.n as i64,
        num_kv_heads: info.n_kv as i64,
        head_size: info.d as i64,
        hidden_size: h,
        inter_size: info.inter_size as i64,
        vocab_size: vocab,
        kvcache_elements_per_token_override: Some(layers * (512.0 + 64.0)),
        context_ops,
        generation_ops,
    })
}

fn deepseek_mla_weight_bytes(h: i64, tp: i64, quant_mode: GemmQuantMode) -> f64 {
    let memory = quant_mode.mapping().memory;
    let downscale = 2112.0 * h as f64;
    let q_b = (24576 / tp) as f64 * 1536.0;
    let kv_b = (32768 / tp) as f64 * 512.0;
    let proj = h as f64 * (128 * 128 / tp) as f64;
    (downscale + q_b + kv_b + proj) * memory
}

fn build_moe_model(model_path: &str, info: &ModelInfo, config: ModelConfig) -> Result<DenseModel> {
    let family = architecture_to_family(&info.architecture);
    if family != "LLAMA" && family != "GPT" {
        bail!(
            "native MoE estimator only supports GPT/LLAMA-style MoE models, got {}",
            info.architecture
        );
    }
    if config.tp_size * config.attention_dp_size != config.moe_tp_size * config.moe_ep_size {
        bail!(
            "tp_size ({}) * attention_dp_size ({}) must equal moe_tp_size ({}) * moe_ep_size ({})",
            config.tp_size,
            config.attention_dp_size,
            config.moe_tp_size,
            config.moe_ep_size
        );
    }
    if info.num_experts < config.moe_ep_size as u64 {
        bail!("moe_ep_size cannot exceed number of experts");
    }

    let h = info.hidden_size as i64;
    let use_qk_norm = matches!(
        info.architecture.as_str(),
        "Qwen3ForCausalLM" | "Qwen3MoeForCausalLM"
    );
    let tp = config.tp_size as i64;
    let pp = config.pp_size as i64;
    let num_heads = info.n as i64;
    let head_size = info.d as i64;
    let n_kv_total = if info.n_kv == 0 { info.n } else { info.n_kv } as i64;
    let num_kv_heads_per_gpu = (n_kv_total + tp - 1) / tp;
    let layers = info.layers as f64;
    let vocab = info.vocab as i64;
    let inter = info.inter_size as i64;
    let moe_inter = if info.moe_inter_size > 0 {
        info.moe_inter_size as i64
    } else {
        inter
    };
    validate_moe_block_quant(config, moe_inter)?;
    let topk = info.topk as i64;
    let num_experts = info.num_experts as i64;
    let workload_distribution = "power_law_1.2";
    let mtp_scale_factor = 1.0;

    let mut context_ops = vec![
        op_embedding("context_embedding", 1.0, vocab / tp, h),
        op_elementwise("context_add_norm_1", layers, 2 * h, 2 * h),
        op_gemm(
            "context_qkv_gemm",
            layers,
            num_heads * head_size / tp + head_size * num_kv_heads_per_gpu * 2,
            h,
            config.gemm_quant_mode,
        ),
        op_context_attention(
            "context_attention",
            layers,
            num_heads / tp,
            num_kv_heads_per_gpu,
            config.kvcache_quant_mode,
            config.fmha_quant_mode,
            head_size,
            use_qk_norm,
        ),
        op_gemm(
            "context_proj_gemm",
            layers,
            h,
            num_heads * head_size / tp,
            config.gemm_quant_mode,
        ),
        op_elementwise("context_add_norm_2", layers, 2 * h, 2 * h),
        op_gemm(
            "context_router_gemm",
            layers,
            num_experts,
            h,
            GemmQuantMode::bfloat16,
        ),
        op_moe_dispatch(
            "context_moe_pre_dispatch",
            layers,
            h,
            config.moe_tp_size,
            config.moe_ep_size,
            config.attention_dp_size,
            true,
        ),
        op_moe(
            "context_moe",
            layers,
            h,
            moe_inter,
            topk,
            num_experts,
            config.moe_tp_size,
            config.moe_ep_size,
            config.moe_quant_mode,
            workload_distribution,
            config.attention_dp_size,
            true,
        ),
        op_moe_dispatch(
            "context_moe_post_dispatch",
            layers,
            h,
            config.moe_tp_size,
            config.moe_ep_size,
            config.attention_dp_size,
            false,
        ),
        op_custom_ar("context_embedding_ar", 1.0, h, config.tp_size),
        op_custom_ar("context_ar_1", layers, h, config.tp_size),
        op_custom_ar("context_ar_2", layers, h, config.tp_size),
        op_p2p("context_p2p", (pp - 1) as f64, h, config.pp_size),
    ];

    let mut generation_ops = vec![
        op_embedding("generation_embedding", mtp_scale_factor, vocab / tp, h),
        op_elementwise(
            "generation_add_norm_1",
            layers * mtp_scale_factor,
            2 * h,
            2 * h,
        ),
        op_gemm(
            "generation_qkv_gemm",
            layers * mtp_scale_factor,
            num_heads * head_size / tp + head_size * num_kv_heads_per_gpu * 2,
            h,
            config.gemm_quant_mode,
        ),
        op_generation_attention(
            "generation_attention",
            layers * mtp_scale_factor,
            num_heads / tp,
            num_kv_heads_per_gpu,
            config.kvcache_quant_mode,
            head_size,
        ),
        op_gemm(
            "generation_proj_gemm",
            layers * mtp_scale_factor,
            h,
            num_heads * head_size / tp,
            config.gemm_quant_mode,
        ),
        op_elementwise(
            "generation_add_norm_2",
            layers * mtp_scale_factor,
            2 * h,
            2 * h,
        ),
        op_gemm(
            "generation_router_gemm",
            layers * mtp_scale_factor,
            num_experts,
            h,
            GemmQuantMode::bfloat16,
        ),
        op_moe_dispatch(
            "generation_moe_pre_dispatch",
            layers * mtp_scale_factor,
            h,
            config.moe_tp_size,
            config.moe_ep_size,
            config.attention_dp_size,
            true,
        ),
        op_moe(
            "generation_moe",
            layers * mtp_scale_factor,
            h,
            moe_inter,
            topk,
            num_experts,
            config.moe_tp_size,
            config.moe_ep_size,
            config.moe_quant_mode,
            workload_distribution,
            config.attention_dp_size,
            true,
        ),
        op_moe_dispatch(
            "generation_moe_post_dispatch",
            layers * mtp_scale_factor,
            h,
            config.moe_tp_size,
            config.moe_ep_size,
            config.attention_dp_size,
            false,
        ),
        op_gemm(
            "generation_logits_gemm",
            mtp_scale_factor,
            vocab / tp,
            h,
            GemmQuantMode::bfloat16,
        ),
        op_custom_ar(
            "generation_embedding_ar",
            mtp_scale_factor,
            h,
            config.tp_size,
        ),
        op_custom_ar(
            "generation_ar_1",
            layers * mtp_scale_factor,
            h,
            config.tp_size,
        ),
        op_custom_ar(
            "generation_ar_2",
            layers * mtp_scale_factor,
            h,
            config.tp_size,
        ),
        op_p2p(
            "generation_p2p",
            (pp - 1) as f64 * mtp_scale_factor,
            h,
            config.pp_size,
        ),
    ];

    if info.architecture == "GptOssForCausalLM" {
        for op in context_ops
            .iter_mut()
            .chain(generation_ops.iter_mut())
            .filter(|op| op.name.ends_with("_attention"))
        {
            op.scale_factor *= 0.5;
        }
    }

    Ok(DenseModel {
        model_path: model_path.to_string(),
        model_family: "MOE".to_string(),
        architecture: info.architecture.clone(),
        config,
        num_layers: layers,
        num_heads,
        num_kv_heads: n_kv_total,
        head_size,
        hidden_size: h,
        inter_size: inter,
        vocab_size: vocab,
        kvcache_elements_per_token_override: None,
        context_ops,
        generation_ops,
    })
}

fn validate_moe_block_quant(config: ModelConfig, moe_inter: i64) -> Result<()> {
    if !matches!(
        config.moe_quant_mode,
        MoeQuantMode::fp8_block | MoeQuantMode::nvfp4
    ) {
        return Ok(());
    }
    const WEIGHT_BLOCK_SIZE: i64 = 128;
    let moe_size_per_gpu = moe_inter / config.moe_tp_size as i64;
    if moe_size_per_gpu % WEIGHT_BLOCK_SIZE != 0 {
        bail!(
            "Invalid quantized MoE configuration: (moe_intermediate_size={} / moe_tp_size={}) % weight_block_size={} != 0",
            moe_inter,
            config.moe_tp_size,
            WEIGHT_BLOCK_SIZE
        );
    }
    Ok(())
}

fn get_memory_usage(
    model: &DenseModel,
    db: &PerfDatabase,
    batch_size: i64,
    beam_width: i64,
    isl: i64,
    osl: i64,
    mut num_tokens: i64,
    prefix: i64,
    max_seq_len: Option<i64>,
) -> MemoryUsage {
    let mut weights = model.context_ops.iter().map(Op::weights).sum::<f64>();
    weights /= model.config.pp_size as f64;
    let h = model.num_heads as f64 * model.head_size as f64;
    if num_tokens == 0 {
        num_tokens = (isl - prefix) * batch_size;
    }
    let tp = model.config.tp_size.min(8);
    let c = match model.model_family.as_str() {
        "GPT" => match tp {
            1 => 10.0,
            2 => 6.0,
            4 => 5.0,
            _ => 5.0,
        },
        "LLAMA" => match tp {
            1 => 11.0,
            2 => 6.5,
            4 => 5.0,
            _ => 5.0,
        },
        _ => 10.0,
    };
    let activations = (2.0 * num_tokens as f64 * h * c).max(70.0 * 1024.0 * 1024.0);
    let seq_tokens = max_seq_len.unwrap_or(isl + beam_width * osl);
    let kvcache = batch_size as f64 * model.kvcache_bytes_per_sequence(seq_tokens);
    let misc = db.system_spec.misc.as_ref();
    let nccl = misc
        .and_then(|m| m.nccl_mem.as_ref())
        .and_then(|m| m.get(&model.config.tp_size.min(8)))
        .copied()
        .unwrap_or(0.0);
    let others = misc.and_then(|m| m.other_mem).unwrap_or(0.0);
    let gib = (1_u64 << 30) as f64;
    MemoryUsage {
        total: (weights + activations + kvcache + nccl + others) / gib,
        weights: weights / gib,
        activations: activations / gib,
        kvcache: kvcache / gib,
        nccl: nccl / gib,
        others: others / gib,
    }
}

fn kv_cache_oom(
    req: &NativeDefaultRequest,
    row: &AggResult,
    db: &PerfDatabase,
    model: &DenseModel,
    b: i64,
) -> bool {
    let max_seq_len = req.max_seq_len.unwrap_or(req.isl + req.osl);
    let memory = get_memory_usage(
        model,
        db,
        b,
        1,
        req.isl,
        req.osl,
        TRTLLM_DEFAULT_MAX_NUM_TOKENS,
        req.prefix,
        Some(max_seq_len),
    );
    let capacity_gib = db.system_spec.gpu.mem_capacity / (1_u64 << 30) as f64;
    if row.memory >= capacity_gib {
        return false;
    }
    let fraction = req
        .free_gpu_memory_fraction
        .unwrap_or(TRTLLM_DEFAULT_FREE_GPU_MEMORY_FRACTION);
    let non_kv = memory.total - memory.kvcache;
    let budget = (capacity_gib - non_kv)
        * fraction
        * (1.0 - KV_CACHE_MEMORY_RESERVED_FRACTION)
        * (1.0 - KV_CACHE_MEMORY_TOLERANCE);
    memory.kvcache > budget
}

impl DenseModel {
    fn kvcache_elements_per_token(&self) -> f64 {
        if let Some(value) = self.kvcache_elements_per_token_override {
            return value;
        }
        let n_kv_per_gpu =
            (self.num_kv_heads + self.config.tp_size as i64 - 1) / self.config.tp_size as i64;
        (n_kv_per_gpu * self.head_size) as f64 * self.num_layers * 2.0
    }

    fn kvcache_bytes_per_sequence(&self, seq_len: i64) -> f64 {
        seq_len.max(0) as f64
            * self.config.kvcache_quant_mode.mapping().memory
            * self.kvcache_elements_per_token()
    }
}

fn infer_quant_modes(info: &ModelInfo, backend: BackendName) -> Result<ModelConfig> {
    let raw = &info.raw_config;
    let mut gemm = None;
    let mut moe = None;
    let mut kv = None;
    let mut fmha = None;
    let quant_cfg = raw.get("quantization_config").and_then(Value::as_object);
    let mut quant_algo = raw
        .get("quant_algo")
        .and_then(Value::as_str)
        .map(ToString::to_string)
        .or_else(|| {
            quant_cfg
                .and_then(|cfg| {
                    cfg.get("quant_algo")
                        .or_else(|| cfg.get("quant_method"))
                        .or_else(|| cfg.get("quantization_method"))
                })
                .and_then(Value::as_str)
                .map(|value| value.to_ascii_lowercase())
        });
    if quant_algo.as_deref() == Some("fp8")
        && quant_cfg
            .and_then(|cfg| cfg.get("weight_block_size"))
            .is_some()
    {
        quant_algo = Some("fp8_block".to_string());
    }
    let kv_algo = raw
        .get("kv_cache_quant_algo")
        .and_then(Value::as_str)
        .map(ToString::to_string)
        .or_else(|| {
            quant_cfg
                .and_then(|cfg| {
                    cfg.get("kv_cache_quant_algo")
                        .or_else(|| cfg.get("kv_cache_quant_method"))
                        .or_else(|| cfg.get("kv_cache_dtype"))
                })
                .and_then(Value::as_str)
                .map(|value| value.to_ascii_lowercase())
        });

    match quant_algo.as_deref() {
        Some("fp8") => {
            gemm = Some(
                if raw.get("quant_dynamic").and_then(Value::as_bool) == Some(false) {
                    GemmQuantMode::fp8_static
                } else {
                    GemmQuantMode::fp8
                },
            );
            moe = Some(MoeQuantMode::fp8);
        }
        Some("fp8_block") => {
            gemm = Some(GemmQuantMode::fp8_block);
            moe = Some(MoeQuantMode::fp8_block);
        }
        Some("nvfp4") => {
            gemm = Some(GemmQuantMode::nvfp4);
            moe = Some(MoeQuantMode::nvfp4);
        }
        Some("mxfp4") => {
            gemm = Some(GemmQuantMode::bfloat16);
            moe = Some(MoeQuantMode::w4a16_mxfp4);
        }
        Some(other) => bail!("native quant default does not support quant_algo={other}"),
        None => {}
    }
    match kv_algo.as_deref() {
        Some("fp8") => kv = Some(KvCacheQuantMode::fp8),
        Some("bfloat16") => kv = Some(KvCacheQuantMode::bfloat16),
        Some(other) => bail!("native quant default does not support kv_cache_quant_algo={other}"),
        None => {}
    }
    if matches!(quant_algo.as_deref(), Some("fp8" | "fp8_block" | "nvfp4"))
        || kv == Some(KvCacheQuantMode::fp8)
    {
        fmha = Some(FmhaQuantMode::fp8);
        if kv.is_none() {
            kv = Some(KvCacheQuantMode::fp8);
        }
    }
    if backend == BackendName::vllm && fmha == Some(FmhaQuantMode::fp8) {
        fmha = Some(FmhaQuantMode::bfloat16);
    }
    if matches!(
        info.architecture.as_str(),
        "DeepseekV3ForCausalLM" | "KimiK25ForConditionalGeneration" | "DeepseekV4ForCausalLM"
    ) && fmha == Some(FmhaQuantMode::fp8)
    {
        fmha = Some(FmhaQuantMode::bfloat16);
    }

    Ok(ModelConfig {
        tp_size: 1,
        pp_size: 1,
        attention_dp_size: 1,
        moe_tp_size: 1,
        moe_ep_size: 1,
        gemm_quant_mode: gemm.unwrap_or(GemmQuantMode::bfloat16),
        moe_quant_mode: moe.unwrap_or(MoeQuantMode::bfloat16),
        kvcache_quant_mode: kv.unwrap_or(KvCacheQuantMode::bfloat16),
        fmha_quant_mode: fmha.unwrap_or(FmhaQuantMode::bfloat16),
        comm_quant_mode: CommQuantMode::half,
        nextn: default_nextn_for_architecture(&info.architecture),
    })
}

fn default_nextn_for_architecture(architecture: &str) -> u32 {
    match architecture {
        "DeepseekV3ForCausalLM"
        | "DeepseekV32ForCausalLM"
        | "DeepseekV4ForCausalLM"
        | "KimiK25ForConditionalGeneration"
        | "Qwen3NextForCausalLM" => 1,
        _ => 0,
    }
}

fn mtp_scale_factor(nextn: u32, num_layers: f64) -> f64 {
    const DEFAULT_NEXTN_ACCEPT_RATES: [f64; 5] = [0.85, 0.3, 0.0, 0.0, 0.0];
    let expectation = calc_mtp_expectation(nextn, &DEFAULT_NEXTN_ACCEPT_RATES);
    if expectation == 0.0 {
        1.0
    } else {
        (nextn as f64 + num_layers) / num_layers / (1.0 + expectation)
    }
}

fn calc_mtp_expectation(nextn: u32, accept_rates: &[f64]) -> f64 {
    if nextn == 0 {
        return 0.0;
    }
    let capped = nextn.min(accept_rates.len() as u32);
    let mut expectation = 0.0;
    let mut prob = 1.0;
    for i in 0..capped {
        prob *= accept_rates[i as usize];
        expectation += prob;
    }
    expectation
}

fn is_moe_model(info: &ModelInfo) -> bool {
    info.num_experts > 0
}

fn architecture_to_family(architecture: &str) -> &'static str {
    if architecture.contains("GPT") || architecture.contains("Gpt") {
        "GPT"
    } else {
        "LLAMA"
    }
}

fn ctx_tokens_list_for_agg_sweep(
    isl: i64,
    mut ctx_stride: i64,
    enable_chunked_prefill: bool,
) -> Vec<i64> {
    let max_normal_ctx_tokens = 8192;
    let max_ctx_tokens = max_normal_ctx_tokens.max(isl * 2);
    ctx_stride = ctx_stride.max(max_normal_ctx_tokens / 16);
    let mut ctx_stride_large = 1024.max(ctx_stride).max(max_ctx_tokens / 8);
    if !enable_chunked_prefill {
        ctx_stride = ctx_stride.max(isl);
        ctx_stride_large = ((ctx_stride_large + isl - 1) / isl) * isl;
    }
    let mut out = Vec::new();
    let mut ctx_tokens = 0;
    loop {
        if ctx_tokens < max_normal_ctx_tokens {
            ctx_tokens += ctx_stride;
        } else {
            ctx_tokens += ctx_stride_large;
        }
        if ctx_tokens > max_ctx_tokens {
            break;
        }
        out.push(ctx_tokens);
    }
    for i in 1..=2 {
        let value = isl * i;
        if !out.contains(&value) {
            out.push(value);
        }
    }
    out.sort_unstable();
    out
}

fn op_embedding(name: &str, scale_factor: f64, row_size: i64, column_size: i64) -> Op {
    Op {
        name: name.to_string(),
        scale_factor,
        kind: OpKind::Embedding {
            row_size,
            column_size,
        },
    }
}

fn op_elementwise(name: &str, scale_factor: f64, dim_in: i64, dim_out: i64) -> Op {
    Op {
        name: name.to_string(),
        scale_factor,
        kind: OpKind::ElementWise {
            dim_in,
            dim_out,
            scale_num_tokens: 1,
        },
    }
}

fn op_gemm(name: &str, scale_factor: f64, n: i64, k: i64, quant_mode: GemmQuantMode) -> Op {
    Op {
        name: name.to_string(),
        scale_factor,
        kind: OpKind::Gemm { n, k, quant_mode },
    }
}

fn op_context_attention(
    name: &str,
    scale_factor: f64,
    n: i64,
    n_kv: i64,
    kvcache_quant_mode: KvCacheQuantMode,
    fmha_quant_mode: FmhaQuantMode,
    head_size: i64,
    use_qk_norm: bool,
) -> Op {
    Op {
        name: name.to_string(),
        scale_factor,
        kind: OpKind::ContextAttention {
            n,
            n_kv,
            kvcache_quant_mode,
            fmha_quant_mode,
            window_size: 0,
            head_size,
            use_qk_norm,
        },
    }
}

fn op_generation_attention(
    name: &str,
    scale_factor: f64,
    n: i64,
    n_kv: i64,
    kvcache_quant_mode: KvCacheQuantMode,
    head_size: i64,
) -> Op {
    Op {
        name: name.to_string(),
        scale_factor,
        kind: OpKind::GenerationAttention {
            n,
            n_kv,
            kvcache_quant_mode,
            window_size: 0,
            head_size,
        },
    }
}

fn op_mla_module(
    name: &str,
    scale_factor: f64,
    is_context: bool,
    num_heads: i64,
    kvcache_quant_mode: KvCacheQuantMode,
    fmha_quant_mode: FmhaQuantMode,
    gemm_quant_mode: GemmQuantMode,
    weight_bytes: f64,
) -> Op {
    Op {
        name: name.to_string(),
        scale_factor,
        kind: OpKind::MlaModule {
            is_context,
            num_heads,
            kvcache_quant_mode,
            fmha_quant_mode,
            gemm_quant_mode,
            weight_bytes,
        },
    }
}

#[allow(clippy::too_many_arguments)]
fn op_moe(
    name: &str,
    scale_factor: f64,
    hidden_size: i64,
    inter_size: i64,
    topk: i64,
    num_experts: i64,
    moe_tp_size: u32,
    moe_ep_size: u32,
    quant_mode: MoeQuantMode,
    workload_distribution: &str,
    attention_dp_size: u32,
    is_gated: bool,
) -> Op {
    Op {
        name: name.to_string(),
        scale_factor,
        kind: OpKind::Moe {
            hidden_size,
            inter_size,
            topk,
            num_experts,
            moe_tp_size,
            moe_ep_size,
            quant_mode,
            workload_distribution: workload_distribution.to_string(),
            attention_dp_size,
            is_gated,
        },
    }
}

fn op_moe_dispatch(
    name: &str,
    scale_factor: f64,
    hidden_size: i64,
    moe_tp_size: u32,
    moe_ep_size: u32,
    attention_dp_size: u32,
    pre_dispatch: bool,
) -> Op {
    Op {
        name: name.to_string(),
        scale_factor,
        kind: OpKind::MoeDispatch {
            hidden_size,
            moe_tp_size,
            moe_ep_size,
            attention_dp_size,
            pre_dispatch,
        },
    }
}

fn op_custom_ar(name: &str, scale_factor: f64, h: i64, tp_size: u32) -> Op {
    Op {
        name: name.to_string(),
        scale_factor,
        kind: OpKind::CustomAllReduce { h, tp_size },
    }
}

fn op_p2p(name: &str, scale_factor: f64, h: i64, pp_size: u32) -> Op {
    Op {
        name: name.to_string(),
        scale_factor,
        kind: OpKind::P2P { h, pp_size },
    }
}

fn op_overlap(name: &str, group_a: Vec<Op>, group_b: Vec<Op>) -> Op {
    Op {
        name: name.to_string(),
        scale_factor: 1.0,
        kind: OpKind::Overlap { group_a, group_b },
    }
}

fn ceil_div(a: i64, b: i64) -> f64 {
    (a as f64 / b as f64).ceil()
}

fn round3(value: f64) -> f64 {
    (value * 1000.0).round() / 1000.0
}
