use crate::types::BackendName;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParallelConfig {
    pub tp: u32,
    pub pp: u32,
    pub dp: u32,
    pub moe_tp: u32,
    pub moe_ep: u32,
}

#[derive(Debug, Clone)]
pub struct ParallelSearch {
    pub num_gpu_list: Vec<u32>,
    pub tp_list: Vec<u32>,
    pub pp_list: Vec<u32>,
    pub dp_list: Vec<u32>,
    pub moe_tp_list: Vec<u32>,
    pub moe_ep_list: Vec<u32>,
    pub is_moe: bool,
    pub backend: BackendName,
    pub enable_wideep: bool,
    pub moe_backend: Option<String>,
    pub real_silicon_sweep: bool,
    pub min_num_gpus: Option<u32>,
    pub max_num_gpus: Option<u32>,
    pub allow_moe_pure_tp: bool,
}

impl Default for ParallelSearch {
    fn default() -> Self {
        Self {
            num_gpu_list: Vec::new(),
            tp_list: Vec::new(),
            pp_list: Vec::new(),
            dp_list: vec![1],
            moe_tp_list: vec![1],
            moe_ep_list: vec![1],
            is_moe: false,
            backend: BackendName::trtllm,
            enable_wideep: false,
            moe_backend: None,
            real_silicon_sweep: false,
            min_num_gpus: None,
            max_num_gpus: None,
            allow_moe_pure_tp: true,
        }
    }
}

pub fn enumerate_parallel_config(search: &ParallelSearch) -> Vec<ParallelConfig> {
    let pp_list = if search.real_silicon_sweep {
        vec![1]
    } else {
        search.pp_list.clone()
    };

    let mut out = Vec::new();
    for &tp in &search.tp_list {
        for &pp in &pp_list {
            if search.is_moe {
                for &dp in &search.dp_list {
                    for &moe_tp in &search.moe_tp_list {
                        for &moe_ep in &search.moe_ep_list {
                            if !search.num_gpu_list.contains(&(dp * tp * pp))
                                || dp * tp != moe_tp * moe_ep
                            {
                                continue;
                            }
                            if search.backend == BackendName::trtllm && dp > 1 && tp > 1 {
                                continue;
                            }
                            if search.backend == BackendName::sglang
                                && (search.enable_wideep
                                    || search.moe_backend.as_deref() == Some("deepep_moe"))
                                && moe_tp > 1
                            {
                                continue;
                            }
                            if search.backend == BackendName::vllm && moe_tp > 1 && moe_ep > 1 {
                                continue;
                            }
                            out.push(ParallelConfig {
                                tp,
                                pp,
                                dp,
                                moe_tp,
                                moe_ep,
                            });
                        }
                    }
                }
            } else if search.num_gpu_list.contains(&(tp * pp)) {
                out.push(ParallelConfig {
                    tp,
                    pp,
                    dp: 1,
                    moe_tp: 1,
                    moe_ep: 1,
                });
            }
        }
    }

    if search.real_silicon_sweep {
        out = filter_real_silicon_configs(
            &out,
            search.is_moe,
            search.min_num_gpus,
            search.max_num_gpus,
            search.allow_moe_pure_tp,
        );
    }
    out
}

pub fn filter_real_silicon_configs(
    configs: &[ParallelConfig],
    is_moe: bool,
    min_num_gpus: Option<u32>,
    max_num_gpus: Option<u32>,
    allow_moe_pure_tp: bool,
) -> Vec<ParallelConfig> {
    configs
        .iter()
        .copied()
        .filter(|cfg| {
            let total_gpus = cfg.tp * cfg.pp * cfg.dp;
            if min_num_gpus.map_or(false, |min| total_gpus < min) {
                return false;
            }
            if max_num_gpus.map_or(false, |max| total_gpus > max) {
                return false;
            }
            if !is_moe {
                return true;
            }
            let is_pure_tep = cfg.tp > 1 && cfg.dp == 1 && cfg.moe_tp == 1 && cfg.moe_ep > 1;
            let is_pure_dep = cfg.tp == 1 && cfg.dp > 1 && cfg.moe_tp == 1 && cfg.moe_ep > 1;
            let mut is_pure_tp = cfg.tp > 1 && cfg.dp == 1 && cfg.moe_tp > 1 && cfg.moe_ep == 1;
            if !allow_moe_pure_tp {
                is_pure_tp = false;
            }
            is_pure_tep || is_pure_dep || is_pure_tp
        })
        .collect()
}
