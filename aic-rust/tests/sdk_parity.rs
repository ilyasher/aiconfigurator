#![cfg(feature = "python-parity")]

use aic_rust::model::load_model_info;
use aic_rust::paths::default_systems_root;
use aic_rust::perf_database::PerfDatabase;
use aic_rust::search::{enumerate_parallel_config, ParallelConfig, ParallelSearch};
use aic_rust::support::check_support;
use aic_rust::system::latest_database_version;
use aic_rust::types::{
    BackendName, CommQuantMode, DatabaseMode, FmhaQuantMode, GemmQuantMode, KvCacheQuantMode,
    MoeQuantMode,
};
use serde_json::Value;
use std::path::PathBuf;
use std::process::Command;

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn python_repo_root() -> PathBuf {
    std::env::var_os("AIC_PYTHON_REPO")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            manifest_dir()
                .parent()
                .map(PathBuf::from)
                .expect("aic-rust should be inside the aiconfigurator repo")
        })
}

fn python_bin() -> PathBuf {
    if let Some(path) = std::env::var_os("AIC_PYTHON_BIN") {
        return PathBuf::from(path);
    }
    let manifest_venv = manifest_dir().join(".venv/bin/python");
    if manifest_venv.exists() {
        return manifest_venv;
    }
    let repo_venv = python_repo_root().join(".venv/bin/python");
    if repo_venv.exists() {
        return repo_venv;
    }
    PathBuf::from("python3")
}

fn python_json(code: &str) -> Value {
    let repo_root = python_repo_root();
    let pythonpath = match std::env::var_os("PYTHONPATH") {
        Some(existing) => {
            let mut paths = vec![repo_root.join("src")];
            paths.extend(std::env::split_paths(&existing));
            std::env::join_paths(paths).expect("failed to build PYTHONPATH")
        }
        None => repo_root.join("src").into_os_string(),
    };
    let output = Command::new(python_bin())
        .arg("-c")
        .arg(code)
        .current_dir(repo_root)
        .env("PYTHONPATH", pythonpath)
        .output()
        .expect("failed to run python oracle");
    assert!(
        output.status.success(),
        "python oracle failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("python oracle emitted invalid json")
}

#[test]
fn model_info_matches_python_for_qwen32b_fp8() {
    let rust = load_model_info("Qwen/Qwen3-32B-FP8").unwrap();
    let py = python_json(
        r#"
import json
from aiconfigurator.sdk.utils import get_model_config_from_model_path
info = get_model_config_from_model_path("Qwen/Qwen3-32B-FP8")
keys = ["architecture", "layers", "n", "n_kv", "d", "hidden_size", "inter_size", "vocab", "context", "topk", "num_experts", "moe_inter_size"]
print(json.dumps({k: info[k] for k in keys}))
"#,
    );

    assert_eq!(rust.architecture, py["architecture"]);
    assert_eq!(rust.layers, py["layers"].as_u64().unwrap());
    assert_eq!(rust.n, py["n"].as_u64().unwrap());
    assert_eq!(rust.n_kv, py["n_kv"].as_u64().unwrap());
    assert_eq!(rust.d, py["d"].as_u64().unwrap());
    assert_eq!(rust.hidden_size, py["hidden_size"].as_u64().unwrap());
    assert_eq!(rust.inter_size, py["inter_size"].as_u64().unwrap());
    assert_eq!(rust.vocab, py["vocab"].as_u64().unwrap());
    assert_eq!(rust.context, py["context"].as_u64().unwrap());
    assert_eq!(rust.topk, py["topk"].as_u64().unwrap());
    assert_eq!(rust.num_experts, py["num_experts"].as_u64().unwrap());
    assert_eq!(rust.moe_inter_size, py["moe_inter_size"].as_u64().unwrap());
}

#[test]
fn support_check_matches_python_exact_model() {
    let rust = check_support(
        "Qwen/Qwen3-32B-FP8",
        "h200_sxm",
        Some(BackendName::trtllm),
        None,
    )
    .unwrap();
    let py = python_json(
        r#"
import json
from aiconfigurator.sdk.common import check_support
r = check_support("Qwen/Qwen3-32B-FP8", "h200_sxm", "trtllm", None)
print(json.dumps({
  "agg_supported": r.agg_supported,
  "disagg_supported": r.disagg_supported,
  "exact_match": r.exact_match,
}))
"#,
    );
    assert_eq!(rust.agg_supported, py["agg_supported"].as_bool().unwrap());
    assert_eq!(
        rust.disagg_supported,
        py["disagg_supported"].as_bool().unwrap()
    );
    assert_eq!(rust.exact_match, py["exact_match"].as_bool().unwrap());
}

#[test]
fn latest_database_version_matches_python() {
    let rust = latest_database_version(default_systems_root(), "h200_sxm", BackendName::trtllm)
        .unwrap()
        .unwrap();
    let py = python_json(
        r#"
import json
from aiconfigurator.sdk.perf_database import get_latest_database_version
print(json.dumps(get_latest_database_version("h200_sxm", "trtllm")))
"#,
    );
    assert_eq!(rust, py.as_str().unwrap());
}

#[test]
fn gemm_exact_query_matches_python() {
    let db = PerfDatabase::load(
        default_systems_root(),
        "h200_sxm",
        BackendName::trtllm,
        "1.2.0rc5",
    )
    .unwrap();
    let rust = db
        .query_gemm(
            8192,
            65536,
            10240,
            GemmQuantMode::int4_wo,
            DatabaseMode::SILICON,
        )
        .unwrap();
    let py = python_json(
        r#"
import json
from aiconfigurator.sdk.perf_database import get_database
from aiconfigurator.sdk import common
db = get_database("h200_sxm", "trtllm", "1.2.0rc5")
r = db.query_gemm(8192, 65536, 10240, common.GEMMQuantMode.int4_wo)
print(json.dumps({"latency": float(r), "energy": r.energy}))
"#,
    );
    assert!((rust.latency - py["latency"].as_f64().unwrap()).abs() < 1e-9);
    assert!((rust.energy - py["energy"].as_f64().unwrap()).abs() < 1e-9);
}

#[test]
fn gemm_cubic_query_matches_python() {
    let db = PerfDatabase::load(
        default_systems_root(),
        "h200_sxm",
        BackendName::trtllm,
        "1.3.0rc10",
    )
    .unwrap();
    let rust = db
        .query_gemm(
            512,
            151936,
            1024,
            GemmQuantMode::bfloat16,
            DatabaseMode::SILICON,
        )
        .unwrap();
    let py = python_json(
        r#"
import json
from aiconfigurator.sdk.perf_database import get_database
from aiconfigurator.sdk import common
db = get_database("h200_sxm", "trtllm", "1.3.0rc10")
r = db.query_gemm(512, 151936, 1024, common.GEMMQuantMode.bfloat16)
print(json.dumps({"latency": float(r), "energy": float(r.energy)}))
"#,
    );
    assert!((rust.latency - py["latency"].as_f64().unwrap()).abs() < 1e-12);
    assert!((rust.energy - py["energy"].as_f64().unwrap()).abs() < 1e-12);
}

#[test]
fn gemm_sol_query_matches_python() {
    let db = PerfDatabase::load(
        default_systems_root(),
        "h200_sxm",
        BackendName::trtllm,
        "1.2.0rc5",
    )
    .unwrap();
    let rust = db
        .query_gemm(4096, 5120, 5120, GemmQuantMode::bfloat16, DatabaseMode::SOL)
        .unwrap();
    let py = python_json(
        r#"
import json
from aiconfigurator.sdk.perf_database import get_database
from aiconfigurator.sdk import common
db = get_database("h200_sxm", "trtllm", "1.2.0rc5")
r = db.query_gemm(4096, 5120, 5120, common.GEMMQuantMode.bfloat16, common.DatabaseMode.SOL)
print(json.dumps({"latency": float(r), "energy": r.energy}))
"#,
    );
    assert!((rust.latency - py["latency"].as_f64().unwrap()).abs() < 1e-12);
    assert!((rust.energy - py["energy"].as_f64().unwrap()).abs() < 1e-12);
}

#[test]
fn context_attention_exact_query_matches_python() {
    let db = PerfDatabase::load(
        default_systems_root(),
        "h200_sxm",
        BackendName::trtllm,
        "1.2.0rc5",
    )
    .unwrap();
    let rust = db
        .query_context_attention(
            8,
            16384,
            0,
            96,
            1,
            KvCacheQuantMode::bfloat16,
            FmhaQuantMode::bfloat16,
            DatabaseMode::SILICON,
            128,
            64,
        )
        .unwrap();
    let py = python_json(
        r#"
import json
from aiconfigurator.sdk.perf_database import get_database
from aiconfigurator.sdk import common
db = get_database("h200_sxm", "trtllm", "1.2.0rc5")
r = db.query_context_attention(
  b=8, s=16384, prefix=0, n=96, n_kv=1,
  kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
  fmha_quant_mode=common.FMHAQuantMode.bfloat16,
  window_size=128, head_size=64,
)
print(json.dumps({"latency": float(r), "energy": r.energy}))
"#,
    );
    assert!((rust.latency - py["latency"].as_f64().unwrap()).abs() < 1e-9);
    assert!((rust.energy - py["energy"].as_f64().unwrap()).abs() < 1e-9);
}

#[test]
fn context_attention_cubic_query_matches_python() {
    let db = PerfDatabase::load(
        default_systems_root(),
        "h200_sxm",
        BackendName::trtllm,
        "1.3.0rc10",
    )
    .unwrap();
    let rust = db
        .query_context_attention(
            17,
            145,
            0,
            16,
            8,
            KvCacheQuantMode::bfloat16,
            FmhaQuantMode::bfloat16,
            DatabaseMode::SILICON,
            0,
            128,
        )
        .unwrap();
    let py = python_json(
        r#"
import json
from aiconfigurator.sdk.perf_database import get_database
from aiconfigurator.sdk import common
db = get_database("h200_sxm", "trtllm", "1.3.0rc10")
r = db.query_context_attention(
  b=17, s=145, prefix=0, n=16, n_kv=8,
  kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
  fmha_quant_mode=common.FMHAQuantMode.bfloat16,
  window_size=0, head_size=128,
)
print(json.dumps({"latency": float(r), "energy": float(r.energy)}))
"#,
    );
    assert!((rust.latency - py["latency"].as_f64().unwrap()).abs() < 1e-12);
    assert!((rust.energy - py["energy"].as_f64().unwrap()).abs() < 1e-12);
}

#[test]
fn context_attention_extrapolated_query_matches_python() {
    let db = PerfDatabase::load(
        default_systems_root(),
        "h200_sxm",
        BackendName::trtllm,
        "1.3.0rc10",
    )
    .unwrap();
    let rust = db
        .query_context_attention(
            480,
            1024,
            0,
            16,
            8,
            KvCacheQuantMode::bfloat16,
            FmhaQuantMode::bfloat16,
            DatabaseMode::SILICON,
            0,
            128,
        )
        .unwrap();
    let py = python_json(
        r#"
import json
from aiconfigurator.sdk.perf_database import get_database
from aiconfigurator.sdk import common
db = get_database("h200_sxm", "trtllm", "1.3.0rc10")
r = db.query_context_attention(
  b=480, s=1024, prefix=0, n=16, n_kv=8,
  kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
  fmha_quant_mode=common.FMHAQuantMode.bfloat16,
  window_size=0, head_size=128,
)
print(json.dumps({"latency": float(r), "energy": float(r.energy)}))
"#,
    );
    assert!((rust.latency - py["latency"].as_f64().unwrap()).abs() < 1e-9);
    assert!((rust.energy - py["energy"].as_f64().unwrap()).abs() < 1e-9);
}

#[test]
fn generation_attention_query_matches_python() {
    let db = PerfDatabase::load(
        default_systems_root(),
        "h200_sxm",
        BackendName::trtllm,
        "1.3.0rc10",
    )
    .unwrap();
    let rust = db
        .query_generation_attention(
            512,
            145,
            16,
            8,
            KvCacheQuantMode::bfloat16,
            DatabaseMode::SILICON,
            0,
            128,
        )
        .unwrap();
    let py = python_json(
        r#"
import json
from aiconfigurator.sdk.perf_database import get_database
from aiconfigurator.sdk import common
db = get_database("h200_sxm", "trtllm", "1.3.0rc10")
r = db.query_generation_attention(
  b=512, s=145, n=16, n_kv=8,
  kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
  window_size=0, head_size=128,
)
print(json.dumps({"latency": float(r), "energy": r.energy}))
"#,
    );
    assert!((rust.latency - py["latency"].as_f64().unwrap()).abs() < 1e-12);
    assert!((rust.energy - py["energy"].as_f64().unwrap()).abs() < 1e-12);
}

#[test]
fn context_mla_query_matches_python() {
    let db = PerfDatabase::load(
        default_systems_root(),
        "h200_sxm",
        BackendName::trtllm,
        "1.3.0rc10",
    )
    .unwrap();
    let rust = db
        .query_context_mla(
            17,
            145,
            0,
            16,
            KvCacheQuantMode::bfloat16,
            FmhaQuantMode::bfloat16,
            DatabaseMode::SILICON,
        )
        .unwrap();
    let py = python_json(
        r#"
import json
from aiconfigurator.sdk.perf_database import get_database
from aiconfigurator.sdk import common
db = get_database("h200_sxm", "trtllm", "1.3.0rc10")
r = db.query_context_mla(
  b=17, s=145, prefix=0, num_heads=16,
  kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
  fmha_quant_mode=common.FMHAQuantMode.bfloat16,
)
print(json.dumps({"latency": float(r), "energy": float(r.energy)}))
"#,
    );
    assert!((rust.latency - py["latency"].as_f64().unwrap()).abs() < 1e-9);
    assert!((rust.energy - py["energy"].as_f64().unwrap()).abs() < 1e-9);
}

#[test]
fn generation_mla_query_matches_python() {
    let db = PerfDatabase::load(
        default_systems_root(),
        "h200_sxm",
        BackendName::trtllm,
        "1.3.0rc10",
    )
    .unwrap();
    let rust = db
        .query_generation_mla(
            17,
            145,
            16,
            KvCacheQuantMode::bfloat16,
            DatabaseMode::SILICON,
        )
        .unwrap();
    let py = python_json(
        r#"
import json
from aiconfigurator.sdk.perf_database import get_database
from aiconfigurator.sdk import common
db = get_database("h200_sxm", "trtllm", "1.3.0rc10")
r = db.query_generation_mla(
  b=17, s=145, num_heads=16,
  kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
)
print(json.dumps({"latency": float(r), "energy": float(r.energy)}))
"#,
    );
    assert!((rust.latency - py["latency"].as_f64().unwrap()).abs() < 1e-9);
    assert!((rust.energy - py["energy"].as_f64().unwrap()).abs() < 1e-9);
}

#[test]
fn context_mla_module_query_matches_python() {
    let db = PerfDatabase::load(
        default_systems_root(),
        "h200_sxm",
        BackendName::trtllm,
        "1.3.0rc10",
    )
    .unwrap();
    let rust = db
        .query_context_mla_module(
            17,
            145,
            0,
            16,
            KvCacheQuantMode::fp8,
            FmhaQuantMode::bfloat16,
            GemmQuantMode::fp8_block,
            DatabaseMode::SILICON,
        )
        .unwrap();
    let py = python_json(
        r#"
import json
from aiconfigurator.sdk.perf_database import get_database
from aiconfigurator.sdk import common
db = get_database("h200_sxm", "trtllm", "1.3.0rc10")
r = db.query_context_mla_module(
  b=17, s=145, prefix=0, num_heads=16,
  kvcache_quant_mode=common.KVCacheQuantMode.fp8,
  fmha_quant_mode=common.FMHAQuantMode.bfloat16,
  gemm_quant_mode=common.GEMMQuantMode.fp8_block,
)
print(json.dumps({"latency": float(r), "energy": float(r.energy)}))
"#,
    );
    assert!((rust.latency - py["latency"].as_f64().unwrap()).abs() < 1e-9);
    assert!((rust.energy - py["energy"].as_f64().unwrap()).abs() < 1e-9);
}

#[test]
fn generation_mla_module_query_matches_python() {
    let db = PerfDatabase::load(
        default_systems_root(),
        "h200_sxm",
        BackendName::trtllm,
        "1.3.0rc10",
    )
    .unwrap();
    let rust = db
        .query_generation_mla_module(
            17,
            145,
            16,
            KvCacheQuantMode::fp8,
            FmhaQuantMode::bfloat16,
            GemmQuantMode::fp8_block,
            DatabaseMode::SILICON,
        )
        .unwrap();
    let py = python_json(
        r#"
import json
from aiconfigurator.sdk.perf_database import get_database
from aiconfigurator.sdk import common
db = get_database("h200_sxm", "trtllm", "1.3.0rc10")
r = db.query_generation_mla_module(
  b=17, s=145, num_heads=16,
  kv_cache_dtype=common.KVCacheQuantMode.fp8,
  fmha_quant_mode=common.FMHAQuantMode.bfloat16,
  gemm_quant_mode=common.GEMMQuantMode.fp8_block,
)
print(json.dumps({"latency": float(r), "energy": float(r.energy)}))
"#,
    );
    assert!((rust.latency - py["latency"].as_f64().unwrap()).abs() < 1e-9);
    assert!((rust.energy - py["energy"].as_f64().unwrap()).abs() < 1e-9);
}

#[test]
fn custom_allreduce_query_matches_python() {
    let db = PerfDatabase::load(
        default_systems_root(),
        "h200_sxm",
        BackendName::trtllm,
        "1.3.0rc10",
    )
    .unwrap();
    let rust = db
        .query_custom_allreduce(CommQuantMode::half, 4, 8192, DatabaseMode::SILICON)
        .unwrap();
    let py = python_json(
        r#"
import json
from aiconfigurator.sdk.perf_database import get_database
from aiconfigurator.sdk import common
db = get_database("h200_sxm", "trtllm", "1.3.0rc10")
r = db.query_custom_allreduce(common.CommQuantMode.half, 4, 8192)
print(json.dumps({"latency": float(r), "energy": r.energy}))
"#,
    );
    assert!((rust.latency - py["latency"].as_f64().unwrap()).abs() < 1e-12);
    assert!((rust.energy - py["energy"].as_f64().unwrap()).abs() < 1e-12);
}

#[test]
fn moe_query_matches_python() {
    let db = PerfDatabase::load(
        default_systems_root(),
        "h200_sxm",
        BackendName::trtllm,
        "1.3.0rc10",
    )
    .unwrap();
    let rust = db
        .query_moe(
            64,
            2688,
            1856,
            6,
            128,
            1,
            64,
            MoeQuantMode::bfloat16,
            "balanced",
            true,
            None,
            DatabaseMode::SILICON,
            false,
            false,
        )
        .unwrap();
    let py = python_json(
        r#"
import json
from aiconfigurator.sdk.perf_database import get_database
from aiconfigurator.sdk import common
db = get_database("h200_sxm", "trtllm", "1.3.0rc10")
r = db.query_moe(
  num_tokens=64, hidden_size=2688, inter_size=1856, topk=6,
  num_experts=128, moe_tp_size=1, moe_ep_size=64,
  quant_mode=common.MoEQuantMode.bfloat16,
  workload_distribution="balanced",
  is_gated=False,
)
print(json.dumps({"latency": float(r), "energy": float(r.energy)}))
"#,
    );
    assert!((rust.latency - py["latency"].as_f64().unwrap()).abs() < 1e-12);
    assert!((rust.energy - py["energy"].as_f64().unwrap()).abs() < 1e-12);
}

#[test]
fn dense_parallel_enumeration_matches_python() {
    let rust = enumerate_parallel_config(&ParallelSearch {
        num_gpu_list: vec![1, 2, 4, 8],
        tp_list: vec![1, 2, 4, 8],
        pp_list: vec![1, 2, 4],
        backend: BackendName::trtllm,
        ..ParallelSearch::default()
    });
    let py = python_json(
        r#"
import json
from aiconfigurator.sdk.utils import enumerate_parallel_config
from aiconfigurator.sdk import common
print(json.dumps(enumerate_parallel_config(
  num_gpu_list=[1,2,4,8],
  tp_list=[1,2,4,8],
  pp_list=[1,2,4],
  is_moe=False,
  backend=common.BackendName.trtllm,
)))
"#,
    );
    assert_eq!(rust, py_parallel_configs(&py));
}

#[test]
fn moe_parallel_enumeration_matches_python_for_sglang_deepep() {
    let rust = enumerate_parallel_config(&ParallelSearch {
        num_gpu_list: vec![1, 2, 4, 8],
        tp_list: vec![1, 2, 4, 8],
        pp_list: vec![1],
        dp_list: vec![1, 2, 4, 8],
        moe_tp_list: vec![1, 2, 4, 8],
        moe_ep_list: vec![1, 2, 4, 8],
        is_moe: true,
        backend: BackendName::sglang,
        moe_backend: Some("deepep_moe".to_string()),
        ..ParallelSearch::default()
    });
    let py = python_json(
        r#"
import json
from aiconfigurator.sdk.utils import enumerate_parallel_config
from aiconfigurator.sdk import common
print(json.dumps(enumerate_parallel_config(
  num_gpu_list=[1,2,4,8],
  tp_list=[1,2,4,8],
  pp_list=[1],
  dp_list=[1,2,4,8],
  moe_tp_list=[1,2,4,8],
  moe_ep_list=[1,2,4,8],
  is_moe=True,
  backend=common.BackendName.sglang,
  moe_backend="deepep_moe",
)))
"#,
    );
    assert_eq!(rust, py_parallel_configs(&py));
}

#[test]
fn cli_generate_parallelism_matches_python() {
    let rust_output = std::process::Command::new(env!("CARGO_BIN_EXE_aic-rust"))
        .args([
            "cli",
            "generate",
            "--model-path",
            "Qwen/Qwen3-32B-FP8",
            "--total-gpus",
            "8",
            "--system",
            "h200_sxm",
            "--backend",
            "trtllm",
            "--json",
        ])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .unwrap();
    assert!(rust_output.status.success());
    let rust: Value = serde_json::from_slice(&rust_output.stdout).unwrap();
    let py = python_json(
        r#"
import json
from aiconfigurator.cli import cli_generate
r = cli_generate(model_path="Qwen/Qwen3-32B-FP8", total_gpus=8, system="h200_sxm", backend="trtllm")
print(json.dumps(r["parallelism"]))
"#,
    );
    assert_eq!(
        rust["tensor_parallel_size"].as_u64().unwrap(),
        py["tp"].as_u64().unwrap()
    );
    assert_eq!(
        rust["pipeline_parallel_size"].as_u64().unwrap(),
        py["pp"].as_u64().unwrap()
    );
    assert_eq!(
        rust["replicas"].as_u64().unwrap(),
        py["replicas"].as_u64().unwrap()
    );
    assert_eq!(
        rust["gpus_used"].as_u64().unwrap(),
        py["gpus_used"].as_u64().unwrap()
    );
}

#[test]
fn cli_default_compatibility_path_accepts_python_options() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_aic-rust"))
        .args(["cli", "default", "--help"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "default compatibility help failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("--request-latency"));
    assert!(stdout.contains("--enable-wideep"));
    assert!(stdout.contains("--top-n"));
}

#[test]
fn cli_default_deepseek_r1_smoke() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_aic-rust"))
        .args([
            "cli",
            "default",
            "--model-path",
            "deepseek-ai/DeepSeek-R1",
            "--total-gpus",
            "32",
            "--system",
            "h200_sxm",
            "--backend",
            "trtllm",
            "--top-n",
            "1",
            "--no-color",
            "--json",
        ])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "DeepSeek-R1 default failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["chosen_exp"].as_str().unwrap(), "disagg");
    assert!((report["best_throughput"].as_f64().unwrap() - 405.375).abs() < 0.001);
    assert!((report["disagg_ratio"].as_f64().unwrap() - 1.088).abs() < 0.001);
    let row = &report["best"]["row"];
    assert_eq!(
        row["prefill_parallel"].as_str().unwrap(),
        "tp1pp1dp8etp1ep8"
    );
    assert_eq!(row["decode_parallel"].as_str().unwrap(), "tp1pp1dp8etp1ep8");
    assert_eq!(row["decode_bs"].as_i64().unwrap(), 26);
    assert!((row["ttft"].as_f64().unwrap() - 1624.694).abs() < 0.001);
    assert!((row["tpot"].as_f64().unwrap() - 29.534).abs() < 0.001);
}

fn py_parallel_configs(value: &Value) -> Vec<ParallelConfig> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            let row = row.as_array().unwrap();
            ParallelConfig {
                tp: row[0].as_u64().unwrap() as u32,
                pp: row[1].as_u64().unwrap() as u32,
                dp: row[2].as_u64().unwrap() as u32,
                moe_tp: row[3].as_u64().unwrap() as u32,
                moe_ep: row[4].as_u64().unwrap() as u32,
            }
        })
        .collect()
}
