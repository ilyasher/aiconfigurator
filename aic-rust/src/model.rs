use crate::paths::default_model_configs_root;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelInfo {
    pub architecture: String,
    pub layers: u64,
    pub n: u64,
    pub n_kv: u64,
    pub d: u64,
    pub hidden_size: u64,
    pub inter_size: u64,
    pub vocab: u64,
    pub context: u64,
    pub topk: u64,
    pub num_experts: u64,
    pub moe_inter_size: u64,
    pub raw_config: Value,
}

pub fn load_model_info(model_path: &str) -> Result<ModelInfo> {
    let raw = load_raw_config(model_path)?;
    parse_raw_config(raw)
}

pub fn load_raw_config(model_path: &str) -> Result<Value> {
    let path = Path::new(model_path);
    if path.is_dir() {
        let raw = read_json_with_non_finite(path.join("config.json"))?;
        let hf_quant = read_optional_json_with_non_finite(path.join("hf_quant_config.json"))?;
        return Ok(attach_hf_quant_config(raw, hf_quant));
    }
    let cached =
        default_model_configs_root().join(format!("{}_config.json", model_path.replace('/', "--")));
    if cached.is_file() {
        let raw = read_json_with_non_finite(cached)?;
        let hf_quant_path = default_model_configs_root().join(format!(
            "{}_hf_quant_config.json",
            model_path.replace('/', "--")
        ));
        let hf_quant = read_optional_json_with_non_finite(hf_quant_path)?;
        return Ok(attach_hf_quant_config(raw, hf_quant));
    }
    bail!("model config not found locally for {model_path}; Rust port currently uses cached/local configs only")
}

fn read_json_with_non_finite(path: PathBuf) -> Result<Value> {
    let mut raw =
        fs::read_to_string(&path).with_context(|| format!("failed to read {}", path.display()))?;
    raw = raw.replace("-Infinity", "null");
    raw = raw.replace("Infinity", "null");
    raw = raw.replace("NaN", "null");
    serde_json::from_str(&raw).with_context(|| format!("failed to parse {}", path.display()))
}

fn read_optional_json_with_non_finite(path: PathBuf) -> Result<Option<Value>> {
    if path.is_file() {
        read_json_with_non_finite(path).map(Some)
    } else {
        Ok(None)
    }
}

fn attach_hf_quant_config(mut raw: Value, hf_quant: Option<Value>) -> Value {
    let Some(hf_quant) = hf_quant else {
        return raw;
    };
    let Some(raw_obj) = raw.as_object_mut() else {
        return raw;
    };

    raw_obj.insert("hf_quant_config".to_string(), hf_quant.clone());
    if raw_obj.contains_key("quantization_config") {
        return raw;
    }

    let Some(quant_section) = hf_quant.get("quantization").and_then(Value::as_object) else {
        return raw;
    };
    let mut quantization_config = Map::new();
    if let Some(value) = quant_section.get("quant_algo") {
        quantization_config.insert("quant_algo".to_string(), value.clone());
    }
    if let Some(value) = quant_section.get("kv_cache_quant_algo") {
        quantization_config.insert("kv_cache_quant_algo".to_string(), value.clone());
    }
    if !quantization_config.is_empty() {
        raw_obj.insert(
            "quantization_config".to_string(),
            Value::Object(quantization_config),
        );
    }
    raw
}

fn parse_raw_config(raw_config: Value) -> Result<ModelInfo> {
    let mut cfg = raw_config.clone();
    let architecture = cfg
        .get("architectures")
        .and_then(Value::as_array)
        .and_then(|arr| arr.first())
        .and_then(Value::as_str)
        .context("missing architectures[0]")?
        .to_string();

    // Same multimodal unwrap shape as Python for common configs. This keeps the
    // top-level architecture while reading LLM dimensions from the text subconfig.
    if let Some(text_key) = multimodal_text_key(&architecture) {
        if let Some(text_cfg) = cfg.get(text_key).and_then(Value::as_object) {
            let mut merged = serde_json::Map::new();
            for (k, v) in text_cfg {
                merged.insert(k.clone(), v.clone());
            }
            merged.insert(
                "architectures".to_string(),
                Value::Array(vec![Value::String(architecture.clone())]),
            );
            cfg = Value::Object(merged);
        }
    }

    let layers = u64_field(&cfg, "num_hidden_layers")?;
    let hidden_size = u64_field(&cfg, "hidden_size")?;
    let n = u64_field(&cfg, "num_attention_heads")?;
    let vocab = u64_field(&cfg, "vocab_size")?;
    let context = u64_field(&cfg, "max_position_embeddings")?;
    let n_kv = optional_u64(&cfg, "num_key_value_heads").unwrap_or(0);
    let inter_size = optional_u64(&cfg, "intermediate_size").unwrap_or(0);
    let d = optional_u64(&cfg, "head_dim")
        .or_else(|| optional_u64(&cfg, "attention_head_dim"))
        .unwrap_or_else(|| if n > 0 { hidden_size / n } else { 0 });
    let topk = optional_u64(&cfg, "num_experts_per_tok").unwrap_or(0);
    let num_experts = optional_u64(&cfg, "num_local_experts")
        .or_else(|| optional_u64(&cfg, "n_routed_experts"))
        .or_else(|| optional_u64(&cfg, "num_experts"))
        .unwrap_or(0);
    let moe_inter_size = optional_u64(&cfg, "moe_intermediate_size").unwrap_or(inter_size);

    Ok(ModelInfo {
        architecture,
        layers,
        n,
        n_kv,
        d,
        hidden_size,
        inter_size,
        vocab,
        context,
        topk,
        num_experts,
        moe_inter_size,
        raw_config,
    })
}

fn multimodal_text_key(architecture: &str) -> Option<&'static str> {
    match architecture {
        "Llama4ForConditionalGeneration" => Some("text_config"),
        _ => None,
    }
}

fn optional_u64(cfg: &Value, key: &str) -> Option<u64> {
    cfg.get(key).and_then(Value::as_u64)
}

fn u64_field(cfg: &Value, key: &str) -> Result<u64> {
    optional_u64(cfg, key).with_context(|| format!("missing or non-integer model field {key}"))
}

pub fn estimate_model_weight_bytes(model_path: &str) -> Result<u128> {
    let config = load_model_info(model_path)?;
    let embedding_params = config.vocab as u128 * config.hidden_size as u128;
    let attention_params = 4_u128 * config.hidden_size as u128 * config.hidden_size as u128;
    let ffn_params = if config.num_experts > 1 {
        let ffn_inter = if config.moe_inter_size > 0 {
            config.moe_inter_size
        } else {
            config.inter_size
        };
        3_u128 * config.hidden_size as u128 * ffn_inter as u128 * config.num_experts as u128
            + config.hidden_size as u128 * config.num_experts as u128
    } else {
        3_u128 * config.hidden_size as u128 * config.inter_size as u128
    };
    let norm_params = 4_u128 * config.hidden_size as u128;
    let per_layer = attention_params + ffn_params + norm_params;
    Ok((embedding_params + config.layers as u128 * per_layer) * 2)
}
