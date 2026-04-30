use crate::types::BackendName;
use anyhow::{Context, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuSpec {
    pub mem_bw: f64,
    #[serde(default)]
    pub mem_bw_empirical_scaling_factor: Option<f64>,
    #[serde(default)]
    pub mem_empirical_constant_latency: Option<f64>,
    pub mem_capacity: f64,
    #[serde(default)]
    pub bfloat16_tc_flops: Option<f64>,
    #[serde(default)]
    pub int8_tc_flops: Option<f64>,
    #[serde(default)]
    pub fp8_tc_flops: Option<f64>,
    #[serde(default)]
    pub fp4_tc_flops: Option<f64>,
    #[serde(default)]
    pub power: Option<f64>,
    #[serde(default)]
    pub sm_version: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeSpec {
    pub num_gpus_per_node: u32,
    #[serde(default)]
    pub num_gpus_per_rack: Option<u32>,
    #[serde(default)]
    pub inter_node_bw: Option<f64>,
    #[serde(default)]
    pub intra_node_bw: Option<f64>,
    #[serde(default)]
    pub inter_rack_bw: Option<f64>,
    #[serde(default)]
    pub pcie_bw: Option<f64>,
    #[serde(default)]
    pub p2p_latency: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiscSpec {
    #[serde(default)]
    pub nccl_version: Option<String>,
    #[serde(default)]
    pub oneccl_version: Option<String>,
    #[serde(default)]
    pub other_mem: Option<f64>,
    #[serde(default)]
    pub nccl_mem: Option<BTreeMap<u32, f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemSpec {
    pub data_dir: String,
    pub gpu: GpuSpec,
    pub node: NodeSpec,
    #[serde(default)]
    pub misc: Option<MiscSpec>,
}

impl SystemSpec {
    pub fn load(systems_root: impl AsRef<Path>, system: &str) -> Result<Self> {
        let path = systems_root.as_ref().join(format!("{system}.yaml"));
        let raw = fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        serde_yaml::from_str(&raw).with_context(|| format!("failed to parse {}", path.display()))
    }

    pub fn data_path(
        &self,
        systems_root: impl AsRef<Path>,
        backend: BackendName,
        version: &str,
    ) -> PathBuf {
        systems_root
            .as_ref()
            .join(&self.data_dir)
            .join(backend.to_string())
            .join(version)
    }
}

pub fn get_supported_databases(
    systems_root: impl AsRef<Path>,
) -> Result<BTreeMap<String, BTreeMap<String, Vec<String>>>> {
    let systems_root = systems_root.as_ref();
    let mut supported: BTreeMap<String, BTreeMap<String, BTreeSet<String>>> = BTreeMap::new();
    for entry in fs::read_dir(systems_root)
        .with_context(|| format!("failed to list {}", systems_root.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("yaml") {
            continue;
        }
        let Some(system_name) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        let spec = match SystemSpec::load(systems_root, system_name) {
            Ok(spec) => spec,
            Err(_) => continue,
        };
        let data_root = systems_root.join(spec.data_dir);
        if !data_root.is_dir() {
            continue;
        }
        for backend in [BackendName::trtllm, BackendName::sglang, BackendName::vllm] {
            let backend_path = data_root.join(backend.to_string());
            if !backend_path.is_dir() {
                continue;
            }
            for version in fs::read_dir(&backend_path)? {
                let version = version?;
                let version_path = version.path();
                if !version_path.is_dir() || version_path.join("INCOMPLETE.txt").is_file() {
                    continue;
                }
                if let Some(name) = version_path.file_name().and_then(|s| s.to_str()) {
                    supported
                        .entry(system_name.to_string())
                        .or_default()
                        .entry(backend.to_string())
                        .or_default()
                        .insert(name.to_string());
                }
            }
        }
    }

    Ok(supported
        .into_iter()
        .map(|(system, backends)| {
            (
                system,
                backends
                    .into_iter()
                    .map(|(backend, versions)| (backend, versions.into_iter().collect()))
                    .collect(),
            )
        })
        .collect())
}

pub fn latest_database_version(
    systems_root: impl AsRef<Path>,
    system: &str,
    backend: BackendName,
) -> Result<Option<String>> {
    let supported = get_supported_databases(systems_root)?;
    let Some(versions) = supported
        .get(system)
        .and_then(|m| m.get(&backend.to_string()))
    else {
        return Ok(None);
    };
    Ok(versions
        .iter()
        .max_by_key(|version| parse_version_key(version))
        .cloned())
}

fn parse_version_key(version: &str) -> (i64, i64, i64, i64, i64) {
    let lower = version.to_lowercase();
    let semver = Regex::new(r"(\d+)\.(\d+)\.(\d+)").unwrap();
    if let Some(caps) = semver.captures(&lower) {
        let major = caps[1].parse().unwrap_or(0);
        let minor = caps[2].parse().unwrap_or(0);
        let patch = caps[3].parse().unwrap_or(0);
        if lower.contains("rc") {
            let rc = Regex::new(r"rc(\d+)").unwrap();
            let rc_num = rc
                .captures(&lower)
                .and_then(|c| c[1].parse().ok())
                .unwrap_or(0);
            return (major, minor, patch, 0, rc_num);
        }
        return (major, minor, patch, 1, 0);
    }
    let short = Regex::new(r"v?(\d+)\.(\d+)").unwrap();
    if let Some(caps) = short.captures(&lower) {
        return (
            caps[1].parse().unwrap_or(0),
            caps[2].parse().unwrap_or(0),
            0,
            1,
            0,
        );
    }
    let nums = Regex::new(r"\d+").unwrap();
    let mut parts = nums
        .find_iter(&lower)
        .take(3)
        .map(|m| m.as_str().parse::<i64>().unwrap_or(0))
        .collect::<Vec<_>>();
    while parts.len() < 3 {
        parts.push(0);
    }
    (parts[0], parts[1], parts[2], 0, 0)
}

#[cfg(test)]
mod tests {
    use super::parse_version_key;

    #[test]
    fn parses_rc_below_stable_like_python() {
        assert!(parse_version_key("1.3.0") > parse_version_key("1.3.0rc10"));
        assert!(parse_version_key("1.3.0rc11") > parse_version_key("1.3.0rc5.post1"));
    }
}
