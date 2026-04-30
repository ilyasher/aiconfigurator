use crate::model::load_model_info;
use crate::paths::default_systems_root;
use crate::types::BackendName;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
struct SupportRow {
    #[serde(rename = "HuggingFaceID")]
    hugging_face_id: String,
    #[serde(rename = "Architecture")]
    architecture: String,
    #[serde(rename = "System")]
    system: String,
    #[serde(rename = "Backend")]
    backend: String,
    #[serde(rename = "Version")]
    version: String,
    #[serde(rename = "Mode")]
    mode: String,
    #[serde(rename = "Status")]
    status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SupportResult {
    pub agg_supported: bool,
    pub disagg_supported: bool,
    pub exact_match: bool,
    pub architecture: Option<String>,
    pub agg_pass_count: usize,
    pub agg_total_count: usize,
    pub disagg_pass_count: usize,
    pub disagg_total_count: usize,
}

pub fn check_support(
    model: &str,
    system: &str,
    backend: Option<BackendName>,
    version: Option<&str>,
) -> Result<SupportResult> {
    let matrix = load_support_matrix(default_systems_root().join("support_matrix.csv"))?;
    let backend_filter = backend.map(|b| b.to_string());

    let matches_filters = |row: &&SupportRow| {
        backend_filter
            .as_ref()
            .map_or(true, |backend| row.backend == *backend)
            && version.map_or(true, |version| row.version == version)
    };

    let exact_matches = matrix
        .iter()
        .filter(|row| {
            row.hugging_face_id.eq_ignore_ascii_case(model)
                && row.system.eq_ignore_ascii_case(system)
                && matches_filters(row)
        })
        .collect::<Vec<_>>();

    if !exact_matches.is_empty() {
        return Ok(SupportResult {
            agg_supported: exact_matches
                .iter()
                .any(|row| row.mode == "agg" && row.status == "PASS"),
            disagg_supported: exact_matches
                .iter()
                .any(|row| row.mode == "disagg" && row.status == "PASS"),
            exact_match: true,
            architecture: None,
            agg_pass_count: 0,
            agg_total_count: 0,
            disagg_pass_count: 0,
            disagg_total_count: 0,
        });
    }

    let architecture = matrix
        .iter()
        .find(|row| row.hugging_face_id == model)
        .map(|row| row.architecture.clone())
        .or_else(|| load_model_info(model).ok().map(|info| info.architecture));

    let Some(architecture) = architecture else {
        return Ok(SupportResult {
            agg_supported: false,
            disagg_supported: false,
            exact_match: false,
            architecture: None,
            agg_pass_count: 0,
            agg_total_count: 0,
            disagg_pass_count: 0,
            disagg_total_count: 0,
        });
    };

    let arch_matches = matrix
        .iter()
        .filter(|row| {
            row.architecture == architecture && row.system == system && matches_filters(row)
        })
        .collect::<Vec<_>>();

    let agg_results = arch_matches
        .iter()
        .filter(|row| row.mode == "agg")
        .map(|row| row.status == "PASS")
        .collect::<Vec<_>>();
    let disagg_results = arch_matches
        .iter()
        .filter(|row| row.mode == "disagg")
        .map(|row| row.status == "PASS")
        .collect::<Vec<_>>();

    let majority = |values: &[bool]| values.iter().filter(|&&v| v).count() * 2 > values.len();

    Ok(SupportResult {
        agg_supported: !agg_results.is_empty() && majority(&agg_results),
        disagg_supported: !disagg_results.is_empty() && majority(&disagg_results),
        exact_match: false,
        architecture: Some(architecture),
        agg_pass_count: agg_results.iter().filter(|&&v| v).count(),
        agg_total_count: agg_results.len(),
        disagg_pass_count: disagg_results.iter().filter(|&&v| v).count(),
        disagg_total_count: disagg_results.len(),
    })
}

fn load_support_matrix(path: impl AsRef<Path>) -> Result<Vec<SupportRow>> {
    let path = path.as_ref();
    let mut reader = csv::Reader::from_path(path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    reader
        .deserialize()
        .collect::<std::result::Result<Vec<_>, _>>()
        .with_context(|| format!("failed to parse {}", path.display()))
}
