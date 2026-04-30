use std::env;
use std::path::PathBuf;

pub fn python_repo_root() -> PathBuf {
    env::var_os("AIC_PYTHON_REPO")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from(".."))
        })
}

pub fn default_systems_root() -> PathBuf {
    env::var_os("AIC_SYSTEMS_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| python_repo_root().join("src/aiconfigurator/systems"))
}

pub fn default_model_configs_root() -> PathBuf {
    env::var_os("AIC_MODEL_CONFIGS_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| python_repo_root().join("src/aiconfigurator/model_configs"))
}
