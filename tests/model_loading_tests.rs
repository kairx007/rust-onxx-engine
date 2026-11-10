use std::{fs, path::PathBuf};

use rust_engine::{
    config::Settings,
    error::AppError,
    models::{ModelRegistry, ModelType},
    runtime::{ModelSession, Runtime},
};

const DEFAULT_CONFIG: &str = include_str!("../configs/default.toml");

#[test]
fn registry_resolves_both_configured_model_paths() {
    let settings = Settings::parse(DEFAULT_CONFIG).expect("default config should parse");
    let registry = ModelRegistry;
    assert_eq!(
        registry.path_for(&settings, ModelType::DeepLob),
        std::path::Path::new("models/deeplob.onnx")
    );
    assert_eq!(
        registry.path_for(&settings, ModelType::TransformerLob),
        std::path::Path::new("models/transformerlob.onnx")
    );
}

#[test]
fn missing_models_are_reported_with_the_model_name() {
    let mut settings = Settings::parse(DEFAULT_CONFIG).expect("default config should parse");
    settings.models.deeplob_path = std::env::temp_dir().join(format!(
        "rust-engine-missing-model-{}.onnx",
        std::process::id()
    ));
    let registry = ModelRegistry;
    let result = registry.ensure_exists(&settings, ModelType::DeepLob);
    assert!(matches!(
        result,
        Err(AppError::MissingModel {
            model: ModelType::DeepLob,
            ..
        })
    ));
}

#[test]
fn invalid_onnx_file_is_rejected_with_a_model_loading_error() {
    let path = PathBuf::from(format!(
        "{}/rust-engine-invalid-model-{}.onnx",
        std::env::temp_dir().display(),
        std::process::id()
    ));
    fs::write(&path, b"not an ONNX model").expect("test model should be writable");
    let runtime = Runtime::initialize("cpu").expect("CPU runtime should initialize");
    let result = ModelSession::load(&runtime, ModelType::DeepLob, &path);
    fs::remove_file(path).expect("test model should be removable");
    assert!(matches!(result, Err(AppError::ModelLoad { .. })));
}
