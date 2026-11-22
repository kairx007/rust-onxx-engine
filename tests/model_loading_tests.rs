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

#[test]
fn committed_models_match_reference_logits_on_zero_input_with_cpu_policy() {
    // Values produced by the PyPI onnxruntime 1.31.0 CPU provider on an all-zero window.
    // With the ort crate's implicit MaxEfficiency device policy, DeepLOB partitions run on
    // CoreML (macOS) and these values differ by ~6e-4, so this test pins CPU-only execution.
    let settings =
        rust_engine::config::Settings::parse(include_str!("../configs/default.toml")).unwrap();
    let runtime = rust_engine::runtime::Runtime::from_settings(&settings.runtime).unwrap();
    let path = std::path::Path::new("models/deeplob.onnx");
    if !path.is_file() {
        return;
    }
    let mut engine =
        rust_engine::inference::DeepLobEngine::load(&runtime, path, &settings.model_specs).unwrap();
    let segment =
        rust_engine::data::LobSegment::new("zeros", vec![0.0; 4_040], vec![1; 101]).unwrap();
    let tensor = rust_engine::tensor::InputTensor::from_window(
        &segment,
        rust_engine::preprocessing::WindowRef {
            segment_index: 0,
            start_row: 0,
            label: 1,
        },
    )
    .unwrap();
    let scores = engine.predict(&tensor).unwrap().scores();
    let reference = [-0.490_300_45_f32, 1.918_380_7, -1.846_455_7];
    for (actual, expected) in scores.iter().zip(reference) {
        assert!(
            (actual - expected).abs() <= 1e-5,
            "scores {scores:?} differ from CPU reference {reference:?}"
        );
    }
}
