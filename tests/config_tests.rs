use std::{fs, path::PathBuf};

use rust_engine::{config::Settings, error::AppError};

const DEFAULT_CONFIG: &str = include_str!("../configs/default.toml");

#[test]
fn parses_valid_default_configuration() {
    let settings = Settings::parse(DEFAULT_CONFIG).expect("default config should parse");
    assert_eq!(settings.application.name, "RustLOB");
    assert_eq!(settings.runtime.execution_provider, "cpu");
    assert_eq!(settings.model_specs.input_shape(), [1, 1, 100, 40]);
    assert_eq!(settings.model_specs.output_classes, 3);
}

#[test]
fn reports_missing_configuration_file() {
    let path = PathBuf::from(format!(
        "{}/rust-engine-missing-config-{}.toml",
        std::env::temp_dir().display(),
        std::process::id()
    ));
    let result = Settings::load(&path);
    assert!(matches!(result, Err(AppError::ConfigRead { .. })));
}

#[test]
fn rejects_invalid_toml() {
    assert!(matches!(
        Settings::parse("[application\nname = "),
        Err(AppError::ConfigParse { .. })
    ));
}

#[test]
fn rejects_unsupported_execution_provider() {
    let invalid = DEFAULT_CONFIG.replace(
        "execution_provider = \"cpu\"",
        "execution_provider = \"cuda\"",
    );
    assert!(matches!(
        Settings::parse(&invalid),
        Err(AppError::UnsupportedExecutionProvider(_))
    ));
}

#[test]
fn rejects_empty_model_paths_and_nonpositive_dimensions() {
    let empty_path = DEFAULT_CONFIG.replace(
        "deeplob_path = \"models/deeplob.onnx\"",
        "deeplob_path = \"\"",
    );
    assert!(matches!(
        Settings::parse(&empty_path),
        Err(AppError::ConfigValidation(_))
    ));

    let invalid_shape = DEFAULT_CONFIG.replace("time_steps = 100", "time_steps = 0");
    assert!(matches!(
        Settings::parse(&invalid_shape),
        Err(AppError::ConfigValidation(_))
    ));
}

#[test]
fn loads_configuration_from_a_file() {
    let path = std::env::temp_dir().join(format!("rust-engine-config-{}.toml", std::process::id()));
    fs::write(&path, DEFAULT_CONFIG).expect("test config should be writable");
    let settings = Settings::load(&path).expect("test config should load");
    fs::remove_file(path).expect("test config should be removable");
    assert_eq!(settings.logging.level, "info");
}
