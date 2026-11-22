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

#[test]
fn default_data_column_mapping_is_unverified() {
    let settings = Settings::parse(DEFAULT_CONFIG).expect("default config should parse");
    assert!(!settings.data.schema_verified);
    assert_eq!(
        settings.data.output_path.as_deref(),
        Some(std::path::Path::new("outputs/predictions.csv"))
    );
}

#[test]
fn older_configs_without_cli_output_setting_remain_compatible() {
    let older = DEFAULT_CONFIG.replace("output_path = \"outputs/predictions.csv\"\n", "");
    let settings =
        Settings::parse(&older).expect("older config should use optional output default");
    assert_eq!(settings.data.output_path, None);
}

#[test]
fn rejects_an_empty_configured_output_path() {
    let invalid = DEFAULT_CONFIG.replace(
        "output_path = \"outputs/predictions.csv\"",
        "output_path = \"\"",
    );
    assert!(matches!(
        Settings::parse(&invalid),
        Err(AppError::ConfigValidation(_))
    ));
}

#[test]
fn default_config_reads_the_real_fi2010_csv_layout() {
    // FI2010_*.csv: a header line, column 0 = row index, features in columns 1-40,
    // labels in columns 145-149 (training uses 145). Windows follow the training convention.
    use rust_engine::{config::Settings, preprocessing::PreprocessingPipeline};
    let dir = std::env::temp_dir().join(format!("rustlob-raw-layout-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let csv = dir.join("FI2010_test.csv");
    let mut text = String::from(",");
    text.push_str(
        &(0..149)
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join(","),
    );
    text.push('\n');
    for row in 0..103_usize {
        let mut fields = vec![row.to_string()];
        fields.extend((0..149).map(|c| format!("{}", (row * 1000 + c) as f32)));
        fields[145] = format!("{}.0", row % 3 + 1); // FI-2010 label 1/2/3 as written by pandas
        text.push_str(&fields.join(","));
        text.push('\n');
    }
    std::fs::write(&csv, text).unwrap();
    let write_npy = |name: &str, value: f32| {
        let path = dir.join(name);
        let file = std::fs::File::create(&path).unwrap();
        let mut writer = npyz::WriteOptions::new()
            .default_dtype()
            .shape(&[40])
            .writer(std::io::BufWriter::new(file))
            .begin_nd()
            .unwrap();
        writer.extend(std::iter::repeat_n(value, 40)).unwrap();
        writer.finish().unwrap();
        path
    };
    use npyz::WriterBuilder;
    let mut settings = Settings::parse(include_str!("../configs/default.toml")).unwrap();
    settings.data.dataset_path = csv;
    settings.data.schema_verified = true;
    settings.normalization.mean_path = write_npy("mu.npy", 0.0);
    settings.normalization.std_path = write_npy("sigma.npy", 1.0);
    let prepared = PreprocessingPipeline::new(&settings)
        .unwrap()
        .prepare()
        .unwrap();
    // 103 rows, stride 1, label row after each window: 3 windows.
    assert_eq!(prepared.windows().len(), 3);
    let first = prepared.window(0).unwrap();
    // Feature 0 is raw column 1 (the index column is skipped); label is row 100's (100 % 3 + 1 = 2 -> class 1).
    assert_eq!(first.tensor.data()[0], 0.0);
    assert_eq!(first.tensor.data()[40], 1000.0 / (1.0 + 1e-8_f32));
    assert_eq!(first.label, 1);
    assert_eq!(prepared.window(2).unwrap().label, 0); // row 102: 102 % 3 + 1 = 1 -> class 0 (Up)
    std::fs::remove_dir_all(&dir).unwrap();
}
