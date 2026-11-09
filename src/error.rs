use std::path::PathBuf;

use thiserror::Error;

use crate::models::ModelType;

#[derive(Debug, Error)]
pub enum AppError {
    #[error("could not read configuration file '{path}': {source}")]
    ConfigRead {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("could not parse configuration file '{path}': {source}")]
    ConfigParse {
        path: PathBuf,
        #[source]
        source: toml::de::Error,
    },
    #[error("invalid configuration: {0}")]
    ConfigValidation(String),
    #[error("unsupported execution provider '{0}'; Phase 1 supports only 'cpu'")]
    UnsupportedExecutionProvider(String),
    #[error("{model} model file does not exist: {path}")]
    MissingModel { model: ModelType, path: PathBuf },
    #[error("failed to initialize ONNX Runtime: {0}")]
    RuntimeInitialization(String),
    #[error("failed to load {model} model at '{path}': {source}")]
    ModelLoad {
        model: ModelType,
        path: PathBuf,
        #[source]
        source: ort::Error,
    },
    #[error("{model} metadata is incompatible: {details}")]
    MetadataValidation { model: ModelType, details: String },
    #[error("could not initialize logging: {0}")]
    Logging(String),
}
