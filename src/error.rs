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
    #[error("could not open dataset CSV '{path}': {source}")]
    CsvOpen {
        path: PathBuf,
        #[source]
        source: csv::Error,
    },
    #[error("malformed CSV record in '{path}' at row {row}: {source}")]
    CsvRecord {
        path: PathBuf,
        row: usize,
        #[source]
        source: csv::Error,
    },
    #[error("dataset path '{path}' is not a file or directory")]
    InvalidDatasetPath { path: PathBuf },
    #[error("no CSV segments found at '{0}'")]
    NoCsvSegments(PathBuf),
    #[error(
        "multiple CSV files were found under '{0}'; list data.segment_paths in chronological order"
    )]
    SegmentOrderRequired(PathBuf),
    #[error("blank CSV row in '{path}' at physical row {row}")]
    CsvBlankRow { path: PathBuf, row: usize },
    #[error("could not scan dataset CSV '{path}': {source}")]
    CsvScan {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("dataset segment '{path}' is empty")]
    EmptyDatasetSegment { path: PathBuf },
    #[error("CSV row {row} in '{path}' has {found} columns; expected {expected}")]
    CsvColumnCount {
        path: PathBuf,
        row: usize,
        expected: usize,
        found: usize,
    },
    #[error("invalid numeric value in '{path}' row {row}, column {column}: '{value}'")]
    CsvNumericValue {
        path: PathBuf,
        row: usize,
        column: usize,
        value: String,
    },
    #[error("invalid FI-2010 label in '{path}' row {row}: '{value}' (expected 1, 2, or 3)")]
    CsvLabel {
        path: PathBuf,
        row: usize,
        value: String,
    },
    #[error("could not enumerate dataset directory '{path}': {source}")]
    DatasetDirectory {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("could not read normalization file '{path}': {source}")]
    NormalizationRead {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid NPY normalization file '{path}': {details}")]
    NormalizationNpy { path: PathBuf, details: String },
    #[error("invalid normalization values in '{path}': {details}")]
    NormalizationValues { path: PathBuf, details: String },
    #[error("invalid dataset segment '{path}': {details}")]
    DatasetInvariant { path: PathBuf, details: String },
    #[error("invalid window configuration: {0}")]
    WindowConfiguration(String),
    #[error("invalid window reference: {0}")]
    WindowReference(String),
    #[error("invalid input tensor: {0}")]
    TensorValidation(String),
    #[error("{model} inference metadata is incompatible: {details}")]
    InferenceMetadata { model: ModelType, details: String },
    #[error("invalid inference input: {0}")]
    InvalidInferenceInput(String),
    #[error("could not create ONNX Runtime inference input: {0}")]
    InputConversion(String),
    #[error("ONNX Runtime inference failed for {model}: {source}")]
    InferenceExecution {
        model: ModelType,
        #[source]
        source: ort::Error,
    },
    #[error("model output '{name}' was not found")]
    MissingInferenceOutput { name: String },
    #[error("invalid model output: {0}")]
    InvalidInferenceOutput(String),
    #[error("could not construct prediction: {0}")]
    Prediction(String),
    #[error("prediction output already exists: {0} (pass --overwrite to replace it)")]
    OutputExists(PathBuf),
    #[error("could not create prediction output '{path}': {source}")]
    OutputCreate {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("could not write prediction output '{path}': {source}")]
    OutputWrite {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("could not serialize prediction CSV '{path}': {source}")]
    OutputCsv {
        path: PathBuf,
        #[source]
        source: csv::Error,
    },
    #[error("could not publish prediction output '{path}': {source}")]
    OutputPublish {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}
