use std::{
    fs,
    path::{Path, PathBuf},
};

use serde::Deserialize;

use crate::{error::AppError, models::ModelType};

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Settings {
    pub application: ApplicationSettings,
    pub runtime: RuntimeSettings,
    pub models: ModelPaths,
    pub model_specs: ModelSpecs,
    pub data: DataSettings,
    pub preprocessing: PreprocessingSettings,
    pub normalization: NormalizationSettings,
    pub tensor: TensorSettings,
    pub logging: LoggingSettings,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ApplicationSettings {
    pub name: String,
    pub environment: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeSettings {
    pub execution_provider: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelPaths {
    pub deeplob_path: PathBuf,
    pub transformerlob_path: PathBuf,
}

impl ModelPaths {
    pub fn for_model(&self, model: ModelType) -> &Path {
        match model {
            ModelType::DeepLob => &self.deeplob_path,
            ModelType::TransformerLob => &self.transformerlob_path,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelSpecs {
    pub batch_size: i64,
    pub channels: i64,
    pub time_steps: i64,
    pub features: i64,
    pub output_classes: i64,
}

impl ModelSpecs {
    pub fn input_shape(&self) -> [i64; 4] {
        [
            self.batch_size,
            self.channels,
            self.time_steps,
            self.features,
        ]
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DataSettings {
    pub dataset_path: PathBuf,
    #[serde(default)]
    pub output_path: Option<PathBuf>,
    #[serde(default)]
    pub segment_paths: Vec<PathBuf>,
    pub feature_columns: Vec<usize>,
    pub label_column: usize,
    pub has_headers: bool,
    #[serde(default)]
    pub schema_verified: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreprocessingSettings {
    pub sequence_length: usize,
    pub stride: usize,
    pub epsilon: f32,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NormalizationSettings {
    pub mean_path: PathBuf,
    pub std_path: PathBuf,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorSettings {
    pub batch_size: usize,
    pub channels: usize,
    pub time_steps: usize,
    pub features: usize,
    pub dtype: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LoggingSettings {
    pub level: String,
}

impl Settings {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, AppError> {
        let path = path.as_ref();
        let contents = fs::read_to_string(path).map_err(|source| AppError::ConfigRead {
            path: path.to_owned(),
            source,
        })?;
        Self::parse(&contents).map_err(|error| match error {
            AppError::ConfigParse { source, .. } => AppError::ConfigParse {
                path: path.to_owned(),
                source,
            },
            other => other,
        })
    }

    pub fn parse(contents: &str) -> Result<Self, AppError> {
        let settings: Self = toml::from_str(contents).map_err(|source| AppError::ConfigParse {
            path: PathBuf::from("<string>"),
            source,
        })?;
        settings.validate()?;
        Ok(settings)
    }

    pub fn validate(&self) -> Result<(), AppError> {
        if self.application.name.trim().is_empty() {
            return Err(AppError::ConfigValidation(
                "application.name must not be empty".to_owned(),
            ));
        }
        if self.application.environment.trim().is_empty() {
            return Err(AppError::ConfigValidation(
                "application.environment must not be empty".to_owned(),
            ));
        }
        if !self.runtime.execution_provider.eq_ignore_ascii_case("cpu") {
            return Err(AppError::UnsupportedExecutionProvider(
                self.runtime.execution_provider.clone(),
            ));
        }
        if self.models.deeplob_path.as_os_str().is_empty()
            || self.models.transformerlob_path.as_os_str().is_empty()
        {
            return Err(AppError::ConfigValidation(
                "model paths must not be empty".to_owned(),
            ));
        }
        if self
            .model_specs
            .input_shape()
            .iter()
            .any(|dimension| *dimension <= 0)
        {
            return Err(AppError::ConfigValidation(
                "model input dimensions must be positive".to_owned(),
            ));
        }
        if self.model_specs.input_shape() != [1, 1, 100, 40] {
            return Err(AppError::ConfigValidation(
                "Phase 2 model input shape must remain [1, 1, 100, 40]".to_owned(),
            ));
        }
        if self.model_specs.output_classes != 3 {
            return Err(AppError::ConfigValidation(
                "model_specs.output_classes must remain 3".to_owned(),
            ));
        }
        if self.data.dataset_path.as_os_str().is_empty() {
            return Err(AppError::ConfigValidation(
                "data.dataset_path must not be empty".to_owned(),
            ));
        }
        if self
            .data
            .output_path
            .as_ref()
            .is_some_and(|path| path.as_os_str().is_empty())
        {
            return Err(AppError::ConfigValidation(
                "data.output_path must not be empty when provided".to_owned(),
            ));
        }
        if self.data.feature_columns.len() != 40 {
            return Err(AppError::ConfigValidation(format!(
                "data.feature_columns must contain exactly 40 indices; found {}",
                self.data.feature_columns.len()
            )));
        }
        let mut feature_columns = self.data.feature_columns.clone();
        feature_columns.sort_unstable();
        feature_columns.dedup();
        if feature_columns.len() != 40 {
            return Err(AppError::ConfigValidation(
                "data.feature_columns must not contain duplicates".to_owned(),
            ));
        }
        if self.data.feature_columns.contains(&self.data.label_column) {
            return Err(AppError::ConfigValidation(
                "data.label_column must not also be a feature column".to_owned(),
            ));
        }
        if self
            .data
            .segment_paths
            .iter()
            .any(|path| path.as_os_str().is_empty())
        {
            return Err(AppError::ConfigValidation(
                "data.segment_paths must not contain empty paths".to_owned(),
            ));
        }
        if self.preprocessing.sequence_length != 100 || self.preprocessing.stride != 100 {
            return Err(AppError::ConfigValidation(
                "Phase 2 requires sequence_length and stride to both equal 100".to_owned(),
            ));
        }
        if !self.preprocessing.epsilon.is_finite() || self.preprocessing.epsilon != 1e-8_f32 {
            return Err(AppError::ConfigValidation(
                "preprocessing.epsilon must be finite and exactly 1e-8".to_owned(),
            ));
        }
        if self.normalization.mean_path.as_os_str().is_empty()
            || self.normalization.std_path.as_os_str().is_empty()
        {
            return Err(AppError::ConfigValidation(
                "normalization mean_path and std_path must not be empty".to_owned(),
            ));
        }
        let tensor_shape = [
            self.tensor.batch_size,
            self.tensor.channels,
            self.tensor.time_steps,
            self.tensor.features,
        ];
        if tensor_shape != [1, 1, 100, 40] {
            return Err(AppError::ConfigValidation(format!(
                "Phase 2 tensor shape must remain [1, 1, 100, 40]; found {tensor_shape:?}"
            )));
        }
        if !self.tensor.dtype.eq_ignore_ascii_case("float32") {
            return Err(AppError::ConfigValidation(
                "tensor.dtype must be float32".to_owned(),
            ));
        }
        if !matches!(
            self.logging.level.to_ascii_lowercase().as_str(),
            "error" | "warn" | "info" | "debug" | "trace"
        ) {
            return Err(AppError::ConfigValidation(format!(
                "unsupported logging level '{}' (expected error, warn, info, debug, or trace)",
                self.logging.level
            )));
        }
        Ok(())
    }
}
