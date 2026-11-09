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
        if self.model_specs.output_classes <= 0 {
            return Err(AppError::ConfigValidation(
                "model_specs.output_classes must be positive".to_owned(),
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
