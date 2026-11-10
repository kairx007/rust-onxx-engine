use std::path::{Path, PathBuf};

use crate::{
    config::Settings,
    error::AppError,
    models::{DeepLobContract, ModelRegistry, ModelType, TransformerLobContract},
    runtime::{ModelMetadata, ModelSession, Runtime},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfiguredModelStatus {
    pub model: ModelType,
    pub path: PathBuf,
    pub file_available: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeSummary {
    pub application_name: String,
    pub application_version: &'static str,
    pub execution_provider: &'static str,
    pub ort_api_target: u32,
    pub models: [ConfiguredModelStatus; 2],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelInspection {
    pub model: ModelType,
    pub path: PathBuf,
    pub metadata: Option<ModelMetadata>,
    pub validation_error: Option<String>,
}

impl ModelInspection {
    pub fn file_available(&self) -> bool {
        self.metadata.is_some()
    }

    pub fn into_error(&self) -> Option<AppError> {
        if !self.file_available() {
            return Some(AppError::MissingModel {
                model: self.model,
                path: self.path.clone(),
            });
        }
        self.validation_error
            .as_ref()
            .map(|details| AppError::InferenceMetadata {
                model: self.model,
                details: details.clone(),
            })
    }
}

pub fn runtime_info(settings: &Settings) -> Result<RuntimeSummary, AppError> {
    let runtime = Runtime::initialize(&settings.runtime.execution_provider)?;
    let registry = ModelRegistry;
    let models = [ModelType::DeepLob, ModelType::TransformerLob].map(|model| {
        let path = registry.path_for(settings, model);
        ConfiguredModelStatus {
            model,
            file_available: path.is_file(),
            path,
        }
    });
    Ok(RuntimeSummary {
        application_name: settings.application.name.clone(),
        application_version: env!("CARGO_PKG_VERSION"),
        execution_provider: runtime.execution_provider(),
        ort_api_target: ort::MINOR_VERSION,
        models,
    })
}

pub fn inspect_model(
    settings: &Settings,
    model: ModelType,
    path_override: Option<&Path>,
) -> Result<ModelInspection, AppError> {
    let configured_path = ModelRegistry.path_for(settings, model);
    let path = path_override.unwrap_or(&configured_path).to_owned();
    if !path.is_file() {
        return Ok(ModelInspection {
            model,
            path,
            metadata: None,
            validation_error: None,
        });
    }

    let runtime = Runtime::initialize(&settings.runtime.execution_provider)?;
    let loaded = ModelSession::load(&runtime, model, &path)?;
    let metadata = loaded.metadata().clone();
    let validation_error = validate_contract(model, &metadata, settings)
        .err()
        .map(|error| error.to_string());
    Ok(ModelInspection {
        model,
        path,
        metadata: Some(metadata),
        validation_error,
    })
}

fn validate_contract(
    model: ModelType,
    metadata: &ModelMetadata,
    settings: &Settings,
) -> Result<(), AppError> {
    match model {
        ModelType::DeepLob => {
            DeepLobContract::from_metadata(metadata, &settings.model_specs).map(|_| ())
        }
        ModelType::TransformerLob => {
            TransformerLobContract::from_metadata(metadata, &settings.model_specs).map(|_| ())
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::{config::Settings, models::ModelType};

    use super::inspect_model;

    #[test]
    fn inspection_reports_a_missing_selected_model_without_claiming_availability() {
        let settings = Settings::parse(include_str!("../../configs/default.toml")).unwrap();
        let path = PathBuf::from("models/not-present-transformer.onnx");
        let report = inspect_model(&settings, ModelType::TransformerLob, Some(&path)).unwrap();
        assert_eq!(report.path, path);
        assert!(!report.file_available());
        assert!(report.metadata.is_none());
        assert!(report.into_error().is_some());
    }
}
