use std::path::Path;

use crate::{
    config::Settings,
    error::AppError,
    inference::{DeepLobEngine, Prediction, TransformerLobEngine},
    models::{ModelRegistry, ModelType},
    runtime::Runtime,
    tensor::InputTensor,
};

#[derive(Debug)]
pub enum LoadedModel {
    DeepLob(DeepLobEngine),
    TransformerLob(TransformerLobEngine),
}

impl LoadedModel {
    pub fn predict(&mut self, input: &InputTensor) -> Result<Prediction, AppError> {
        match self {
            Self::DeepLob(engine) => engine.predict(input),
            Self::TransformerLob(engine) => engine.predict(input),
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ModelFactory;

impl ModelFactory {
    pub fn load(
        &self,
        runtime: &Runtime,
        settings: &Settings,
        model: ModelType,
        path_override: Option<&Path>,
    ) -> Result<LoadedModel, AppError> {
        let registry = ModelRegistry;
        let configured_path = registry.path_for(settings, model);
        let path = path_override.unwrap_or(&configured_path);
        match model {
            ModelType::DeepLob => Ok(LoadedModel::DeepLob(DeepLobEngine::load(
                runtime,
                path,
                &settings.model_specs,
            )?)),
            ModelType::TransformerLob => Ok(LoadedModel::TransformerLob(
                TransformerLobEngine::load(runtime, path, &settings.model_specs)?,
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use crate::{config::Settings, error::AppError, models::ModelType, runtime::Runtime};

    use super::ModelFactory;

    #[test]
    fn selects_transformerlob_path_and_reports_its_missing_file() {
        let settings = Settings::parse(include_str!("../../configs/default.toml")).unwrap();
        let runtime = Runtime::initialize("cpu").unwrap();
        let missing = Path::new("models/transformerlob-not-present.onnx");
        assert!(matches!(
            ModelFactory.load(
                &runtime,
                &settings,
                ModelType::TransformerLob,
                Some(missing)
            ),
            Err(AppError::MissingModel {
                model: ModelType::TransformerLob,
                ..
            })
        ));
    }
}
