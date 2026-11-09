use std::{
    fmt,
    path::{Path, PathBuf},
};

use clap::ValueEnum;

use crate::{config::Settings, error::AppError};

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ModelType {
    #[value(name = "deeplob")]
    DeepLob,
    #[value(name = "transformerlob")]
    TransformerLob,
}

impl fmt::Display for ModelType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::DeepLob => "DeepLOB",
            Self::TransformerLob => "TransformerLOB",
        })
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ModelRegistry;

impl ModelRegistry {
    pub fn path_for(&self, settings: &Settings, model: ModelType) -> PathBuf {
        settings.models.for_model(model).to_owned()
    }

    pub fn ensure_exists(
        &self,
        settings: &Settings,
        model: ModelType,
    ) -> Result<PathBuf, AppError> {
        let path = self.path_for(settings, model);
        if !Path::new(&path).is_file() {
            return Err(AppError::MissingModel { model, path });
        }
        Ok(path)
    }
}
