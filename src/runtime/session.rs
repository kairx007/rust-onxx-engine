use std::{
    fmt,
    path::{Path, PathBuf},
};

use ort::session::Session;

use crate::{
    config::ModelSpecs,
    error::AppError,
    models::ModelType,
    runtime::{ModelMetadata, Runtime},
};

pub struct ModelSession {
    model: ModelType,
    path: PathBuf,
    session: Session,
    metadata: ModelMetadata,
}

impl fmt::Debug for ModelSession {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ModelSession")
            .field("model", &self.model)
            .field("path", &self.path)
            .field("metadata", &self.metadata)
            .finish_non_exhaustive()
    }
}

impl ModelSession {
    pub fn load(
        runtime: &Runtime,
        model: ModelType,
        path: impl AsRef<Path>,
    ) -> Result<Self, AppError> {
        let path = path.as_ref();
        if !path.is_file() {
            return Err(AppError::MissingModel {
                model,
                path: path.to_owned(),
            });
        }

        let session = Session::builder(runtime.environment())
            .and_then(|mut builder| builder.commit_from_file(path))
            .map_err(|source| AppError::ModelLoad {
                model,
                path: path.to_owned(),
                source,
            })?;
        let metadata = ModelMetadata::from_session(&session);
        Ok(Self {
            model,
            path: path.to_owned(),
            session,
            metadata,
        })
    }

    pub fn model(&self) -> ModelType {
        self.model
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    pub fn validate(&self, specs: &ModelSpecs) -> Result<(), AppError> {
        self.metadata.validate(self.model, specs)
    }

    pub fn session(&self) -> &Session {
        &self.session
    }

    /// Returns mutable access for ONNX Runtime inference calls.
    pub fn session_mut(&mut self) -> &mut Session {
        &mut self.session
    }
}
