use std::path::Path;

use crate::{
    config::ModelSpecs,
    error::AppError,
    inference::{Prediction, input, output},
    models::{DeepLobContract, ModelType},
    runtime::{ModelSession, Runtime},
    tensor::InputTensor,
};

/// Reusable DeepLOB ONNX inference engine. The session is loaded once and
/// reused for every prediction.
#[derive(Debug)]
pub struct DeepLobEngine {
    session: ModelSession,
    contract: DeepLobContract,
}

impl DeepLobEngine {
    pub fn load(
        runtime: &Runtime,
        model_path: impl AsRef<Path>,
        specs: &ModelSpecs,
    ) -> Result<Self, AppError> {
        let session = ModelSession::load(runtime, ModelType::DeepLob, model_path)?;
        let contract = DeepLobContract::from_metadata(session.metadata(), specs)?;
        Ok(Self { session, contract })
    }

    pub fn contract(&self) -> &DeepLobContract {
        &self.contract
    }

    pub fn predict(&mut self, input_tensor: &InputTensor) -> Result<Prediction, AppError> {
        let ort_input = input::adapt(input_tensor)?;
        let outputs = self
            .session
            .session_mut()
            .run(ort::inputs![self.contract.input_name() => ort_input])
            .map_err(|source| AppError::InferenceExecution {
                model: ModelType::DeepLob,
                source,
            })?;
        let output = outputs.get(self.contract.output_name()).ok_or_else(|| {
            AppError::MissingInferenceOutput {
                name: self.contract.output_name().to_owned(),
            }
        })?;
        output::extract(output, self.contract.output_metadata())
    }
}
