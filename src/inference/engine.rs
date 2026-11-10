use std::path::Path;

use crate::{
    config::ModelSpecs,
    error::AppError,
    inference::{Prediction, input, runner},
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
        runner::predict(
            &mut self.session,
            ModelType::DeepLob,
            self.contract.input_name(),
            self.contract.output_metadata(),
            ort_input,
        )
    }
}
