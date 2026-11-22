use std::path::Path;

use crate::{
    config::ModelSpecs,
    error::AppError,
    inference::{Prediction, input, runner},
    models::{ModelType, TransformerLobContract},
    runtime::{ModelSession, Runtime},
    tensor::InputTensor,
};

/// TransformerLOB inference engine backed by a single reusable ONNX session.
#[derive(Debug)]
pub struct TransformerLobEngine {
    session: ModelSession,
    contract: TransformerLobContract,
}

impl TransformerLobEngine {
    pub fn load(
        runtime: &Runtime,
        model_path: impl AsRef<Path>,
        specs: &ModelSpecs,
    ) -> Result<Self, AppError> {
        let session = ModelSession::load(runtime, ModelType::TransformerLob, model_path)?;
        let contract = TransformerLobContract::from_metadata(session.metadata(), specs)?;
        Ok(Self { session, contract })
    }

    pub fn contract(&self) -> &TransformerLobContract {
        &self.contract
    }

    pub fn predict(&mut self, input_tensor: &InputTensor) -> Result<Prediction, AppError> {
        self.predict_timed(input_tensor)
            .map(|(prediction, _)| prediction)
    }

    /// Like [`Self::predict`], also returning the time spent in `Session::run` alone.
    pub fn predict_timed(
        &mut self,
        input_tensor: &InputTensor,
    ) -> Result<(Prediction, std::time::Duration), AppError> {
        let ort_input = input::adapt_transformer(input_tensor, &self.contract)?;
        runner::predict(
            &mut self.session,
            ModelType::TransformerLob,
            self.contract.input_name(),
            self.contract.output_metadata(),
            ort_input,
        )
    }
}
