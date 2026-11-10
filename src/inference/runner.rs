use ort::value::TensorRef;

use crate::{
    error::AppError,
    inference::{Prediction, output},
    models::ModelType,
    runtime::{ModelSession, TensorMetadata},
};

/// Shared named-input ONNX execution and three-class output handling.
pub(super) fn predict(
    session: &mut ModelSession,
    model: ModelType,
    input_name: &str,
    output_metadata: &TensorMetadata,
    input: TensorRef<'_, f32>,
) -> Result<Prediction, AppError> {
    let output_name = output_metadata.name.as_str();
    let outputs = session
        .session_mut()
        .run(ort::inputs![input_name => input])
        .map_err(|source| AppError::InferenceExecution { model, source })?;
    let output = outputs
        .get(output_name)
        .ok_or_else(|| AppError::MissingInferenceOutput {
            name: output_name.to_owned(),
        })?;
    output::extract(output, output_metadata)
}
