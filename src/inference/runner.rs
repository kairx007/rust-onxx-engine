use std::time::{Duration, Instant};

use ort::value::TensorRef;

use crate::{
    error::AppError,
    inference::{Prediction, output},
    models::ModelType,
    runtime::{ModelSession, TensorMetadata},
};

/// Shared named-input ONNX execution and three-class output handling.
///
/// Returns the prediction and the wall-clock time spent inside
/// `Session::run` alone, so callers can separate ONNX Runtime execution from
/// the Rust-side input adaptation and output parsing around it.
pub(super) fn predict(
    session: &mut ModelSession,
    model: ModelType,
    input_name: &str,
    output_metadata: &TensorMetadata,
    input: TensorRef<'_, f32>,
) -> Result<(Prediction, Duration), AppError> {
    let output_name = output_metadata.name.as_str();
    let run_start = Instant::now();
    let outputs = session
        .session_mut()
        .run(ort::inputs![input_name => input])
        .map_err(|source| AppError::InferenceExecution { model, source })?;
    let run_time = run_start.elapsed();
    let output = outputs
        .get(output_name)
        .ok_or_else(|| AppError::MissingInferenceOutput {
            name: output_name.to_owned(),
        })?;
    Ok((output::extract(output, output_metadata)?, run_time))
}
