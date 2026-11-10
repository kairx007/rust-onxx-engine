use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use crate::{
    config::Settings,
    data::{FEATURE_COUNT, LobSegment},
    error::AppError,
    inference::Prediction,
    models::ModelType,
    preprocessing::{PreparedWindow, PreprocessingPipeline, WindowRef},
    runtime::Runtime,
    tensor::InputTensor,
};

use super::{
    model_factory::ModelFactory,
    output_writer::{PredictionRow, PredictionWriter},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PredictionSummary {
    pub model: ModelType,
    pub input: String,
    pub output_path: PathBuf,
    pub rows_loaded: usize,
    pub windows_generated: usize,
    pub predictions_completed: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SmokeTestResult {
    pub model: ModelType,
    pub synthetic_input: bool,
    pub prediction: Prediction,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BenchmarkSummary {
    pub model: ModelType,
    pub input: String,
    pub iterations: usize,
    pub windows_available: usize,
    pub model_load_time: Duration,
    pub inference_time: Duration,
    pub average_latency: Duration,
    pub throughput_per_second: f64,
}

pub fn run_prediction(
    settings: &Settings,
    model_type: ModelType,
    model_path_override: Option<&Path>,
    output_path: impl AsRef<Path>,
    overwrite: bool,
) -> Result<PredictionSummary, AppError> {
    let input = effective_input_description(settings);
    let runtime = Runtime::initialize(&settings.runtime.execution_provider)?;
    let mut model = ModelFactory.load(&runtime, settings, model_type, model_path_override)?;
    let pipeline = PreprocessingPipeline::new(settings)?;
    let prepared = pipeline.prepare()?;
    if prepared.windows().is_empty() {
        return Err(AppError::ConfigValidation(
            "input dataset produced no complete 100-row windows".to_owned(),
        ));
    }

    let output_path = output_path.as_ref().to_owned();
    let mut writer = PredictionWriter::create(&output_path, overwrite)?;
    for window_index in 0..prepared.windows().len() {
        let window = prepared.window(window_index)?;
        let prediction = model.predict(&window.tensor)?;
        writer.write(&prediction_row(window_index, &window, prediction))?;
    }
    writer.finish()?;

    Ok(PredictionSummary {
        model: model_type,
        input,
        output_path,
        rows_loaded: prepared.total_rows(),
        windows_generated: prepared.windows().len(),
        predictions_completed: prepared.windows().len(),
    })
}

pub fn run_smoke_test(
    settings: &Settings,
    model_type: ModelType,
    model_path_override: Option<&Path>,
    use_dataset_input: bool,
) -> Result<SmokeTestResult, AppError> {
    let runtime = Runtime::initialize(&settings.runtime.execution_provider)?;
    let mut model = ModelFactory.load(&runtime, settings, model_type, model_path_override)?;
    let (input, synthetic_input) = if use_dataset_input {
        let pipeline = PreprocessingPipeline::new(settings)?;
        let prepared = pipeline.prepare()?;
        if prepared.windows().is_empty() {
            return Err(AppError::ConfigValidation(
                "input dataset produced no complete 100-row windows".to_owned(),
            ));
        }
        (prepared.window(0)?.tensor, false)
    } else {
        (synthetic_input_tensor()?, true)
    };
    let prediction = model.predict(&input)?;
    Ok(SmokeTestResult {
        model: model_type,
        synthetic_input,
        prediction,
    })
}

pub fn run_benchmark(
    settings: &Settings,
    model_type: ModelType,
    model_path_override: Option<&Path>,
    iterations: usize,
) -> Result<BenchmarkSummary, AppError> {
    if iterations == 0 {
        return Err(AppError::ConfigValidation(
            "benchmark iterations must be greater than zero".to_owned(),
        ));
    }
    let runtime = Runtime::initialize(&settings.runtime.execution_provider)?;
    let model_load_start = Instant::now();
    let mut model = ModelFactory.load(&runtime, settings, model_type, model_path_override)?;
    let model_load_time = model_load_start.elapsed();

    let pipeline = PreprocessingPipeline::new(settings)?;
    let prepared = pipeline.prepare()?;
    let windows_available = prepared.windows().len();
    if windows_available == 0 {
        return Err(AppError::ConfigValidation(
            "input dataset produced no complete 100-row windows".to_owned(),
        ));
    }

    // Materialize one valid tensor before timing. Repeating it keeps this
    // basic command's timer focused on model execution instead of tensor copy.
    let benchmark_input = prepared.window(0)?.tensor;
    let inference_start = Instant::now();
    for _ in 0..iterations {
        let _prediction = model.predict(&benchmark_input)?;
    }
    let inference_time = inference_start.elapsed();
    let average_latency = Duration::from_secs_f64(inference_time.as_secs_f64() / iterations as f64);
    let throughput_per_second = if inference_time.is_zero() {
        0.0
    } else {
        iterations as f64 / inference_time.as_secs_f64()
    };

    Ok(BenchmarkSummary {
        model: model_type,
        input: effective_input_description(settings),
        iterations,
        windows_available,
        model_load_time,
        inference_time,
        average_latency,
        throughput_per_second,
    })
}

fn prediction_row(
    window_index: usize,
    window: &PreparedWindow,
    prediction: Prediction,
) -> PredictionRow {
    PredictionRow::new(
        window_index,
        &window.source,
        window.reference.segment_index,
        window.reference.start_row,
        window.label,
        prediction,
    )
}

fn effective_input_description(settings: &Settings) -> String {
    if !settings.data.segment_paths.is_empty() {
        settings
            .data
            .segment_paths
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    } else {
        settings.data.dataset_path.display().to_string()
    }
}

fn synthetic_input_tensor() -> Result<InputTensor, AppError> {
    let features = vec![0.0; 100 * FEATURE_COUNT];
    let labels = vec![0; 100];
    let segment = LobSegment::new("<synthetic smoke-test input>", features, labels)?;
    InputTensor::from_window(
        &segment,
        WindowRef {
            segment_index: 0,
            start_row: 0,
            label: 0,
        },
    )
}

#[cfg(test)]
mod tests {
    use crate::{config::Settings, error::AppError, models::ModelType};

    use super::{run_benchmark, synthetic_input_tensor};

    #[test]
    fn synthetic_smoke_tensor_uses_the_phase2_contract() {
        let tensor = synthetic_input_tensor().unwrap();
        assert_eq!(tensor.shape(), [1, 1, 100, 40]);
        assert_eq!(tensor.data().len(), 4_000);
        assert!(tensor.data().iter().all(|value| *value == 0.0));
    }

    #[test]
    fn benchmark_rejects_zero_iterations_before_loading_runtime_or_model() {
        let settings = Settings::parse(include_str!("../../configs/default.toml")).unwrap();
        assert!(matches!(
            run_benchmark(&settings, ModelType::DeepLob, None, 0),
            Err(AppError::ConfigValidation(_))
        ));
    }
}
