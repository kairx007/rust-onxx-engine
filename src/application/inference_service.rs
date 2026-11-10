use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use crate::{
    benchmark::{
        BenchmarkMetadata, BenchmarkReport, BenchmarkTrial, PreprocessingStageMetrics,
        ReportFormat, ResourceMetrics, ResourceSampler, calculate_latency_statistics,
        calculate_throughput,
    },
    cli::BenchmarkMode,
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
    pub runtime_init_time: Duration,
    pub inference_time: Duration,
    pub average_latency: Duration,
    pub throughput_per_second: f64,
    pub warmup_iterations: usize,
    pub trials: Vec<BenchmarkTrial>,
    pub aggregate: Option<crate::benchmark::LatencyStatistics>,
    pub resource_metrics: Option<ResourceMetrics>,
    pub report_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy)]
pub struct BenchmarkOptions<'a> {
    pub iterations: usize,
    pub warmup: usize,
    pub trial_count: usize,
    pub mode: BenchmarkMode,
    pub profile_memory: bool,
    pub profile_cpu: bool,
    pub report: Option<(&'a Path, ReportFormat)>,
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
    options: BenchmarkOptions<'_>,
) -> Result<BenchmarkSummary, AppError> {
    let iterations = options.iterations;
    if iterations == 0 || options.trial_count == 0 {
        return Err(AppError::ConfigValidation(
            "benchmark iterations and trials must be greater than zero".to_owned(),
        ));
    }
    let sampler = (options.profile_memory || options.profile_cpu)
        .then(|| ResourceSampler::start(Duration::from_millis(100)));
    let input_description = effective_input_description(settings);
    let measured_iterations = if options.mode == BenchmarkMode::Inference {
        iterations
    } else {
        1
    };
    let model_path = model_path_override.unwrap_or_else(|| settings.models.for_model(model_type));
    let runtime_start = Instant::now();
    let runtime = if options.mode == BenchmarkMode::Preprocessing {
        None
    } else {
        Some(Runtime::initialize(&settings.runtime.execution_provider)?)
    };
    let runtime_init_time = runtime_start.elapsed();
    let load_start = Instant::now();
    let mut model = match (&runtime, options.mode) {
        (Some(runtime), BenchmarkMode::Inference | BenchmarkMode::EndToEnd) => {
            Some(ModelFactory.load(runtime, settings, model_type, Some(model_path))?)
        }
        _ => None,
    };
    let model_load_time = load_start.elapsed();
    let pipeline = (options.mode != BenchmarkMode::Load)
        .then(|| PreprocessingPipeline::new(settings))
        .transpose()?;
    let mut windows_available = 0;
    let mut trials = Vec::with_capacity(options.trial_count);
    let mut all_samples = Vec::new();

    if options.mode == BenchmarkMode::Inference {
        let prepared = pipeline
            .as_ref()
            .expect("inference mode prepares input")
            .prepare()?;
        windows_available = prepared.windows().len();
        if windows_available == 0 {
            return Err(no_windows_error());
        }
        let input = prepared.window(0)?.tensor;
        let model = model.as_mut().expect("inference mode loads model");
        for _ in 0..options.warmup {
            let _ = model.predict(&input)?;
        }
        for _ in 0..options.trial_count {
            let start = Instant::now();
            let mut samples = Vec::with_capacity(iterations);
            for _ in 0..iterations {
                let call_start = Instant::now();
                let _prediction = model.predict(&input)?;
                samples.push(call_start.elapsed().as_secs_f64() * 1_000.0);
            }
            let elapsed = start.elapsed().as_secs_f64();
            let stats = calculate_latency_statistics(&samples);
            all_samples.extend(samples.iter().copied());
            trials.push(BenchmarkTrial {
                elapsed_seconds: elapsed,
                completed_windows: iterations,
                throughput_per_second: calculate_throughput(iterations, elapsed),
                samples_ms: samples,
                statistics: stats,
                preprocessing_stages: None,
            });
        }
    } else if options.mode == BenchmarkMode::Preprocessing {
        for _ in 0..options.warmup {
            let (prepared, _) = pipeline
                .as_ref()
                .expect("preprocessing mode prepares input")
                .prepare_with_timings()?;
            for index in 0..prepared.windows().len() {
                let _ = prepared.window(index)?;
            }
        }
        for _ in 0..options.trial_count {
            let start = Instant::now();
            let (prepared, stage_times) = pipeline
                .as_ref()
                .expect("preprocessing mode prepares input")
                .prepare_with_timings()?;
            windows_available = prepared.windows().len();
            if windows_available == 0 {
                return Err(no_windows_error());
            }
            let tensor_start = Instant::now();
            for index in 0..windows_available {
                let _ = prepared.window(index)?;
            }
            let tensor_construction = tensor_start.elapsed();
            let elapsed = start.elapsed().as_secs_f64();
            let sample = elapsed * 1_000.0;
            all_samples.push(sample);
            trials.push(BenchmarkTrial {
                elapsed_seconds: elapsed,
                completed_windows: windows_available,
                throughput_per_second: calculate_throughput(windows_available, elapsed),
                samples_ms: vec![sample],
                statistics: calculate_latency_statistics(&[sample]),
                preprocessing_stages: Some(PreprocessingStageMetrics {
                    csv_loading_ms: stage_times.csv_loading.as_secs_f64() * 1_000.0,
                    normalization_ms: stage_times.normalization.as_secs_f64() * 1_000.0,
                    window_generation_ms: stage_times.window_generation.as_secs_f64() * 1_000.0,
                    tensor_construction_ms: tensor_construction.as_secs_f64() * 1_000.0,
                }),
            });
        }
    } else if options.mode == BenchmarkMode::EndToEnd {
        let model = model.as_mut().expect("end-to-end mode loads model");
        for _ in 0..options.warmup {
            let (prepared, _) = pipeline
                .as_ref()
                .expect("end-to-end mode prepares input")
                .prepare_with_timings()?;
            for index in 0..prepared.windows().len() {
                let window = prepared.window(index)?;
                let _ = model.predict(&window.tensor)?;
            }
        }
        for _ in 0..options.trial_count {
            let start = Instant::now();
            let (prepared, stage_times) = pipeline
                .as_ref()
                .expect("end-to-end mode prepares input")
                .prepare_with_timings()?;
            windows_available = prepared.windows().len();
            if windows_available == 0 {
                return Err(no_windows_error());
            }
            let mut output = csv::Writer::from_writer(Vec::new());
            let mut tensor_construction = Duration::ZERO;
            for index in 0..windows_available {
                let tensor_start = Instant::now();
                let window = prepared.window(index)?;
                tensor_construction += tensor_start.elapsed();
                let prediction = model.predict(&window.tensor)?;
                output
                    .serialize(prediction_row(index, &window, prediction))
                    .map_err(|error| AppError::BenchmarkSerialization(error.to_string()))?;
            }
            let _serialized_predictions = output
                .into_inner()
                .map_err(|error| AppError::BenchmarkSerialization(error.to_string()))?;
            let elapsed = start.elapsed().as_secs_f64();
            let sample = elapsed * 1_000.0;
            all_samples.push(sample);
            trials.push(BenchmarkTrial {
                elapsed_seconds: elapsed,
                completed_windows: windows_available,
                throughput_per_second: calculate_throughput(windows_available, elapsed),
                samples_ms: vec![sample],
                statistics: calculate_latency_statistics(&[sample]),
                preprocessing_stages: Some(PreprocessingStageMetrics {
                    csv_loading_ms: stage_times.csv_loading.as_secs_f64() * 1_000.0,
                    normalization_ms: stage_times.normalization.as_secs_f64() * 1_000.0,
                    window_generation_ms: stage_times.window_generation.as_secs_f64() * 1_000.0,
                    tensor_construction_ms: tensor_construction.as_secs_f64() * 1_000.0,
                }),
            });
        }
    } else {
        // Model loading trials include session construction; runtime initialization is reported separately.
        let runtime = runtime.as_ref().expect("load mode initializes runtime");
        for _ in 0..options.trial_count {
            let start = Instant::now();
            let _loaded = ModelFactory.load(runtime, settings, model_type, Some(model_path))?;
            let elapsed = start.elapsed().as_secs_f64();
            let sample = elapsed * 1_000.0;
            all_samples.push(sample);
            trials.push(BenchmarkTrial {
                elapsed_seconds: elapsed,
                completed_windows: 0,
                throughput_per_second: None,
                samples_ms: vec![sample],
                statistics: calculate_latency_statistics(&[sample]),
                preprocessing_stages: None,
            });
        }
    }

    let resource_metrics = sampler.map(ResourceSampler::finish);
    let aggregate = calculate_latency_statistics(&all_samples);
    let elapsed_seconds = trials
        .iter()
        .map(|trial| trial.elapsed_seconds)
        .sum::<f64>();
    let completed = trials
        .iter()
        .map(|trial| trial.completed_windows)
        .sum::<usize>();
    let inference_time = Duration::from_secs_f64(elapsed_seconds);
    let average_latency = Duration::from_secs_f64(
        aggregate
            .as_ref()
            .map_or(0.0, |stats| stats.mean_ms / 1_000.0),
    );
    let throughput_per_second = calculate_throughput(completed, elapsed_seconds).unwrap_or(0.0);
    let report_path = options
        .report
        .map(|(path, format)| {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            let metadata = BenchmarkMetadata {
                model: model_type.to_string(),
                model_file: model_path.display().to_string(),
                model_file_hash: file_fingerprint(model_path),
                mode: benchmark_mode_name(options.mode).to_owned(),
                input: input_description.clone(),
                input_shape: format!("{:?}", settings.model_specs.input_shape()),
                input_type: "float32".to_owned(),
                execution_provider: settings.runtime.execution_provider.clone(),
                runtime_initialization_ms: runtime_init_time.as_secs_f64() * 1_000.0,
                initial_model_load_ms: model_load_time.as_secs_f64() * 1_000.0,
                warmup_iterations: if options.mode == BenchmarkMode::Load {
                    0
                } else {
                    options.warmup
                },
                measured_iterations,
                trial_count: options.trial_count,
                runtime_version: "ONNX Runtime 1.31; ort crate 2.0.0-rc.15".to_owned(),
                rust_version: command_version("rustc", &["--version"]),
                build_mode: if cfg!(debug_assertions) {
                    "debug"
                } else {
                    "release"
                }
                .to_owned(),
                operating_system: format!("{} {}", std::env::consts::OS, std::env::consts::ARCH),
                cpu: cpu_description(),
                ram_bytes: ram_bytes(),
                thread_configuration: "ONNX Runtime defaults; custom thread counts are not exposed"
                    .to_owned(),
                git_commit: command_version("git", &["rev-parse", "--short", "HEAD"]),
                timestamp_unix_seconds: now,
                cpu_utilization_percent: resource_metrics
                    .as_ref()
                    .and_then(|metrics| metrics.average_cpu_percent),
                cpu_peak_percent: resource_metrics
                    .as_ref()
                    .and_then(|metrics| metrics.peak_cpu_percent),
                memory: resource_metrics.clone(),
            };
            BenchmarkReport {
                metadata,
                trials: trials.clone(),
                aggregate: aggregate.clone(),
            }
            .write(path, format)?;
            Ok::<_, AppError>(path.to_owned())
        })
        .transpose()?;
    Ok(BenchmarkSummary {
        model: model_type,
        input: input_description,
        iterations: measured_iterations,
        windows_available,
        model_load_time,
        runtime_init_time,
        inference_time,
        average_latency,
        throughput_per_second,
        warmup_iterations: options.warmup,
        trials,
        aggregate,
        resource_metrics,
        report_path,
    })
}

fn no_windows_error() -> AppError {
    AppError::ConfigValidation("input dataset produced no complete 100-row windows".to_owned())
}

fn benchmark_mode_name(mode: BenchmarkMode) -> &'static str {
    match mode {
        BenchmarkMode::Inference => "inference",
        BenchmarkMode::Preprocessing => "preprocessing",
        BenchmarkMode::EndToEnd => "end-to-end",
        BenchmarkMode::Load => "load",
    }
}

fn command_version(program: &str, args: &[&str]) -> Option<String> {
    std::process::Command::new(program)
        .args(args)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        .filter(|value| !value.is_empty())
}

fn cpu_description() -> Option<String> {
    command_version("sysctl", &["-n", "machdep.cpu.brand_string"]).or_else(|| {
        std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|text| {
                text.lines()
                    .find(|line| line.starts_with("model name"))
                    .map(|line| {
                        line.split_once(':')
                            .map_or(line, |(_, value)| value.trim())
                            .to_owned()
                    })
            })
    })
}

fn file_fingerprint(path: &Path) -> Option<String> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).ok()?;
    let mut hash = 0xcbf29ce484222325_u64;
    let mut buffer = [0_u8; 16 * 1024];
    loop {
        let bytes_read = file.read(&mut buffer).ok()?;
        if bytes_read == 0 {
            break;
        }
        for byte in &buffer[..bytes_read] {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    Some(format!("fnv1a64:{hash:016x}"))
}

fn ram_bytes() -> Option<u64> {
    command_version("sysctl", &["-n", "hw.memsize"])
        .and_then(|value| value.parse().ok())
        .or_else(|| {
            std::fs::read_to_string("/proc/meminfo")
                .ok()
                .and_then(|text| {
                    text.lines()
                        .find(|line| line.starts_with("MemTotal:"))
                        .and_then(|line| line.split_whitespace().nth(1))
                        .and_then(|value| value.parse::<u64>().ok())
                        .map(|kib| kib.saturating_mul(1024))
                })
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

    use super::{BenchmarkOptions, run_benchmark, synthetic_input_tensor};
    use crate::cli::BenchmarkMode;

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
            run_benchmark(
                &settings,
                ModelType::DeepLob,
                None,
                BenchmarkOptions {
                    iterations: 0,
                    warmup: 20,
                    trial_count: 1,
                    mode: BenchmarkMode::Inference,
                    profile_memory: false,
                    profile_cpu: false,
                    report: None,
                }
            ),
            Err(AppError::ConfigValidation(_))
        ));
    }
}
