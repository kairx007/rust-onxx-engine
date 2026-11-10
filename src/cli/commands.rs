use std::path::{Path, PathBuf};

use crate::{
    application::{
        inspect_model as inspect_model_service, run_benchmark, run_prediction, run_smoke_test,
        runtime_info,
    },
    cli::{BenchmarkFormat, Cli, Command, InputArgs},
    config::Settings,
    error::AppError,
    inference::Prediction,
    logging,
    models::ModelType,
    runtime::TensorMetadata,
};
use anyhow::Result;

impl Cli {
    pub fn execute(self) -> Result<()> {
        let mut settings = Settings::load(&self.config)?;
        if let Some(level) = &self.log_level {
            settings.logging.level = level.clone();
        }
        logging::init(&settings.logging.level)?;

        match self.command {
            Command::RuntimeInfo => show_runtime_info(&settings),
            Command::Inspect { model, model_path } => {
                inspect_model(&settings, model, model_path.as_deref(), false)
            }
            Command::Validate { model, model_path } => {
                inspect_model(&settings, model, model_path.as_deref(), true)
            }
            Command::SmokeTest {
                model,
                input,
                model_path,
            } => {
                let settings = apply_input_args(settings, &input)?;
                let result = run_smoke_test(
                    &settings,
                    model,
                    model_path.as_deref(),
                    input.input.is_some() || !input.segments.is_empty(),
                )?;
                println!("Smoke Test");
                println!("Model: {}", result.model);
                println!(
                    "Input: {}",
                    if result.synthetic_input {
                        "synthetic"
                    } else {
                        "dataset"
                    }
                );
                print_prediction(&result.prediction);
                println!("Smoke test completed successfully.");
                Ok(())
            }
            Command::Predict {
                model,
                input,
                output,
                overwrite,
                model_path,
            } => {
                let settings = apply_input_args(settings, &input)?;
                let output = output
                    .or_else(|| settings.data.output_path.clone())
                    .unwrap_or_else(|| PathBuf::from("outputs/predictions.csv"));
                if output.as_os_str().is_empty() {
                    return Err(AppError::ConfigValidation(
                        "prediction output path must not be empty".to_owned(),
                    )
                    .into());
                }
                let summary =
                    run_prediction(&settings, model, model_path.as_deref(), &output, overwrite)?;
                println!("Inference Summary");
                println!("Model: {}", summary.model);
                println!("Input: {}", summary.input);
                println!("Output: {}", summary.output_path.display());
                println!("Rows loaded: {}", summary.rows_loaded);
                println!("Windows generated: {}", summary.windows_generated);
                println!("Predictions completed: {}", summary.predictions_completed);
                println!("Failed predictions: 0");
                println!("Output written successfully.");
                Ok(())
            }
            Command::Benchmark {
                model,
                input,
                iterations,
                warmup,
                trials,
                mode,
                profile_memory,
                profile_cpu,
                output,
                format,
                model_path,
            } => {
                let settings = apply_input_args(settings, &input)?;
                let report_format = format
                    .map(|value| match value {
                        BenchmarkFormat::Json => crate::benchmark::ReportFormat::Json,
                        BenchmarkFormat::Csv => crate::benchmark::ReportFormat::Csv,
                    })
                    .or_else(|| {
                        output
                            .as_ref()
                            .and_then(|path| path.extension())
                            .and_then(|extension| extension.to_str())
                            .map(|extension| {
                                if extension.eq_ignore_ascii_case("csv") {
                                    crate::benchmark::ReportFormat::Csv
                                } else {
                                    crate::benchmark::ReportFormat::Json
                                }
                            })
                    });
                if format.is_some() && output.is_none() {
                    return Err(AppError::ConfigValidation(
                        "--format requires --output".to_owned(),
                    )
                    .into());
                }
                let report = output.as_deref().zip(report_format);
                let summary = run_benchmark(
                    &settings,
                    model,
                    model_path.as_deref(),
                    crate::application::BenchmarkOptions {
                        iterations,
                        warmup,
                        trial_count: trials,
                        mode,
                        profile_memory,
                        profile_cpu,
                        report,
                    },
                )?;
                println!("Benchmark Summary");
                println!("Model: {}", summary.model);
                println!("Mode: {:?}", mode);
                println!("Execution Provider: CPU");
                println!("Input: {}", summary.input);
                println!("Windows available: {}", summary.windows_available);
                println!("Iterations: {}", summary.iterations);
                println!("Warm-up iterations: {}", summary.warmup_iterations);
                println!("Trials: {}", summary.trials.len());
                println!(
                    "Runtime initialization time: {:.6}s",
                    summary.runtime_init_time.as_secs_f64()
                );
                println!(
                    "Model loading time: {:.6}s",
                    summary.model_load_time.as_secs_f64()
                );
                println!(
                    "Measured elapsed time: {:.6}s",
                    summary.inference_time.as_secs_f64()
                );
                println!(
                    "Average latency: {:.6}ms",
                    summary.average_latency.as_secs_f64() * 1_000.0
                );
                println!("Throughput: {:.2} windows/s", summary.throughput_per_second);
                if let Some(stats) = &summary.aggregate {
                    println!("Latency samples: {}", stats.sample_count);
                    println!("Mean latency: {:.6}ms", stats.mean_ms);
                    println!("Std deviation: {:.6}ms", stats.standard_deviation_ms);
                    println!(
                        "P50/P90/P95/P99: {:.6}/{:.6}/{:.6}/{:.6}ms",
                        stats.p50_ms, stats.p90_ms, stats.p95_ms, stats.p99_ms
                    );
                    println!("Min/Max: {:.6}/{:.6}ms", stats.minimum_ms, stats.maximum_ms);
                }
                if let Some(resources) = &summary.resource_metrics {
                    println!(
                        "Process RSS baseline/average/peak: {:?}/{:?}/{:?} bytes",
                        resources.baseline_rss_bytes,
                        resources.average_rss_bytes,
                        resources.peak_rss_bytes
                    );
                    println!(
                        "CPU average/peak: {:?}/{:?}%",
                        resources.average_cpu_percent, resources.peak_cpu_percent
                    );
                    println!("Resource method: {}", resources.method);
                }
                if let Some(stages) = summary
                    .trials
                    .last()
                    .and_then(|trial| trial.preprocessing_stages.as_ref())
                {
                    println!(
                        "Preprocessing stages, last trial: CSV {:.3}ms, normalization {:.3}ms, windows {:.3}ms, tensors {:.3}ms",
                        stages.csv_loading_ms,
                        stages.normalization_ms,
                        stages.window_generation_ms,
                        stages.tensor_construction_ms
                    );
                }
                if let Some(path) = &summary.report_path {
                    println!("Report: {}", path.display());
                }
                Ok(())
            }
            Command::Compare {
                model,
                rust_results,
                cpp_results,
                output,
            } => {
                let comparison =
                    crate::benchmark::compare_json_reports(model, &rust_results, &cpp_results)?;
                println!("Rust vs C++ comparison ({})", comparison.model);
                println!(
                    "Mean latency, Rust/C++: {:.6}/{:.6}ms",
                    comparison.rust_mean_latency_ms, comparison.cpp_mean_latency_ms
                );
                println!(
                    "P50 latency, Rust/C++: {:.6}/{:.6}ms",
                    comparison.rust_p50_latency_ms, comparison.cpp_p50_latency_ms
                );
                println!(
                    "P95 latency, Rust/C++: {:.6}/{:.6}ms",
                    comparison.rust_p95_latency_ms, comparison.cpp_p95_latency_ms
                );
                println!(
                    "P99 latency, Rust/C++: {:.6}/{:.6}ms",
                    comparison.rust_p99_latency_ms, comparison.cpp_p99_latency_ms
                );
                println!(
                    "Latency improvement: {:.2}%",
                    comparison.latency_improvement_percent
                );
                println!(
                    "Throughput, Rust/C++: {:.2}/{:.2} windows/s",
                    comparison.rust_throughput_per_second, comparison.cpp_throughput_per_second
                );
                println!(
                    "Throughput ratio (Rust/C++): {:.3}x",
                    comparison.throughput_ratio
                );
                if let Some(path) = output {
                    let file = std::fs::OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .open(&path)
                        .map_err(|source| AppError::BenchmarkReport {
                            path: path.clone(),
                            source,
                        })?;
                    serde_json::to_writer_pretty(file, &comparison)
                        .map_err(|error| AppError::BenchmarkSerialization(error.to_string()))?;
                    println!("Comparison report: {}", path.display());
                }
                Ok(())
            }
        }
    }
}

fn apply_input_args(mut settings: Settings, args: &InputArgs) -> Result<Settings> {
    if let Some(input) = &args.input {
        settings.data.dataset_path = input.clone();
        settings.data.segment_paths.clear();
    } else if !args.segments.is_empty() {
        settings.data.segment_paths = args.segments.clone();
    }
    if let Some(mean_path) = &args.mean_path {
        settings.normalization.mean_path = mean_path.clone();
    }
    if let Some(std_path) = &args.std_path {
        settings.normalization.std_path = std_path.clone();
    }
    settings.validate()?;
    Ok(settings)
}

fn show_runtime_info(settings: &Settings) -> Result<()> {
    let summary = runtime_info(settings)?;
    println!("LOB Inference Runtime");
    println!("Application: {}", summary.application_name);
    println!("Version: {}", summary.application_version);
    println!("ONNX Runtime: initialized");
    println!("ONNX Runtime API target: 1.{}", summary.ort_api_target);
    println!("Execution Provider: {}", summary.execution_provider);
    println!("Runtime configuration: CPU provider (thread counts use ONNX Runtime defaults)");
    for model in summary.models {
        println!(
            "{}: configured at {} (file {})",
            model.model,
            model.path.display(),
            if model.file_available {
                "available"
            } else {
                "missing"
            }
        );
    }
    Ok(())
}

fn inspect_model(
    settings: &Settings,
    model: ModelType,
    path_override: Option<&Path>,
    strict_validation: bool,
) -> Result<()> {
    let inspection = inspect_model_service(settings, model, path_override)?;
    println!("Model: {}", inspection.model);
    println!("Path: {}", inspection.path.display());
    if !inspection.file_available() {
        println!("File available: no");
        println!("Validation: failed (model file is missing)");
        return Err(inspection
            .into_error()
            .expect("missing model has an error")
            .into());
    }
    println!("File available: yes");
    println!(
        "Expected class count: {}",
        settings.model_specs.output_classes
    );
    if let Some(metadata) = &inspection.metadata {
        print_metadata("Input", &metadata.inputs);
        print_metadata("Output", &metadata.outputs);
    }
    if let Some(error) = &inspection.validation_error {
        println!("Metadata validation: failed ({error})");
        if strict_validation {
            return Err(inspection
                .into_error()
                .expect("invalid metadata has an error")
                .into());
        }
    } else {
        println!("Metadata validation: passed");
    }
    Ok(())
}

fn print_metadata(label: &str, tensors: &[TensorMetadata]) {
    if tensors.is_empty() {
        println!("{label}s: none");
        return;
    }
    for tensor in tensors {
        let dimensions = tensor
            .dimensions
            .as_ref()
            .map_or_else(|| "non-tensor".to_owned(), |shape| format!("{shape:?}"));
        println!(
            "{label}: {} | type={} | shape={dimensions}",
            tensor.name, tensor.element_type
        );
    }
}

fn print_prediction(prediction: &Prediction) {
    println!("Prediction:");
    println!("  Class: {}", prediction.class_index());
    println!("  Direction: {:?}", prediction.direction());
    let scores = prediction.scores();
    println!(
        "  Scores: Down={} Stationary={} Up={}",
        scores[0], scores[1], scores[2]
    );
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::config::Settings;

    use super::apply_input_args;
    use crate::cli::args::InputArgs;

    #[test]
    fn cli_paths_override_config_and_explicit_segments_keep_order() {
        let settings = Settings::parse(include_str!("../../configs/default.toml")).unwrap();
        let overridden = apply_input_args(
            settings.clone(),
            &InputArgs {
                input: Some(PathBuf::from("custom.csv")),
                segments: vec![],
                mean_path: Some(PathBuf::from("mean.npy")),
                std_path: None,
            },
        )
        .unwrap();
        assert_eq!(overridden.data.dataset_path, PathBuf::from("custom.csv"));
        assert!(overridden.data.segment_paths.is_empty());
        assert_eq!(
            overridden.normalization.mean_path,
            PathBuf::from("mean.npy")
        );

        let segmented = apply_input_args(
            settings,
            &InputArgs {
                input: None,
                segments: vec![PathBuf::from("day-2.csv"), PathBuf::from("day-1.csv")],
                mean_path: None,
                std_path: None,
            },
        )
        .unwrap();
        assert_eq!(
            segmented.data.segment_paths,
            [PathBuf::from("day-2.csv"), PathBuf::from("day-1.csv")]
        );
    }
}
