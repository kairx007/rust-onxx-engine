use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{error::AppError, models::ModelType};

use super::{BenchmarkReport, calculate_throughput};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComparisonReport {
    pub model: String,
    pub rust_mean_latency_ms: f64,
    pub cpp_mean_latency_ms: f64,
    pub rust_p50_latency_ms: f64,
    pub cpp_p50_latency_ms: f64,
    pub rust_p95_latency_ms: f64,
    pub cpp_p95_latency_ms: f64,
    pub rust_p99_latency_ms: f64,
    pub cpp_p99_latency_ms: f64,
    pub latency_improvement_percent: f64,
    pub rust_throughput_per_second: f64,
    pub cpp_throughput_per_second: f64,
    pub throughput_ratio: f64,
}

pub fn compare_json_reports(
    model: ModelType,
    rust_path: &Path,
    cpp_path: &Path,
) -> Result<ComparisonReport, AppError> {
    let rust = BenchmarkReport::read_json(rust_path)?;
    let cpp = BenchmarkReport::read_json(cpp_path)?;
    if !rust.metadata.model.eq_ignore_ascii_case(&model.to_string())
        || !cpp.metadata.model.eq_ignore_ascii_case(&model.to_string())
        || rust.metadata.mode != "inference"
        || cpp.metadata.mode != "inference"
        || rust.metadata.execution_provider != cpp.metadata.execution_provider
        || rust.metadata.input_shape != cpp.metadata.input_shape
        || rust.metadata.input_type != cpp.metadata.input_type
        || rust.metadata.model_file_hash.is_none()
        || rust.metadata.model_file_hash != cpp.metadata.model_file_hash
        || rust.metadata.runtime_version != cpp.metadata.runtime_version
    {
        return Err(AppError::ConfigValidation("comparison requires reports for the requested model with matching inference mode, provider, input shape, and input type".to_owned()));
    }
    let rust_mean = rust
        .aggregate
        .as_ref()
        .map(|stats| stats.mean_ms)
        .ok_or_else(|| {
            AppError::ConfigValidation("Rust report has no valid latency samples".to_owned())
        })?;
    let cpp_mean = cpp
        .aggregate
        .as_ref()
        .map(|stats| stats.mean_ms)
        .ok_or_else(|| {
            AppError::ConfigValidation("C++ report has no valid latency samples".to_owned())
        })?;
    let rust_throughput = total_throughput(&rust)?;
    let cpp_throughput = total_throughput(&cpp)?;
    if cpp_mean == 0.0 || cpp_throughput == 0.0 {
        return Err(AppError::ConfigValidation(
            "comparison contains a zero denominator".to_owned(),
        ));
    }
    Ok(ComparisonReport {
        model: model.to_string(),
        rust_mean_latency_ms: rust_mean,
        cpp_mean_latency_ms: cpp_mean,
        rust_p50_latency_ms: rust
            .aggregate
            .as_ref()
            .expect("validated Rust latency summary")
            .p50_ms,
        cpp_p50_latency_ms: cpp
            .aggregate
            .as_ref()
            .expect("validated C++ latency summary")
            .p50_ms,
        rust_p95_latency_ms: rust
            .aggregate
            .as_ref()
            .expect("validated Rust latency summary")
            .p95_ms,
        cpp_p95_latency_ms: cpp
            .aggregate
            .as_ref()
            .expect("validated C++ latency summary")
            .p95_ms,
        rust_p99_latency_ms: rust
            .aggregate
            .as_ref()
            .expect("validated Rust latency summary")
            .p99_ms,
        cpp_p99_latency_ms: cpp
            .aggregate
            .as_ref()
            .expect("validated C++ latency summary")
            .p99_ms,
        latency_improvement_percent: (cpp_mean - rust_mean) / cpp_mean * 100.0,
        rust_throughput_per_second: rust_throughput,
        cpp_throughput_per_second: cpp_throughput,
        throughput_ratio: rust_throughput / cpp_throughput,
    })
}

fn total_throughput(report: &BenchmarkReport) -> Result<f64, AppError> {
    let windows = report
        .trials
        .iter()
        .map(|trial| trial.completed_windows)
        .sum();
    let elapsed = report
        .trials
        .iter()
        .map(|trial| trial.elapsed_seconds)
        .sum();
    calculate_throughput(windows, elapsed).ok_or_else(|| {
        AppError::ConfigValidation("report has no valid throughput samples".to_owned())
    })
}
