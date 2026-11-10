use std::{fs::OpenOptions, path::Path};

use serde::{Deserialize, Serialize};

use crate::{benchmark::LatencyStatistics, error::AppError};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ReportFormat {
    Json,
    Csv,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkTrial {
    pub elapsed_seconds: f64,
    pub completed_windows: usize,
    pub throughput_per_second: Option<f64>,
    pub samples_ms: Vec<f64>,
    pub statistics: Option<LatencyStatistics>,
    #[serde(default)]
    pub preprocessing_stages: Option<PreprocessingStageMetrics>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PreprocessingStageMetrics {
    pub csv_loading_ms: f64,
    pub normalization_ms: f64,
    pub window_generation_ms: f64,
    pub tensor_construction_ms: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkMetadata {
    pub model: String,
    pub model_file: String,
    pub model_file_hash: Option<String>,
    pub mode: String,
    pub input: String,
    pub input_shape: String,
    pub input_type: String,
    pub execution_provider: String,
    pub runtime_initialization_ms: f64,
    pub initial_model_load_ms: f64,
    pub warmup_iterations: usize,
    pub measured_iterations: usize,
    pub trial_count: usize,
    pub runtime_version: String,
    pub rust_version: Option<String>,
    pub build_mode: String,
    pub operating_system: String,
    pub cpu: Option<String>,
    pub ram_bytes: Option<u64>,
    pub thread_configuration: String,
    pub git_commit: Option<String>,
    pub timestamp_unix_seconds: u64,
    pub memory: Option<super::ResourceMetrics>,
    pub cpu_utilization_percent: Option<f64>,
    pub cpu_peak_percent: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub metadata: BenchmarkMetadata,
    pub trials: Vec<BenchmarkTrial>,
    pub aggregate: Option<LatencyStatistics>,
}

impl BenchmarkReport {
    pub fn read_json(path: &Path) -> Result<Self, AppError> {
        let file = std::fs::File::open(path).map_err(|source| AppError::BenchmarkReport {
            path: path.to_owned(),
            source,
        })?;
        serde_json::from_reader(file)
            .map_err(|error| AppError::BenchmarkSerialization(error.to_string()))
    }

    /// Create a new file without replacing prior measurements.
    pub fn write(&self, path: &Path, format: ReportFormat) -> Result<(), AppError> {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent).map_err(|source| AppError::BenchmarkReport {
                path: parent.to_owned(),
                source,
            })?;
        }
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|source| AppError::BenchmarkReport {
                path: path.to_owned(),
                source,
            })?;
        match format {
            ReportFormat::Json => serde_json::to_writer_pretty(file, self)
                .map_err(|error| AppError::BenchmarkSerialization(error.to_string())),
            ReportFormat::Csv => self.write_csv(file),
        }
    }

    fn write_csv(&self, file: std::fs::File) -> Result<(), AppError> {
        let mut writer = csv::Writer::from_writer(file);
        writer
            .write_record([
                "trial",
                "sample",
                "latency_ms",
                "elapsed_seconds",
                "completed_windows",
                "throughput_per_second",
                "model",
                "mode",
                "input",
                "execution_provider",
                "model_file",
                "model_file_hash",
                "input_shape",
                "input_type",
                "runtime_version",
                "rust_version",
                "build_mode",
                "operating_system",
                "cpu",
                "ram_bytes",
                "thread_configuration",
                "git_commit",
                "timestamp_unix_seconds",
                "warmup_iterations",
                "measured_iterations",
                "trial_count",
                "runtime_initialization_ms",
                "initial_model_load_ms",
                "average_rss_bytes",
                "peak_rss_bytes",
                "average_cpu_percent",
                "peak_cpu_percent",
                "mean_ms",
                "standard_deviation_ms",
                "p50_ms",
                "p90_ms",
                "p95_ms",
                "p99_ms",
                "minimum_ms",
                "maximum_ms",
                "csv_loading_ms",
                "normalization_ms",
                "window_generation_ms",
                "tensor_construction_ms",
            ])
            .map_err(|error| AppError::BenchmarkSerialization(error.to_string()))?;
        for (trial_index, trial) in self.trials.iter().enumerate() {
            for (sample_index, sample) in trial.samples_ms.iter().enumerate() {
                writer
                    .write_record([
                        trial_index.to_string(),
                        sample_index.to_string(),
                        sample.to_string(),
                        trial.elapsed_seconds.to_string(),
                        trial.completed_windows.to_string(),
                        trial
                            .throughput_per_second
                            .map_or_else(String::new, |v| v.to_string()),
                        self.metadata.model.clone(),
                        self.metadata.mode.clone(),
                        self.metadata.input.clone(),
                        self.metadata.execution_provider.clone(),
                        self.metadata.model_file.clone(),
                        self.metadata.model_file_hash.clone().unwrap_or_default(),
                        self.metadata.input_shape.clone(),
                        self.metadata.input_type.clone(),
                        self.metadata.runtime_version.clone(),
                        self.metadata.rust_version.clone().unwrap_or_default(),
                        self.metadata.build_mode.clone(),
                        self.metadata.operating_system.clone(),
                        self.metadata.cpu.clone().unwrap_or_default(),
                        self.metadata
                            .ram_bytes
                            .map_or_else(String::new, |v| v.to_string()),
                        self.metadata.thread_configuration.clone(),
                        self.metadata.git_commit.clone().unwrap_or_default(),
                        self.metadata.timestamp_unix_seconds.to_string(),
                        self.metadata.warmup_iterations.to_string(),
                        self.metadata.measured_iterations.to_string(),
                        self.metadata.trial_count.to_string(),
                        self.metadata.runtime_initialization_ms.to_string(),
                        self.metadata.initial_model_load_ms.to_string(),
                        self.metadata
                            .memory
                            .as_ref()
                            .and_then(|m| m.average_rss_bytes)
                            .map_or_else(String::new, |v| v.to_string()),
                        self.metadata
                            .memory
                            .as_ref()
                            .and_then(|m| m.peak_rss_bytes)
                            .map_or_else(String::new, |v| v.to_string()),
                        self.metadata
                            .cpu_utilization_percent
                            .map_or_else(String::new, |v| v.to_string()),
                        self.metadata
                            .cpu_peak_percent
                            .map_or_else(String::new, |v| v.to_string()),
                        trial
                            .statistics
                            .as_ref()
                            .map_or_else(String::new, |stats| stats.mean_ms.to_string()),
                        trial.statistics.as_ref().map_or_else(String::new, |stats| {
                            stats.standard_deviation_ms.to_string()
                        }),
                        trial
                            .statistics
                            .as_ref()
                            .map_or_else(String::new, |stats| stats.p50_ms.to_string()),
                        trial
                            .statistics
                            .as_ref()
                            .map_or_else(String::new, |stats| stats.p90_ms.to_string()),
                        trial
                            .statistics
                            .as_ref()
                            .map_or_else(String::new, |stats| stats.p95_ms.to_string()),
                        trial
                            .statistics
                            .as_ref()
                            .map_or_else(String::new, |stats| stats.p99_ms.to_string()),
                        trial
                            .statistics
                            .as_ref()
                            .map_or_else(String::new, |stats| stats.minimum_ms.to_string()),
                        trial
                            .statistics
                            .as_ref()
                            .map_or_else(String::new, |stats| stats.maximum_ms.to_string()),
                        trial
                            .preprocessing_stages
                            .as_ref()
                            .map_or_else(String::new, |stages| stages.csv_loading_ms.to_string()),
                        trial
                            .preprocessing_stages
                            .as_ref()
                            .map_or_else(String::new, |stages| stages.normalization_ms.to_string()),
                        trial
                            .preprocessing_stages
                            .as_ref()
                            .map_or_else(String::new, |stages| {
                                stages.window_generation_ms.to_string()
                            }),
                        trial
                            .preprocessing_stages
                            .as_ref()
                            .map_or_else(String::new, |stages| {
                                stages.tensor_construction_ms.to_string()
                            }),
                    ])
                    .map_err(|error| AppError::BenchmarkSerialization(error.to_string()))?;
            }
        }
        writer.flush().map_err(|source| AppError::BenchmarkReport {
            path: Path::new("<report>").to_owned(),
            source,
        })?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::PathBuf,
        sync::atomic::{AtomicUsize, Ordering},
    };

    use super::{BenchmarkMetadata, BenchmarkReport, BenchmarkTrial, ReportFormat};
    use crate::benchmark::calculate_latency_statistics;

    static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

    fn path(extension: &str) -> PathBuf {
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "rust-engine-benchmark-{}-{id}.{extension}",
            std::process::id()
        ))
    }

    fn report() -> BenchmarkReport {
        let samples = vec![1.0, 2.0, 3.0];
        BenchmarkReport {
            metadata: BenchmarkMetadata {
                model: "DeepLOB".to_owned(),
                model_file: "model.onnx".to_owned(),
                model_file_hash: Some("fnv1a64:1234".to_owned()),
                mode: "inference".to_owned(),
                input: "input.csv".to_owned(),
                input_shape: "[1, 1, 100, 40]".to_owned(),
                input_type: "float32".to_owned(),
                execution_provider: "cpu".to_owned(),
                runtime_initialization_ms: 0.0,
                initial_model_load_ms: 0.0,
                warmup_iterations: 20,
                measured_iterations: 3,
                trial_count: 1,
                runtime_version: "test runtime".to_owned(),
                rust_version: Some("rustc test".to_owned()),
                build_mode: "release".to_owned(),
                operating_system: "test".to_owned(),
                cpu: None,
                ram_bytes: None,
                thread_configuration: "defaults".to_owned(),
                git_commit: None,
                timestamp_unix_seconds: 1,
                memory: None,
                cpu_utilization_percent: None,
                cpu_peak_percent: None,
            },
            trials: vec![BenchmarkTrial {
                elapsed_seconds: 0.006,
                completed_windows: 3,
                throughput_per_second: Some(500.0),
                statistics: calculate_latency_statistics(&samples),
                samples_ms: samples,
                preprocessing_stages: None,
            }],
            aggregate: calculate_latency_statistics(&[1.0, 2.0, 3.0]),
        }
    }

    #[test]
    fn json_report_round_trips_and_does_not_overwrite_existing_results() {
        let report = report();
        let path = path("json");
        report.write(&path, ReportFormat::Json).unwrap();
        assert_eq!(BenchmarkReport::read_json(&path).unwrap(), report);
        assert!(report.write(&path, ReportFormat::Json).is_err());
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn csv_report_preserves_raw_samples_and_summary_columns() {
        let report = report();
        let path = path("csv");
        report.write(&path, ReportFormat::Csv).unwrap();
        let csv = fs::read_to_string(&path).unwrap();
        assert!(csv.contains("latency_ms"));
        assert!(csv.contains("mean_ms"));
        assert!(csv.lines().count() == 4);
        assert!(csv.contains(",1,"));
        assert!(csv.contains(",2,"));
        assert!(csv.contains(",3,"));
        fs::remove_file(path).unwrap();
    }
}
