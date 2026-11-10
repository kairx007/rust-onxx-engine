use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};

use crate::models::ModelType;

#[derive(Debug, Parser)]
#[command(
    name = "lob-inference",
    version,
    about = "Run DeepLOB and TransformerLOB inference with ONNX Runtime"
)]
pub struct Cli {
    #[arg(long, global = true, default_value = "configs/default.toml")]
    pub config: PathBuf,
    #[arg(long, global = true, value_parser = ["error", "warn", "info", "debug", "trace"])]
    pub log_level: Option<String>,
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Initialize and display the configured ONNX Runtime environment.
    RuntimeInfo,
    /// Load a model and display its ONNX metadata.
    Inspect {
        #[arg(long, value_enum)]
        model: ModelType,
        /// Override the configured ONNX model path for this command.
        #[arg(long)]
        model_path: Option<PathBuf>,
    },
    /// Load a model and validate its model-specific inference contract.
    Validate {
        #[arg(long, value_enum)]
        model: ModelType,
        /// Override the configured ONNX model path for this command.
        #[arg(long)]
        model_path: Option<PathBuf>,
    },
    /// Run one prediction using a dataset window or a synthetic smoke tensor.
    SmokeTest {
        #[arg(long, value_enum)]
        model: ModelType,
        #[command(flatten)]
        input: InputArgs,
        /// Override the configured ONNX model path for this command.
        #[arg(long)]
        model_path: Option<PathBuf>,
    },
    /// Run inference over all windows produced from the configured input data.
    Predict {
        #[arg(long, value_enum)]
        model: ModelType,
        #[command(flatten)]
        input: InputArgs,
        /// Write predictions to this CSV instead of `data.output_path`.
        #[arg(long)]
        output: Option<PathBuf>,
        /// Replace an existing output file. The default is to preserve it and fail.
        #[arg(long)]
        overwrite: bool,
        /// Override the configured ONNX model path for this command.
        #[arg(long)]
        model_path: Option<PathBuf>,
    },
    /// Run a reproducible inference, preprocessing, load, or end-to-end benchmark.
    Benchmark {
        #[arg(long, value_enum)]
        model: ModelType,
        #[command(flatten)]
        input: InputArgs,
        /// Number of inference calls per trial.
        #[arg(long, default_value_t = 200, value_parser = parse_positive_usize)]
        iterations: usize,
        /// Number of warm-up inference calls, excluded from measurements.
        #[arg(long, default_value_t = 20)]
        warmup: usize,
        /// Number of independent measurement trials.
        #[arg(long, default_value_t = 5, value_parser = parse_positive_usize)]
        trials: usize,
        /// Measurement path.
        #[arg(long, value_enum, default_value_t = BenchmarkMode::Inference)]
        mode: BenchmarkMode,
        /// Sample process RSS concurrently with the benchmark.
        #[arg(long)]
        profile_memory: bool,
        /// Sample CPU utilization concurrently (when supported).
        #[arg(long)]
        profile_cpu: bool,
        /// Write a machine-readable report. Existing files are never replaced.
        #[arg(long)]
        output: Option<PathBuf>,
        /// Report file format; defaults to the output file extension.
        #[arg(long, value_enum)]
        format: Option<BenchmarkFormat>,
        /// Override the configured ONNX model path for this command.
        #[arg(long)]
        model_path: Option<PathBuf>,
    },
    /// Compare two compatible JSON benchmark reports.
    Compare {
        #[arg(long, value_enum)]
        model: ModelType,
        #[arg(long)]
        rust_results: PathBuf,
        #[arg(long)]
        cpp_results: PathBuf,
        /// Optional JSON output path; existing files are never replaced.
        #[arg(long)]
        output: Option<PathBuf>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum BenchmarkMode {
    Inference,
    Preprocessing,
    EndToEnd,
    Load,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum BenchmarkFormat {
    Json,
    Csv,
}

fn parse_positive_usize(value: &str) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|error| format!("invalid iteration count '{value}': {error}"))?;
    if parsed == 0 {
        return Err("iterations must be greater than zero".to_owned());
    }
    Ok(parsed)
}

#[derive(Debug, Args, Clone, Default)]
pub struct InputArgs {
    /// Override the configured dataset path with one CSV file or directory.
    #[arg(long, conflicts_with = "segments")]
    pub input: Option<PathBuf>,
    /// Supply explicit chronological CSV segment paths (repeat the option).
    #[arg(long = "segment", num_args = 1, conflicts_with = "input")]
    pub segments: Vec<PathBuf>,
    /// Override the training mean statistics NPY path.
    #[arg(long)]
    pub mean_path: Option<PathBuf>,
    /// Override the training standard-deviation statistics NPY path.
    #[arg(long)]
    pub std_path: Option<PathBuf>,
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::{Cli, Command};

    #[test]
    fn parses_existing_diagnostics_and_new_workflows() {
        assert!(matches!(
            Cli::try_parse_from(["lob-inference", "runtime-info"])
                .unwrap()
                .command,
            Command::RuntimeInfo
        ));
        assert!(Cli::try_parse_from(["lob-inference", "inspect", "--model", "deeplob"]).is_ok());
        assert!(
            Cli::try_parse_from(["lob-inference", "validate", "--model", "transformerlob"]).is_ok()
        );
        assert!(Cli::try_parse_from(["lob-inference", "smoke-test", "--model", "deeplob"]).is_ok());
        assert!(
            Cli::try_parse_from([
                "lob-inference",
                "predict",
                "--model",
                "deeplob",
                "--output",
                "out.csv"
            ])
            .is_ok()
        );
        assert!(
            Cli::try_parse_from([
                "lob-inference",
                "benchmark",
                "--model",
                "deeplob",
                "--warmup",
                "10",
                "--iterations",
                "50",
                "--trials",
                "2",
                "--mode",
                "inference",
                "--profile-memory",
                "--output",
                "run.json"
            ])
            .is_ok()
        );
        assert!(
            Cli::try_parse_from([
                "lob-inference",
                "compare",
                "--model",
                "deeplob",
                "--rust-results",
                "rust.json",
                "--cpp-results",
                "cpp.json"
            ])
            .is_ok()
        );
    }

    #[test]
    fn requires_models_and_valid_benchmark_iterations() {
        assert!(Cli::try_parse_from(["lob-inference", "predict"]).is_err());
        assert!(Cli::try_parse_from(["lob-inference", "predict", "--model", "unknown"]).is_err());
        assert!(
            Cli::try_parse_from([
                "lob-inference",
                "benchmark",
                "--model",
                "deeplob",
                "--iterations",
                "0"
            ])
            .is_err()
        );
        assert!(
            Cli::try_parse_from([
                "lob-inference",
                "predict",
                "--model",
                "deeplob",
                "--input",
                "a.csv",
                "--segment",
                "b.csv"
            ])
            .is_err()
        );
    }
}
