use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};

use crate::{
    config::Settings,
    error::AppError,
    logging,
    models::{ModelRegistry, ModelType},
    runtime::{ModelSession, Runtime, TensorMetadata},
};

#[derive(Debug, Parser)]
#[command(
    name = "rust-engine",
    version,
    about = "RustLOB ONNX Runtime diagnostics"
)]
pub struct Cli {
    #[arg(long, global = true, default_value = "configs/default.toml")]
    config: PathBuf,
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Initialize and display the configured ONNX Runtime environment.
    RuntimeInfo,
    /// Load a model and display its ONNX input/output metadata.
    Inspect {
        #[arg(long, value_enum)]
        model: ModelType,
    },
    /// Load a model and validate its metadata against the configured model spec.
    Validate {
        #[arg(long, value_enum)]
        model: ModelType,
    },
}

impl Cli {
    pub fn execute(self) -> Result<()> {
        let settings = Settings::load(&self.config)?;
        logging::init(&settings.logging.level)?;

        match self.command {
            Command::RuntimeInfo => show_runtime_info(&settings),
            Command::Inspect { model } => inspect_model(&settings, model),
            Command::Validate { model } => inspect_model(&settings, model),
        }
    }
}

fn show_runtime_info(settings: &Settings) -> Result<()> {
    let runtime = Runtime::initialize(&settings.runtime.execution_provider)?;
    println!("RustLOB version: {}", env!("CARGO_PKG_VERSION"));
    println!("ONNX Runtime: initialized");
    println!("Execution provider: {}", runtime.execution_provider());
    println!("Input shape: {:?}", settings.model_specs.input_shape());
    println!("Output classes: {}", settings.model_specs.output_classes);
    println!("DeepLOB model: {}", settings.models.deeplob_path.display());
    println!(
        "TransformerLOB model: {}",
        settings.models.transformerlob_path.display()
    );
    Ok(())
}

fn inspect_model(settings: &Settings, model: ModelType) -> Result<()> {
    let registry = ModelRegistry;
    let path = registry.path_for(settings, model);
    println!("Model: {model}");
    println!("Path: {}", path.display());

    if !path.is_file() {
        println!("File available: no");
        println!("Validation: failed (model file is missing)");
        return Err(AppError::MissingModel { model, path }.into());
    }
    println!("File available: yes");

    let runtime = Runtime::initialize(&settings.runtime.execution_provider)?;
    let loaded = ModelSession::load(&runtime, model, &path)?;
    print_metadata("Input", &loaded.metadata().inputs);
    print_metadata("Output", &loaded.metadata().outputs);

    match loaded.validate(&settings.model_specs) {
        Ok(()) => {
            println!("Validation: compatible");
            Ok(())
        }
        Err(error) => {
            println!("Validation: incompatible ({error})");
            Err(error.into())
        }
    }
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
