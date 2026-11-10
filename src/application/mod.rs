//! Application workflows used by the command-line interface.

mod diagnostics;
mod inference_service;
mod model_factory;
mod output_writer;

pub use diagnostics::{
    ConfiguredModelStatus, ModelInspection, RuntimeSummary, inspect_model, runtime_info,
};
pub use inference_service::{
    BenchmarkOptions, BenchmarkSummary, PredictionSummary, SmokeTestResult, run_benchmark,
    run_prediction, run_smoke_test,
};
pub use model_factory::{LoadedModel, ModelFactory};
