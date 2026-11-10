//! Reusable normalization and window preparation.

pub mod normalization;
mod pipeline;
pub mod window;

pub use normalization::{NormalizationStats, Normalizer};
pub use pipeline::{PreparedDataset, PreparedWindow, PreprocessingPipeline, PreprocessingTimings};
pub use window::{WindowConfig, WindowExtractor, WindowRef};
