//! FI-2010 data loading, feature extraction, and label handling.

pub mod csv_loader;
mod dataset;
pub mod feature_extractor;
pub mod labels;

pub use csv_loader::CsvLoader;
pub use dataset::{FEATURE_COUNT, LobDataset, LobSegment};
