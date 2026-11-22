use std::path::Path;

use crate::{config::DataSettings, error::AppError};

/// Maps FI-2010 labels 1 (Up), 2 (Stationary), 3 (Down) to class indices 0, 1, 2.
pub fn encode_label(value: &str, path: &Path, row: usize) -> Result<u8, AppError> {
    let numeric = value.trim().parse::<f64>().ok();
    match numeric {
        Some(1.0) => Ok(0),
        Some(2.0) => Ok(1),
        Some(3.0) => Ok(2),
        _ => Err(AppError::CsvLabel {
            path: path.to_owned(),
            row,
            value: value.to_owned(),
        }),
    }
}

pub fn decode_label(label: u8) -> Option<&'static str> {
    match label {
        0 => Some("Up"),
        1 => Some("Stationary"),
        2 => Some("Down"),
        _ => None,
    }
}

pub fn extract_label(
    record: &csv::StringRecord,
    settings: &DataSettings,
    path: &Path,
    row: usize,
) -> Result<u8, AppError> {
    let value = record
        .get(settings.label_column)
        .ok_or_else(|| AppError::CsvColumnCount {
            path: path.to_owned(),
            row,
            expected: settings.label_column + 1,
            found: record.len(),
        })?;
    encode_label(value, path, row)
}
