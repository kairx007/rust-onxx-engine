use std::path::Path;

use crate::{config::DataSettings, error::AppError};

pub fn encode_label(value: &str, path: &Path, row: usize) -> Result<u8, AppError> {
    match value.trim().parse::<u8>() {
        Ok(1) => Ok(0),
        Ok(2) => Ok(1),
        Ok(3) => Ok(2),
        _ => Err(AppError::CsvLabel {
            path: path.to_owned(),
            row,
            value: value.to_owned(),
        }),
    }
}

pub fn decode_label(label: u8) -> Option<&'static str> {
    match label {
        0 => Some("Down"),
        1 => Some("Stationary"),
        2 => Some("Up"),
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
