use csv::StringRecord;
use std::path::Path;

use crate::{config::DataSettings, error::AppError};

pub fn extract_features(
    record: &StringRecord,
    settings: &DataSettings,
    path: &Path,
    row: usize,
) -> Result<[f32; 40], AppError> {
    let mut features = [0.0_f32; 40];
    for (feature_index, &column) in settings.feature_columns.iter().enumerate() {
        let value = record.get(column).ok_or_else(|| AppError::CsvColumnCount {
            path: path.to_owned(),
            row,
            expected: column + 1,
            found: record.len(),
        })?;
        let parsed = value
            .parse::<f32>()
            .map_err(|_| AppError::CsvNumericValue {
                path: path.to_owned(),
                row,
                column,
                value: value.to_owned(),
            })?;
        if !parsed.is_finite() {
            return Err(AppError::CsvNumericValue {
                path: path.to_owned(),
                row,
                column,
                value: value.to_owned(),
            });
        }
        features[feature_index] = parsed;
    }
    Ok(features)
}
