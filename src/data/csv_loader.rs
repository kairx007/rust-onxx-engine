use std::{
    fs,
    path::{Path, PathBuf},
};

use crate::{
    config::DataSettings,
    data::{LobDataset, LobSegment},
    error::AppError,
};

use super::{feature_extractor::extract_features, labels::extract_label};

#[derive(Debug, Clone)]
pub struct CsvLoader {
    settings: DataSettings,
}

impl CsvLoader {
    pub fn new(settings: DataSettings) -> Self {
        Self { settings }
    }

    pub fn load(&self) -> Result<LobDataset, AppError> {
        let paths = self.resolve_paths()?;
        let mut segments = Vec::with_capacity(paths.len());
        for path in paths {
            segments.push(self.load_segment(&path)?);
        }
        Ok(LobDataset::new(segments))
    }

    pub fn load_segment(&self, path: &Path) -> Result<LobSegment, AppError> {
        reject_blank_rows(path)?;
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(self.settings.has_headers)
            .flexible(true)
            .from_path(path)
            .map_err(|source| AppError::CsvOpen {
                path: path.to_owned(),
                source,
            })?;
        let required_columns = self
            .settings
            .feature_columns
            .iter()
            .copied()
            .chain(std::iter::once(self.settings.label_column))
            .max()
            .map_or(0, |index| index + 1);

        let mut expected_columns = if self.settings.has_headers {
            let headers = reader.headers().map_err(|source| AppError::CsvRecord {
                path: path.to_owned(),
                row: 1,
                source,
            })?;
            if headers.len() < required_columns {
                return Err(AppError::CsvColumnCount {
                    path: path.to_owned(),
                    row: 1,
                    expected: required_columns,
                    found: headers.len(),
                });
            }
            Some(headers.len())
        } else {
            None
        };
        let mut features = Vec::new();
        let mut labels = Vec::new();
        for (record_index, result) in reader.records().enumerate() {
            let row = record_index + usize::from(self.settings.has_headers) + 1;
            let record = result.map_err(|source| AppError::CsvRecord {
                path: path.to_owned(),
                row,
                source,
            })?;
            if record.len() < required_columns {
                return Err(AppError::CsvColumnCount {
                    path: path.to_owned(),
                    row,
                    expected: required_columns,
                    found: record.len(),
                });
            }
            match expected_columns {
                Some(expected) if expected != record.len() => {
                    return Err(AppError::CsvColumnCount {
                        path: path.to_owned(),
                        row,
                        expected,
                        found: record.len(),
                    });
                }
                None => expected_columns = Some(record.len()),
                _ => {}
            }

            let row_features = extract_features(&record, &self.settings, path, row)?;
            let label = extract_label(&record, &self.settings, path, row)?;
            features.extend_from_slice(&row_features);
            labels.push(label);
        }
        if labels.is_empty() {
            return Err(AppError::EmptyDatasetSegment {
                path: path.to_owned(),
            });
        }
        LobSegment::new(path, features, labels)
    }

    fn resolve_paths(&self) -> Result<Vec<PathBuf>, AppError> {
        if !self.settings.segment_paths.is_empty() {
            return Ok(self.settings.segment_paths.clone());
        }
        let path = &self.settings.dataset_path;
        if path.is_file() {
            return Ok(vec![path.clone()]);
        }
        if path.is_dir() {
            let entries = fs::read_dir(path).map_err(|source| AppError::DatasetDirectory {
                path: path.clone(),
                source,
            })?;
            let entries = entries.collect::<Result<Vec<_>, _>>().map_err(|source| {
                AppError::DatasetDirectory {
                    path: path.clone(),
                    source,
                }
            })?;
            let mut paths = entries
                .into_iter()
                .map(|entry| entry.path())
                .filter(|entry| entry.is_file())
                .filter(|entry| {
                    entry
                        .extension()
                        .is_some_and(|extension| extension.eq_ignore_ascii_case("csv"))
                })
                .collect::<Vec<_>>();
            paths.sort();
            return match paths.len() {
                0 => Err(AppError::NoCsvSegments(path.clone())),
                1 => Ok(paths),
                _ => Err(AppError::SegmentOrderRequired(path.clone())),
            };
        }
        Err(AppError::InvalidDatasetPath { path: path.clone() })
    }
}

fn reject_blank_rows(path: &Path) -> Result<(), AppError> {
    use std::io::{BufRead, BufReader};

    let file = std::fs::File::open(path).map_err(|source| AppError::CsvScan {
        path: path.to_owned(),
        source,
    })?;
    let mut reader = BufReader::new(file);
    let mut bytes = Vec::new();
    let mut in_quotes = false;
    let mut physical_row = 0;
    loop {
        bytes.clear();
        let read = reader
            .read_until(b'\n', &mut bytes)
            .map_err(|source| AppError::CsvScan {
                path: path.to_owned(),
                source,
            })?;
        if read == 0 {
            break;
        }
        physical_row += 1;
        if bytes.last() == Some(&b'\n') {
            bytes.pop();
            if bytes.last() == Some(&b'\r') {
                bytes.pop();
            }
        } else if bytes.last() == Some(&b'\r') {
            bytes.pop();
        }
        if bytes.is_empty() && !in_quotes {
            return Err(AppError::CsvBlankRow {
                path: path.to_owned(),
                row: physical_row,
            });
        }
        let mut index = 0;
        while index < bytes.len() {
            if bytes[index] == b'"' {
                if in_quotes && bytes.get(index + 1) == Some(&b'"') {
                    index += 2;
                    continue;
                }
                in_quotes = !in_quotes;
            }
            index += 1;
        }
    }
    Ok(())
}
