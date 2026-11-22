//! Export of the exact preprocessed input tensors, so other inference engines
//! (C++, Python) can be fed bit-identical inputs for latency and numerical
//! comparisons.

use std::{
    fs::{File, OpenOptions},
    io::BufWriter,
    path::{Path, PathBuf},
};

use npyz::WriterBuilder;
use serde::Serialize;

use crate::{
    config::Settings, data::FEATURE_COUNT, error::AppError, preprocessing::PreprocessingPipeline,
    tensor::INPUT_SHAPE,
};

use super::inference_service::file_fingerprint;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportSummary {
    pub windows: usize,
    pub windows_path: PathBuf,
    pub labels_path: PathBuf,
    pub manifest_path: PathBuf,
}

#[derive(Debug, Serialize)]
struct WindowManifest {
    format: &'static str,
    windows: usize,
    tensor_shape: [usize; 4],
    tensor_dtype: &'static str,
    labels_dtype: &'static str,
    class_mapping: [&'static str; 3],
    label_rule: &'static str,
    sequence_length: usize,
    stride: usize,
    normalization_mean_path: String,
    normalization_std_path: String,
    normalization_epsilon: f32,
    input_sources: Vec<String>,
    window_refs: Vec<WindowRefRecord>,
    windows_file_hash: Option<String>,
    labels_file_hash: Option<String>,
}

#[derive(Debug, Serialize)]
struct WindowRefRecord {
    window_index: usize,
    source_file: String,
    segment_index: usize,
    start_row: usize,
    label: u8,
}

/// Writes `windows.npy` (`[N, 1, 100, 40]` float32, C order), `labels.npy`
/// (`[N]` uint8, 0=Up 1=Stationary 2=Down), and `windows.json` (manifest)
/// into `output_dir`. Existing files are never replaced.
pub fn export_windows(settings: &Settings, output_dir: &Path) -> Result<ExportSummary, AppError> {
    let prepared = PreprocessingPipeline::new(settings)?.prepare()?;
    let count = prepared.windows().len();
    if count == 0 {
        return Err(AppError::ConfigValidation(
            "input dataset produced no complete windows (100 rows plus a label row)".to_owned(),
        ));
    }

    let values_per_window = INPUT_SHAPE.iter().product::<usize>();
    let mut values = Vec::with_capacity(count * values_per_window);
    let mut labels = Vec::with_capacity(count);
    let mut refs = Vec::with_capacity(count);
    for index in 0..count {
        let window = prepared.window(index)?;
        values.extend_from_slice(window.tensor.data());
        labels.push(window.label);
        refs.push(WindowRefRecord {
            window_index: index,
            source_file: window.source.display().to_string(),
            segment_index: window.reference.segment_index,
            start_row: window.reference.start_row,
            label: window.label,
        });
    }

    std::fs::create_dir_all(output_dir).map_err(|source| AppError::OutputCreate {
        path: output_dir.to_owned(),
        source,
    })?;
    let windows_path = output_dir.join("windows.npy");
    let labels_path = output_dir.join("labels.npy");
    let manifest_path = output_dir.join("windows.json");
    for path in [&windows_path, &labels_path, &manifest_path] {
        if path.exists() {
            return Err(AppError::OutputExists(path.clone()));
        }
    }

    let shape = [
        count as u64,
        INPUT_SHAPE[1] as u64,
        INPUT_SHAPE[2] as u64,
        FEATURE_COUNT as u64,
    ];
    write_npy(&windows_path, &shape, &values)?;
    write_npy(&labels_path, &[count as u64], &labels)?;

    let mut input_sources = refs
        .iter()
        .map(|record| record.source_file.clone())
        .collect::<Vec<_>>();
    input_sources.dedup();
    let manifest = WindowManifest {
        format: "rustlob-windows-v1",
        windows: count,
        tensor_shape: [count, INPUT_SHAPE[1], INPUT_SHAPE[2], INPUT_SHAPE[3]],
        tensor_dtype: "float32 little-endian, C order",
        labels_dtype: "uint8",
        class_mapping: ["0=Up", "1=Stationary", "2=Down"],
        label_rule: "label of the row immediately after each window (training convention)",
        sequence_length: settings.preprocessing.sequence_length,
        stride: settings.preprocessing.stride,
        normalization_mean_path: settings.normalization.mean_path.display().to_string(),
        normalization_std_path: settings.normalization.std_path.display().to_string(),
        normalization_epsilon: settings.preprocessing.epsilon,
        input_sources,
        window_refs: refs,
        windows_file_hash: file_fingerprint(&windows_path),
        labels_file_hash: file_fingerprint(&labels_path),
    };
    let file = create_new(&manifest_path)?;
    serde_json::to_writer_pretty(file, &manifest)
        .map_err(|error| AppError::BenchmarkSerialization(error.to_string()))?;

    Ok(ExportSummary {
        windows: count,
        windows_path,
        labels_path,
        manifest_path,
    })
}

fn create_new(path: &Path) -> Result<File, AppError> {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|source| {
            if source.kind() == std::io::ErrorKind::AlreadyExists {
                AppError::OutputExists(path.to_owned())
            } else {
                AppError::OutputCreate {
                    path: path.to_owned(),
                    source,
                }
            }
        })
}

fn write_npy<T>(path: &Path, shape: &[u64], values: &[T]) -> Result<(), AppError>
where
    T: Clone + npyz::AutoSerialize,
{
    let file = create_new(path)?;
    let to_error = |source: std::io::Error| AppError::OutputWrite {
        path: path.to_owned(),
        source,
    };
    let mut writer = npyz::WriteOptions::new()
        .default_dtype()
        .shape(shape)
        .writer(BufWriter::new(file))
        .begin_nd()
        .map_err(to_error)?;
    writer.extend(values.iter().cloned()).map_err(to_error)?;
    writer.finish().map_err(to_error)
}

#[cfg(test)]
mod tests {
    use super::export_windows;
    use crate::config::Settings;

    #[test]
    fn export_requires_a_verified_schema_before_writing_anything() {
        let settings = Settings::parse(include_str!("../../configs/default.toml")).unwrap();
        let dir = std::env::temp_dir().join(format!("rustlob-export-{}", std::process::id()));
        assert!(export_windows(&settings, &dir).is_err());
        assert!(!dir.join("windows.npy").exists());
    }
}
