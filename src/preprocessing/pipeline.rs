use std::path::PathBuf;

use crate::{
    config::Settings,
    data::{CsvLoader, LobDataset},
    error::AppError,
    preprocessing::{NormalizationStats, Normalizer, WindowConfig, WindowExtractor, WindowRef},
    tensor::InputTensor,
};

#[derive(Debug, Clone)]
pub struct PreprocessingPipeline {
    data_settings: crate::config::DataSettings,
    window_config: WindowConfig,
    normalizer: Normalizer,
}

impl PreprocessingPipeline {
    pub fn new(settings: &Settings) -> Result<Self, AppError> {
        if !settings.data.schema_verified {
            return Err(AppError::ConfigValidation(
                "data.schema_verified is false; inspect the FI-2010 files and confirm header, feature, and label columns before preprocessing".to_owned(),
            ));
        }
        let stats = NormalizationStats::load(&settings.normalization)?;
        let normalizer = Normalizer::new(stats, settings.preprocessing.epsilon)?;
        let window_config = WindowConfig {
            sequence_length: settings.preprocessing.sequence_length,
            stride: settings.preprocessing.stride,
        }
        .validate()?;
        Ok(Self {
            data_settings: settings.data.clone(),
            window_config,
            normalizer,
        })
    }

    pub fn prepare(&self) -> Result<PreparedDataset, AppError> {
        let mut dataset = CsvLoader::new(self.data_settings.clone()).load()?;
        self.normalizer.normalize(&mut dataset)?;
        let windows = WindowExtractor::extract(&dataset, self.window_config)?;
        Ok(PreparedDataset { dataset, windows })
    }
}

#[derive(Debug, Clone)]
pub struct PreparedDataset {
    dataset: LobDataset,
    windows: Vec<WindowRef>,
}

impl PreparedDataset {
    pub fn total_rows(&self) -> usize {
        self.dataset.total_rows()
    }

    pub fn segment_count(&self) -> usize {
        self.dataset.segments().len()
    }

    pub fn windows(&self) -> &[WindowRef] {
        &self.windows
    }

    pub fn window(&self, index: usize) -> Result<PreparedWindow, AppError> {
        let reference = *self.windows.get(index).ok_or_else(|| {
            AppError::WindowReference(format!(
                "window index {index} is out of range ({} windows available)",
                self.windows.len()
            ))
        })?;
        let segment = self
            .dataset
            .segments()
            .get(reference.segment_index)
            .ok_or_else(|| {
                AppError::WindowReference(format!(
                    "segment index {} is out of range",
                    reference.segment_index
                ))
            })?;
        let tensor = InputTensor::from_window(segment, reference)?;
        Ok(PreparedWindow {
            reference,
            tensor,
            label: reference.label,
            source: PathBuf::from(segment.source()),
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PreparedWindow {
    pub reference: WindowRef,
    pub tensor: InputTensor,
    pub label: u8,
    pub source: PathBuf,
}
