use std::path::{Path, PathBuf};

use crate::error::AppError;

pub const FEATURE_COUNT: usize = 40;

#[derive(Debug, Clone, Default)]
pub struct LobDataset {
    segments: Vec<LobSegment>,
}

#[derive(Debug, Clone)]
pub struct LobSegment {
    source: PathBuf,
    features: Vec<f32>,
    labels: Vec<u8>,
}

impl LobSegment {
    pub fn new(
        source: impl AsRef<Path>,
        features: Vec<f32>,
        labels: Vec<u8>,
    ) -> Result<Self, AppError> {
        let source = source.as_ref().to_owned();
        let expected =
            labels
                .len()
                .checked_mul(FEATURE_COUNT)
                .ok_or_else(|| AppError::DatasetInvariant {
                    path: source.clone(),
                    details: "feature buffer length overflow".to_owned(),
                })?;
        if features.len() != expected {
            return Err(AppError::DatasetInvariant {
                path: source,
                details: format!(
                    "{} feature values for {} labels; expected {expected}",
                    features.len(),
                    labels.len()
                ),
            });
        }
        if let Some((index, value)) = features
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(AppError::DatasetInvariant {
                path: source,
                details: format!("feature value at flat index {index} is not finite: {value}"),
            });
        }
        if let Some((index, label)) = labels.iter().enumerate().find(|(_, label)| **label > 2) {
            return Err(AppError::DatasetInvariant {
                path: source,
                details: format!("internal label at row {index} is outside 0..=2: {label}"),
            });
        }
        Ok(Self {
            source,
            features,
            labels,
        })
    }

    pub fn source(&self) -> &Path {
        &self.source
    }

    pub fn rows(&self) -> usize {
        self.labels.len()
    }

    pub fn features(&self) -> &[f32] {
        &self.features
    }

    pub fn features_mut(&mut self) -> &mut [f32] {
        &mut self.features
    }

    pub fn labels(&self) -> &[u8] {
        &self.labels
    }
}

impl LobDataset {
    pub fn new(segments: Vec<LobSegment>) -> Self {
        Self { segments }
    }

    pub fn segments(&self) -> &[LobSegment] {
        &self.segments
    }

    pub fn segments_mut(&mut self) -> &mut [LobSegment] {
        &mut self.segments
    }

    pub fn total_rows(&self) -> usize {
        self.segments.iter().map(LobSegment::rows).sum()
    }
}
