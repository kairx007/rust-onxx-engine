use crate::{
    data::{FEATURE_COUNT, LobDataset},
    error::AppError,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowConfig {
    pub sequence_length: usize,
    pub stride: usize,
}

impl WindowConfig {
    pub fn validate(self) -> Result<Self, AppError> {
        if self.sequence_length != 100 {
            return Err(AppError::WindowConfiguration(
                "the model input contract requires a sequence length of 100".to_owned(),
            ));
        }
        if self.stride == 0 {
            return Err(AppError::WindowConfiguration(
                "window stride must be at least 1".to_owned(),
            ));
        }
        Ok(self)
    }
}

/// A window of `sequence_length` consecutive rows starting at `start_row`.
///
/// `label` is the label of row `start_row + sequence_length`, the tick immediately
/// after the window. This is the training convention (`cpp-engine/src/dataset.py`,
/// `__getitem__`) and the one the C++ `full_inference` engine evaluates with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowRef {
    pub segment_index: usize,
    pub start_row: usize,
    pub label: u8,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct WindowExtractor;

impl WindowExtractor {
    pub fn extract(dataset: &LobDataset, config: WindowConfig) -> Result<Vec<WindowRef>, AppError> {
        let config = config.validate()?;
        let mut windows = Vec::new();
        for (segment_index, segment) in dataset.segments().iter().enumerate() {
            let mut start_row = 0;
            // The label row (the tick after the window) must exist inside the segment.
            while start_row + config.sequence_length < segment.rows() {
                let label_row = start_row + config.sequence_length;
                let label = *segment.labels().get(label_row).ok_or_else(|| {
                    AppError::WindowReference(format!(
                        "label row {label_row} is missing in segment {}",
                        segment.source().display()
                    ))
                })?;
                let feature_end = (start_row + config.sequence_length)
                    .checked_mul(FEATURE_COUNT)
                    .ok_or_else(|| {
                        AppError::WindowReference("window feature offset overflow".to_owned())
                    })?;
                if feature_end > segment.features().len() {
                    return Err(AppError::WindowReference(format!(
                        "window ending at row {} exceeds segment {} feature buffer",
                        label_row,
                        segment.source().display()
                    )));
                }
                windows.push(WindowRef {
                    segment_index,
                    start_row,
                    label,
                });
                start_row += config.stride;
            }
        }
        Ok(windows)
    }
}
