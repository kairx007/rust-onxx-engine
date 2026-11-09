use crate::{
    data::{FEATURE_COUNT, LobSegment},
    error::AppError,
    preprocessing::WindowRef,
};

use super::{INPUT_SHAPE, validation};

#[derive(Debug, Clone, PartialEq)]
pub struct InputTensor {
    data: Vec<f32>,
    shape: [usize; 4],
}

impl InputTensor {
    pub fn from_window(segment: &LobSegment, window: WindowRef) -> Result<Self, AppError> {
        let end_row = window
            .start_row
            .checked_add(100)
            .ok_or_else(|| AppError::WindowReference("window row offset overflow".to_owned()))?;
        if end_row > segment.rows() {
            return Err(AppError::WindowReference(format!(
                "window rows {}..{} exceed segment with {} rows",
                window.start_row,
                end_row,
                segment.rows()
            )));
        }
        let label_row = end_row - 1;
        if segment.labels()[label_row] != window.label {
            return Err(AppError::WindowReference(format!(
                "window label {} does not match segment label {} at row {label_row}",
                window.label,
                segment.labels()[label_row]
            )));
        }
        let start = window.start_row * FEATURE_COUNT;
        let end = end_row * FEATURE_COUNT;
        let data = segment.features()[start..end].to_vec();
        validation::validate(&data, &INPUT_SHAPE)?;
        Ok(Self {
            data,
            shape: INPUT_SHAPE,
        })
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn shape(&self) -> [usize; 4] {
        self.shape
    }

    pub fn validate(&self) -> Result<(), AppError> {
        validation::validate(&self.data, &self.shape)
    }

    pub fn into_ort_tensor(self) -> Result<ort::value::Tensor<f32>, AppError> {
        self.validate()?;
        ort::value::Tensor::from_array((self.shape, self.data.into_boxed_slice())).map_err(
            |error| {
                AppError::TensorValidation(format!(
                    "could not create ONNX Runtime input tensor: {error}"
                ))
            },
        )
    }
}
