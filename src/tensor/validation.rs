use crate::error::AppError;

pub const INPUT_SHAPE: [usize; 4] = [1, 1, 100, 40];
pub const INPUT_VALUE_COUNT: usize = 4_000;

pub fn validate(data: &[f32], shape: &[usize; 4]) -> Result<(), AppError> {
    if shape != &INPUT_SHAPE {
        return Err(AppError::TensorValidation(format!(
            "expected shape {INPUT_SHAPE:?}, found {shape:?}"
        )));
    }
    if data.len() != INPUT_VALUE_COUNT {
        return Err(AppError::TensorValidation(format!(
            "expected {INPUT_VALUE_COUNT} values, found {}",
            data.len()
        )));
    }
    if let Some((index, value)) = data
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(AppError::TensorValidation(format!(
            "value at flat index {index} is not finite: {value}"
        )));
    }
    Ok(())
}
