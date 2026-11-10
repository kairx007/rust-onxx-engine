use ort::value::DynValue;

use crate::{error::AppError, inference::Prediction, runtime::TensorMetadata};

pub(super) fn extract(value: &DynValue, metadata: &TensorMetadata) -> Result<Prediction, AppError> {
    let (shape, values) = value
        .try_extract_tensor::<f32>()
        .map_err(|error| AppError::InvalidInferenceOutput(error.to_string()))?;
    let expected = metadata.dimensions.as_deref().ok_or_else(|| {
        AppError::InvalidInferenceOutput(format!(
            "output '{}' has no tensor dimensions in model metadata",
            metadata.name
        ))
    })?;
    parse_scores(shape, expected, values)
}

fn parse_scores(
    shape: &[i64],
    expected_shape: &[i64],
    values: &[f32],
) -> Result<Prediction, AppError> {
    if shape.len() != expected_shape.len()
        || shape
            .iter()
            .zip(expected_shape)
            .any(|(actual, expected)| *actual <= 0 || (*expected >= 0 && actual != expected))
    {
        return Err(AppError::InvalidInferenceOutput(format!(
            "runtime output shape {shape:?} does not match model metadata {expected_shape:?}"
        )));
    }
    if shape.last() != Some(&3)
        || shape[..shape.len() - 1]
            .iter()
            .any(|dimension| *dimension != 1)
    {
        return Err(AppError::InvalidInferenceOutput(format!(
            "expected one sample with 3 classes on the final axis, found {shape:?}"
        )));
    }
    if values.len() != 3 {
        return Err(AppError::InvalidInferenceOutput(format!(
            "expected 3 class scores, found {}",
            values.len()
        )));
    }
    Prediction::from_scores([values[0], values[1], values[2]])
}

#[cfg(test)]
mod tests {
    use super::parse_scores;
    use crate::error::AppError;

    #[test]
    fn extracts_three_scores_from_vector_or_single_batch() {
        for shape in [&[3][..], &[1, 3][..], &[1, 1, 3][..]] {
            let expected = shape
                .iter()
                .map(|dimension| if *dimension == 1 { -1 } else { *dimension })
                .collect::<Vec<_>>();
            let result = parse_scores(shape, &expected, &[0.1, 0.7, 0.2]).unwrap();
            assert_eq!(result.scores(), [0.1, 0.7, 0.2]);
            assert_eq!(result.class_index(), 1);
        }
    }

    #[test]
    fn rejects_wrong_shapes_counts_and_nonfinite_scores() {
        assert!(matches!(
            parse_scores(&[2, 3], &[-1, 3], &[0.1, 0.2, 0.3]),
            Err(AppError::InvalidInferenceOutput(_))
        ));
        assert!(parse_scores(&[3], &[3], &[0.1, 0.2]).is_err());
        assert!(parse_scores(&[3], &[3], &[0.1, f32::NAN, 0.3]).is_err());
    }
}
