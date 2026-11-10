use ort::value::TensorRef;

use crate::{error::AppError, models::TransformerLobContract, tensor::InputTensor};

pub(super) fn adapt(input: &InputTensor) -> Result<TensorRef<'_, f32>, AppError> {
    input
        .validate()
        .map_err(|error| AppError::InvalidInferenceInput(error.to_string()))?;
    TensorRef::from_array_view((input.shape(), input.data()))
        .map_err(|error| AppError::InputConversion(error.to_string()))
}

pub(super) fn adapt_transformer<'a>(
    input: &'a InputTensor,
    contract: &TransformerLobContract,
) -> Result<TensorRef<'a, f32>, AppError> {
    input
        .validate()
        .map_err(|error| AppError::InvalidInferenceInput(error.to_string()))?;
    let expected = contract
        .input_metadata()
        .dimensions
        .as_deref()
        .ok_or_else(|| {
            AppError::InvalidInferenceInput("model input has no dimensions".to_owned())
        })?;
    let actual = input.shape().map(|dimension| dimension as i64);
    if expected.len() != actual.len()
        || expected
            .iter()
            .zip(actual)
            .any(|(expected, actual)| *expected >= 0 && *expected != actual)
    {
        return Err(AppError::InvalidInferenceInput(format!(
            "Phase 2 tensor shape {:?} does not satisfy TransformerLOB model input shape {expected:?}; no layout transform is configured",
            input.shape()
        )));
    }
    TensorRef::from_array_view((input.shape(), input.data()))
        .map_err(|error| AppError::InputConversion(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::{adapt, adapt_transformer};
    use crate::{
        config::Settings,
        data::{FEATURE_COUNT, LobSegment},
        models::TransformerLobContract,
        preprocessing::WindowRef,
        runtime::{ModelMetadata, Runtime, TensorMetadata},
        tensor::InputTensor,
    };

    #[test]
    fn creates_a_borrowed_tensor_view_with_the_phase_two_shape_and_order() {
        let _runtime = Runtime::initialize("cpu").expect("runtime should initialize");
        let values = (0..4_000).map(|value| value as f32).collect::<Vec<_>>();
        let segment = LobSegment::new("synthetic.csv", values, vec![2; 100]).unwrap();
        assert_eq!(segment.features().len(), 100 * FEATURE_COUNT);
        let input = InputTensor::from_window(
            &segment,
            WindowRef {
                segment_index: 0,
                start_row: 0,
                label: 2,
            },
        )
        .unwrap();

        let ort_input = adapt(&input).expect("valid input should adapt");
        let shape = ort_input
            .dtype()
            .tensor_shape()
            .expect("adapter should return a tensor");
        assert_eq!(shape.iter().copied().collect::<Vec<_>>(), [1, 1, 100, 40]);
        let (_, data) = ort_input
            .try_extract_tensor::<f32>()
            .expect("borrowed tensor should retain f32 data");
        assert_eq!(data[0], 0.0);
        assert_eq!(data[40], 40.0);
        assert_eq!(data[3_999], 3_999.0);
    }

    #[test]
    fn transformer_adapter_preserves_phase2_shape_and_values_for_dynamic_batch() {
        let _runtime = Runtime::initialize("cpu").expect("runtime should initialize");
        let settings = Settings::parse(include_str!("../../configs/default.toml")).unwrap();
        let metadata = ModelMetadata {
            inputs: vec![TensorMetadata {
                name: "dynamic_transformer_input".to_owned(),
                element_type: "Float32".to_owned(),
                dimensions: Some(vec![-1, 1, 100, 40]),
            }],
            outputs: vec![TensorMetadata {
                name: "scores".to_owned(),
                element_type: "Float32".to_owned(),
                dimensions: Some(vec![-1, 3]),
            }],
        };
        let contract = TransformerLobContract::from_metadata(&metadata, &settings.model_specs)
            .expect("dynamic batch metadata should be compatible");
        let values = (0..4_000).map(|value| value as f32).collect::<Vec<_>>();
        let segment = LobSegment::new("synthetic.csv", values, vec![2; 100]).unwrap();
        let input = InputTensor::from_window(
            &segment,
            WindowRef {
                segment_index: 0,
                start_row: 0,
                label: 2,
            },
        )
        .unwrap();

        let adapted = adapt_transformer(&input, &contract).unwrap();
        let shape = adapted.dtype().tensor_shape().unwrap();
        assert_eq!(shape.iter().copied().collect::<Vec<_>>(), [1, 1, 100, 40]);
        let (_, values) = adapted.try_extract_tensor::<f32>().unwrap();
        assert_eq!(values[0], 0.0);
        assert_eq!(values[40], 40.0);
        assert_eq!(values[3_999], 3_999.0);
    }
}
