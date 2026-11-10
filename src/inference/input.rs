use ort::value::TensorRef;

use crate::{error::AppError, tensor::InputTensor};

pub(super) fn adapt(input: &InputTensor) -> Result<TensorRef<'_, f32>, AppError> {
    input
        .validate()
        .map_err(|error| AppError::InvalidInferenceInput(error.to_string()))?;
    TensorRef::from_array_view((input.shape(), input.data()))
        .map_err(|error| AppError::InputConversion(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::adapt;
    use crate::{
        data::{FEATURE_COUNT, LobSegment},
        preprocessing::WindowRef,
        runtime::Runtime,
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
}
