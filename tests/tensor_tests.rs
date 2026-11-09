use rust_engine::{
    data::{FEATURE_COUNT, LobSegment},
    error::AppError,
    preprocessing::WindowRef,
    runtime::Runtime,
    tensor::{INPUT_SHAPE, INPUT_VALUE_COUNT, InputTensor, validation},
};

#[test]
fn builds_contiguous_row_major_tensor_and_preserves_window_label() {
    let rows = 100;
    let features = (0..rows)
        .flat_map(|row| {
            (0..FEATURE_COUNT).map(move |feature| (row * FEATURE_COUNT + feature) as f32)
        })
        .collect::<Vec<_>>();
    let labels = (0..rows).map(|row| (row % 3) as u8).collect::<Vec<_>>();
    let segment = LobSegment::new("tensor.csv", features, labels).unwrap();
    let tensor = InputTensor::from_window(
        &segment,
        WindowRef {
            segment_index: 0,
            start_row: 0,
            label: 0,
        },
    )
    .unwrap();
    assert_eq!(tensor.shape(), [1, 1, 100, 40]);
    assert_eq!(tensor.data().len(), INPUT_VALUE_COUNT);
    assert_eq!(tensor.data()[0], 0.0);
    assert_eq!(tensor.data()[39], 39.0);
    assert_eq!(tensor.data()[40], 40.0);
    assert_eq!(tensor.data()[3_999], 3_999.0);
    assert!(tensor.validate().is_ok());
}

#[test]
fn rejects_wrong_tensor_shape_length_and_nonfinite_values() {
    assert!(matches!(
        validation::validate(&vec![0.0; INPUT_VALUE_COUNT], &[1, 1, 100, 39]),
        Err(AppError::TensorValidation(_))
    ));
    assert!(matches!(
        validation::validate(&vec![0.0; INPUT_VALUE_COUNT - 1], &INPUT_SHAPE),
        Err(AppError::TensorValidation(_))
    ));
    let mut values = vec![0.0; INPUT_VALUE_COUNT];
    values[9] = f32::NAN;
    assert!(matches!(
        validation::validate(&values, &INPUT_SHAPE),
        Err(AppError::TensorValidation(_))
    ));
}

#[test]
fn transfers_owned_buffer_into_ort_value_without_running_inference() {
    let _runtime = Runtime::initialize("cpu").expect("ORT runtime should initialize");
    let segment =
        LobSegment::new("tensor.csv", vec![1.0; INPUT_VALUE_COUNT], vec![2; 100]).unwrap();
    let tensor = InputTensor::from_window(
        &segment,
        WindowRef {
            segment_index: 0,
            start_row: 0,
            label: 2,
        },
    )
    .unwrap()
    .into_ort_tensor()
    .expect("valid tensor should convert to ORT");
    let shape = tensor
        .dtype()
        .tensor_shape()
        .expect("ORT value should be a tensor");
    assert_eq!(
        shape.iter().copied().collect::<Vec<_>>(),
        vec![1, 1, 100, 40]
    );
}
