mod common;

use rust_engine::{
    preprocessing::PreprocessingPipeline,
    tensor::{INPUT_SHAPE, INPUT_VALUE_COUNT},
};

use common::{TempDir, default_settings, write_csv_segment, write_npy};

#[test]
fn runs_synthetic_data_through_loading_normalization_windows_and_tensor_validation() {
    let temp = TempDir::new();
    let short_segment = temp.path().join("segment-01.csv");
    let full_segment = temp.path().join("segment-02.csv");
    let means = temp.path().join("mu.npy");
    let stds = temp.path().join("sigma.npy");
    write_csv_segment(&short_segment, 60, 1);
    write_csv_segment(&full_segment, 120, 3);
    write_npy(&means, &[0.0_f32; 40]);
    write_npy(&stds, &[1.0_f32; 40]);

    let mut settings = default_settings();
    settings.data.segment_paths = vec![short_segment, full_segment.clone()];
    settings.data.schema_verified = true;
    settings.normalization.mean_path = means;
    settings.normalization.std_path = stds;
    settings
        .validate()
        .expect("synthetic settings should remain valid");

    let prepared = PreprocessingPipeline::new(&settings)
        .expect("synthetic normalization arrays should load")
        .prepare()
        .expect("synthetic CSV segments should preprocess");
    assert_eq!(prepared.segment_count(), 2);
    assert_eq!(prepared.total_rows(), 180);
    assert_eq!(prepared.windows().len(), 1);
    assert_eq!(prepared.windows()[0].segment_index, 1);
    assert_eq!(prepared.windows()[0].label, 2);

    let window = prepared
        .window(0)
        .expect("first complete window should materialize");
    assert_eq!(window.source, full_segment);
    assert_eq!(window.label, 2);
    assert_eq!(window.tensor.shape(), INPUT_SHAPE);
    assert_eq!(window.tensor.data().len(), INPUT_VALUE_COUNT);
    assert_eq!(window.tensor.data()[39], 39.0 / (1.0 + 1e-8_f32));
    window
        .tensor
        .validate()
        .expect("pipeline tensor should validate");
}
