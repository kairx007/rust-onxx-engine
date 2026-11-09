mod common;

use csv::StringRecord;
use rust_engine::data::feature_extractor::extract_features;

use common::{TempDir, default_settings};

#[test]
fn extracts_exactly_40_features_in_configured_order() {
    let mut settings = default_settings();
    settings.data.feature_columns.swap(0, 39);
    let record = StringRecord::from((0..145).map(|value| value.to_string()).collect::<Vec<_>>());
    let temp = TempDir::new();
    let features =
        extract_features(&record, &settings.data, &temp.path().join("row.csv"), 1).unwrap();
    assert_eq!(features.len(), 40);
    assert_eq!(features[0], 39.0);
    assert_eq!(features[1], 1.0);
    assert_eq!(features[39], 0.0);
}

#[test]
fn rejects_nonfinite_feature_values() {
    let settings = default_settings();
    let mut values = (0..145).map(|value| value.to_string()).collect::<Vec<_>>();
    values[5] = "NaN".to_owned();
    let record = StringRecord::from(values);
    let temp = TempDir::new();
    assert!(extract_features(&record, &settings.data, &temp.path().join("row.csv"), 1).is_err());
}
