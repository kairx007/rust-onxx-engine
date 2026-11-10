mod common;

use rust_engine::data::labels::{decode_label, encode_label};
use rust_engine::error::AppError;

use common::TempDir;

#[test]
fn encodes_and_decodes_fi2010_labels() {
    let path = TempDir::new();
    assert_eq!(encode_label("1", &path.path().join("x.csv"), 1).unwrap(), 0);
    assert_eq!(encode_label("2", &path.path().join("x.csv"), 2).unwrap(), 1);
    assert_eq!(encode_label("3", &path.path().join("x.csv"), 3).unwrap(), 2);
    assert_eq!(
        encode_label("1.0", &path.path().join("x.csv"), 4).unwrap(),
        0
    );
    assert_eq!(
        encode_label("2.0", &path.path().join("x.csv"), 5).unwrap(),
        1
    );
    assert_eq!(
        encode_label("3.0", &path.path().join("x.csv"), 6).unwrap(),
        2
    );
    assert_eq!(decode_label(0), Some("Down"));
    assert_eq!(decode_label(1), Some("Stationary"));
    assert_eq!(decode_label(2), Some("Up"));
}

#[test]
fn rejects_unknown_and_missing_label_values() {
    let temp = TempDir::new();
    assert!(matches!(
        encode_label("4", &temp.path().join("x.csv"), 7),
        Err(AppError::CsvLabel { row: 7, .. })
    ));
    assert!(encode_label("", &temp.path().join("x.csv"), 1).is_err());
    assert!(encode_label("2.5", &temp.path().join("x.csv"), 2).is_err());
}
