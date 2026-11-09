mod common;

use std::fs;

use rust_engine::{data::CsvLoader, error::AppError};

use common::{TempDir, default_settings};

fn record(label: &str) -> String {
    let mut fields = (0..145)
        .map(|index| format!("{index}.5"))
        .collect::<Vec<_>>();
    fields[144] = label.to_owned();
    fields.join(",")
}

#[test]
fn loads_features_labels_and_preserves_row_order() {
    let temp = TempDir::new();
    let path = temp.path().join("rows.csv");
    let rows = [record("1"), record("2"), record("3")].join("\n");
    fs::write(&path, rows).expect("CSV fixture should be written");
    let mut settings = default_settings();
    settings.data.dataset_path = path.clone();

    let dataset = CsvLoader::new(settings.data)
        .load()
        .expect("valid rows should load");
    let segment = &dataset.segments()[0];
    assert_eq!(segment.rows(), 3);
    assert_eq!(&segment.features()[..3], &[0.5, 1.5, 2.5]);
    assert_eq!(segment.features()[39], 39.5);
    assert_eq!(segment.labels(), &[0, 1, 2]);
}

#[test]
fn supports_headers_and_rejects_inconsistent_record_width() {
    let temp = TempDir::new();
    let path = temp.path().join("with-headers.csv");
    let header = (0..145)
        .map(|index| format!("c{index}"))
        .collect::<Vec<_>>()
        .join(",");
    fs::write(&path, format!("{header}\n{}\n", record("1")))
        .expect("CSV fixture should be written");
    let mut settings = default_settings();
    settings.data.dataset_path = path.clone();
    settings.data.has_headers = true;
    assert_eq!(
        CsvLoader::new(settings.data.clone())
            .load()
            .unwrap()
            .total_rows(),
        1
    );

    fs::write(&path, format!("{header}\n{}\n1,2\n", record("1")))
        .expect("CSV fixture should be rewritten");
    assert!(matches!(
        CsvLoader::new(settings.data).load(),
        Err(AppError::CsvColumnCount { row: 3, .. })
    ));
}

#[test]
fn reports_missing_empty_malformed_and_short_rows() {
    let temp = TempDir::new();
    let mut settings = default_settings();
    settings.data.dataset_path = temp.path().join("missing.csv");
    assert!(matches!(
        CsvLoader::new(settings.data.clone()).load(),
        Err(AppError::InvalidDatasetPath { .. })
    ));

    let empty = temp.path().join("empty.csv");
    fs::write(&empty, "").unwrap();
    settings.data.dataset_path = empty;
    assert!(matches!(
        CsvLoader::new(settings.data.clone()).load(),
        Err(AppError::EmptyDatasetSegment { .. })
    ));

    let malformed = temp.path().join("malformed.csv");
    fs::write(
        &malformed,
        format!(
            "\"closed\"x,{}\n",
            (1..145)
                .map(|value| value.to_string())
                .collect::<Vec<_>>()
                .join(",")
        ),
    )
    .unwrap();
    settings.data.dataset_path = malformed;
    assert!(matches!(
        CsvLoader::new(settings.data.clone()).load(),
        Err(AppError::CsvRecord { .. } | AppError::CsvNumericValue { column: 0, .. })
    ));

    let short = temp.path().join("short.csv");
    fs::write(&short, "1,2,3\n").unwrap();
    settings.data.dataset_path = short;
    assert!(matches!(
        CsvLoader::new(settings.data).load(),
        Err(AppError::CsvColumnCount { .. })
    ));
}

#[test]
fn rejects_invalid_numeric_values_and_labels() {
    let temp = TempDir::new();
    let mut invalid_number = record("1")
        .split(',')
        .map(str::to_owned)
        .collect::<Vec<_>>();
    invalid_number[7] = "not-a-number".to_owned();
    let numeric_path = temp.path().join("numeric.csv");
    fs::write(&numeric_path, invalid_number.join(",")).unwrap();
    let mut settings = default_settings();
    settings.data.dataset_path = numeric_path;
    assert!(matches!(
        CsvLoader::new(settings.data.clone()).load(),
        Err(AppError::CsvNumericValue { column: 7, .. })
    ));

    let label_path = temp.path().join("label.csv");
    fs::write(&label_path, record("4")).unwrap();
    settings.data.dataset_path = label_path;
    assert!(matches!(
        CsvLoader::new(settings.data).load(),
        Err(AppError::CsvLabel { .. })
    ));
}

#[test]
fn configured_segment_order_is_preserved_and_boundaries_retained() {
    let temp = TempDir::new();
    let later_name = temp.path().join("z.csv");
    let earlier_name = temp.path().join("a.csv");
    fs::write(&later_name, record("3")).unwrap();
    fs::write(&earlier_name, record("1")).unwrap();
    let mut settings = default_settings();
    settings.data.segment_paths = vec![later_name, earlier_name];
    let dataset = CsvLoader::new(settings.data).load().unwrap();
    assert_eq!(dataset.segments().len(), 2);
    assert_eq!(dataset.segments()[0].labels(), &[2]);
    assert_eq!(dataset.segments()[1].labels(), &[0]);
}

#[test]
fn rejects_blank_rows_instead_of_silently_skipping_them() {
    let temp = TempDir::new();
    let path = temp.path().join("blank-row.csv");
    fs::write(&path, format!("{}\n\n{}", record("1"), record("2")))
        .expect("CSV fixture should be written");
    let mut settings = default_settings();
    settings.data.dataset_path = path;
    assert!(matches!(
        CsvLoader::new(settings.data).load(),
        Err(AppError::CsvBlankRow { row: 2, .. })
    ));
}

#[test]
fn requires_explicit_order_for_multiple_csv_files_in_a_directory() {
    let temp = TempDir::new();
    fs::write(temp.path().join("day-b.csv"), record("1")).unwrap();
    fs::write(temp.path().join("day-a.csv"), record("2")).unwrap();
    let mut settings = default_settings();
    settings.data.dataset_path = temp.path().to_owned();
    assert!(matches!(
        CsvLoader::new(settings.data).load(),
        Err(AppError::SegmentOrderRequired(_))
    ));
}
