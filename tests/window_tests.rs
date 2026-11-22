use rust_engine::{
    data::{FEATURE_COUNT, LobDataset, LobSegment},
    error::AppError,
    preprocessing::{WindowConfig, WindowExtractor},
};

fn dataset_with_rows(rows: usize) -> LobDataset {
    let features = (0..rows)
        .flat_map(|row| std::iter::repeat_n(row as f32, FEATURE_COUNT))
        .collect();
    let labels = (0..rows).map(|row| (row % 3) as u8).collect();
    LobDataset::new(vec![
        LobSegment::new("segment.csv", features, labels).unwrap(),
    ])
}

#[test]
fn extracts_nonoverlapping_windows_and_discards_trailing_rows() {
    // A window needs 100 rows plus the label row that follows them.
    let cases = [
        (50, 0),
        (100, 0),
        (101, 1),
        (200, 1),
        (201, 2),
        (300, 2),
        (301, 3),
    ];
    for (rows, expected) in cases {
        let dataset = dataset_with_rows(rows);
        let windows = WindowExtractor::extract(
            &dataset,
            WindowConfig {
                sequence_length: 100,
                stride: 100,
            },
        )
        .unwrap();
        assert_eq!(windows.len(), expected, "row count {rows}");
        for (index, window) in windows.iter().enumerate() {
            assert_eq!(window.start_row, index * 100);
            assert_eq!(window.label, ((window.start_row + 100) % 3) as u8);
        }
    }
}

#[test]
fn never_combines_rows_across_segment_boundaries() {
    let first = dataset_with_rows(60).segments()[0].clone();
    let second = dataset_with_rows(60).segments()[0].clone();
    let dataset = LobDataset::new(vec![first, second]);
    let windows = WindowExtractor::extract(
        &dataset,
        WindowConfig {
            sequence_length: 100,
            stride: 100,
        },
    )
    .unwrap();
    assert!(windows.is_empty());
}

#[test]
fn stride_one_matches_the_training_window_convention() {
    // cpp-engine/src/dataset.py: len = rows - 100; window i = rows i..i+99, label = row i+100.
    let dataset = dataset_with_rows(250);
    let windows = WindowExtractor::extract(
        &dataset,
        WindowConfig {
            sequence_length: 100,
            stride: 1,
        },
    )
    .unwrap();
    assert_eq!(windows.len(), 150);
    for (index, window) in windows.iter().enumerate() {
        assert_eq!(window.start_row, index);
        assert_eq!(window.label, ((index + 100) % 3) as u8);
    }
}

#[test]
fn rejects_nonstandard_length_and_zero_stride() {
    let dataset = dataset_with_rows(200);
    for config in [
        WindowConfig {
            sequence_length: 99,
            stride: 1,
        },
        WindowConfig {
            sequence_length: 100,
            stride: 0,
        },
    ] {
        assert!(matches!(
            WindowExtractor::extract(&dataset, config),
            Err(AppError::WindowConfiguration(_))
        ));
    }
}
