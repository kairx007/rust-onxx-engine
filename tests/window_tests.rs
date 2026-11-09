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
    let cases = [(50, 0), (100, 1), (150, 1), (200, 2), (250, 2), (300, 3)];
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
            assert_eq!(window.label, ((window.start_row + 99) % 3) as u8);
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
fn rejects_overlapping_or_nonstandard_window_configuration() {
    let dataset = dataset_with_rows(200);
    assert!(matches!(
        WindowExtractor::extract(
            &dataset,
            WindowConfig {
                sequence_length: 100,
                stride: 50
            }
        ),
        Err(AppError::WindowConfiguration(_))
    ));
}
