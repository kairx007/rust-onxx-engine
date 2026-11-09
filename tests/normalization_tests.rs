mod common;

use rust_engine::{
    config::NormalizationSettings,
    data::{FEATURE_COUNT, LobDataset, LobSegment},
    error::AppError,
    preprocessing::{NormalizationStats, Normalizer},
};

use common::{TempDir, write_npy, write_npy_shape};

#[test]
fn loads_f32_and_f64_npy_statistics_with_expected_shape() {
    let temp = TempDir::new();
    let mean_path = temp.path().join("mu.npy");
    let std_path = temp.path().join("sigma.npy");
    write_npy(&mean_path, &[0.0_f32; FEATURE_COUNT]);
    write_npy(&std_path, &[2.0_f64; FEATURE_COUNT]);
    let settings = NormalizationSettings {
        mean_path,
        std_path,
    };
    let stats = NormalizationStats::load(&settings).expect("valid 1D statistics should load");
    assert_eq!(stats.mean(), &[0.0; FEATURE_COUNT]);
    assert_eq!(stats.std(), &[2.0; FEATURE_COUNT]);
}

#[test]
fn applies_feature_wise_z_score_without_reordering_rows() {
    let temp = TempDir::new();
    let mean = (0..FEATURE_COUNT)
        .map(|index| index as f32)
        .collect::<Vec<_>>();
    let std = vec![2.0_f32; FEATURE_COUNT];
    let stats = NormalizationStats::from_vectors(
        &temp.path().join("mu.npy"),
        &temp.path().join("sigma.npy"),
        mean.clone(),
        std,
    )
    .unwrap();
    let features = mean.iter().map(|value| value + 2.0).collect::<Vec<_>>();
    let segment = LobSegment::new("fixture.csv", features.clone(), vec![1]).unwrap();
    let mut dataset = LobDataset::new(vec![segment]);
    Normalizer::new(stats, 1e-8)
        .unwrap()
        .normalize(&mut dataset)
        .unwrap();
    assert!(
        dataset.segments()[0]
            .features()
            .iter()
            .all(|value| (*value - 1.0).abs() < 1e-6)
    );
    assert_eq!(dataset.segments()[0].labels(), &[1]);
}

#[test]
fn handles_zero_standard_deviation_with_the_specified_epsilon() {
    let temp = TempDir::new();
    let mean = vec![5.0_f32; FEATURE_COUNT];
    let stats = NormalizationStats::from_vectors(
        &temp.path().join("mu.npy"),
        &temp.path().join("sigma.npy"),
        mean.clone(),
        vec![0.0; FEATURE_COUNT],
    )
    .unwrap();
    let segment = LobSegment::new("fixture.csv", mean, vec![0]).unwrap();
    let mut dataset = LobDataset::new(vec![segment]);
    Normalizer::new(stats, 1e-8)
        .unwrap()
        .normalize(&mut dataset)
        .unwrap();
    assert!(
        dataset.segments()[0]
            .features()
            .iter()
            .all(|value| *value == 0.0)
    );
}

#[test]
fn rejects_missing_wrong_length_negative_and_nonfinite_statistics() {
    let temp = TempDir::new();
    let missing = NormalizationSettings {
        mean_path: temp.path().join("missing.npy"),
        std_path: temp.path().join("also-missing.npy"),
    };
    assert!(matches!(
        NormalizationStats::load(&missing),
        Err(AppError::NormalizationRead { .. })
    ));

    let wrong_shape = temp.path().join("wrong-shape.npy");
    write_npy_shape(&wrong_shape, &[20, 2], &[0.0_f32; 40]);
    let valid_std = temp.path().join("valid-std.npy");
    write_npy(&valid_std, &[1.0_f32; 40]);
    assert!(matches!(
        NormalizationStats::load(&NormalizationSettings {
            mean_path: wrong_shape,
            std_path: valid_std,
        }),
        Err(AppError::NormalizationNpy { .. })
    ));

    let mean_path = temp.path().join("mu.npy");
    let std_path = temp.path().join("sigma.npy");
    assert!(matches!(
        NormalizationStats::from_vectors(&mean_path, &std_path, vec![0.0; 39], vec![1.0; 40]),
        Err(AppError::NormalizationValues { .. })
    ));
    assert!(matches!(
        NormalizationStats::from_vectors(&mean_path, &std_path, vec![0.0; 40], vec![-1.0; 40]),
        Err(AppError::NormalizationValues { .. })
    ));
    let mut mean = vec![0.0; 40];
    mean[3] = f32::INFINITY;
    assert!(matches!(
        NormalizationStats::from_vectors(&mean_path, &std_path, mean, vec![1.0; 40]),
        Err(AppError::NormalizationValues { .. })
    ));
}
