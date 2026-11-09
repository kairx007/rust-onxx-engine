use std::{fs::File, io::BufReader, path::Path};

use crate::{
    config::NormalizationSettings,
    data::{FEATURE_COUNT, LobDataset},
    error::AppError,
};

#[derive(Debug, Clone, PartialEq)]
pub struct NormalizationStats {
    mean: [f32; FEATURE_COUNT],
    std: [f32; FEATURE_COUNT],
}

impl NormalizationStats {
    pub fn load(settings: &NormalizationSettings) -> Result<Self, AppError> {
        let mean = read_vector(&settings.mean_path)?;
        let std = read_vector(&settings.std_path)?;
        Self::from_vectors(&settings.mean_path, &settings.std_path, mean, std)
    }

    pub fn from_vectors(
        mean_path: &Path,
        std_path: &Path,
        mean: Vec<f32>,
        std: Vec<f32>,
    ) -> Result<Self, AppError> {
        let mean = to_fixed_vector(mean_path, mean, "mean")?;
        let std = to_fixed_vector(std_path, std, "standard deviation")?;
        for (index, value) in std.iter().enumerate() {
            if *value < 0.0 {
                return Err(AppError::NormalizationValues {
                    path: std_path.to_owned(),
                    details: format!("standard deviation at feature {index} is negative: {value}"),
                });
            }
        }
        Ok(Self { mean, std })
    }

    pub fn mean(&self) -> &[f32; FEATURE_COUNT] {
        &self.mean
    }

    pub fn std(&self) -> &[f32; FEATURE_COUNT] {
        &self.std
    }
}

fn to_fixed_vector(
    path: &Path,
    values: Vec<f32>,
    name: &str,
) -> Result<[f32; FEATURE_COUNT], AppError> {
    if values.len() != FEATURE_COUNT {
        return Err(AppError::NormalizationValues {
            path: path.to_owned(),
            details: format!(
                "{name} vector has {} values, expected {FEATURE_COUNT}",
                values.len()
            ),
        });
    }
    if let Some((index, value)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(AppError::NormalizationValues {
            path: path.to_owned(),
            details: format!("{name} value at feature {index} is not finite: {value}"),
        });
    }
    values
        .try_into()
        .map_err(|values: Vec<f32>| AppError::NormalizationValues {
            path: path.to_owned(),
            details: format!(
                "{name} vector has {} values, expected {FEATURE_COUNT}",
                values.len()
            ),
        })
}

fn read_vector(path: &Path) -> Result<Vec<f32>, AppError> {
    let file = File::open(path).map_err(|source| AppError::NormalizationRead {
        path: path.to_owned(),
        source,
    })?;
    let npy =
        npyz::NpyFile::new(BufReader::new(file)).map_err(|error| AppError::NormalizationNpy {
            path: path.to_owned(),
            details: error.to_string(),
        })?;
    if npy.shape() != [FEATURE_COUNT as u64] {
        return Err(AppError::NormalizationNpy {
            path: path.to_owned(),
            details: format!("expected shape [{FEATURE_COUNT}], found {:?}", npy.shape()),
        });
    }
    if npy.order() != npyz::Order::C {
        return Err(AppError::NormalizationNpy {
            path: path.to_owned(),
            details: "expected C-order NPY data".to_owned(),
        });
    }

    let data =
        match npy.try_data::<f32>() {
            Ok(reader) => reader.collect::<Result<Vec<_>, _>>().map_err(|error| {
                AppError::NormalizationNpy {
                    path: path.to_owned(),
                    details: error.to_string(),
                }
            })?,
            Err(npy) => npy
                .into_vec::<f64>()
                .map_err(|error| AppError::NormalizationNpy {
                    path: path.to_owned(),
                    details: error.to_string(),
                })?
                .into_iter()
                .map(|value| value as f32)
                .collect(),
        };
    Ok(data)
}

#[derive(Debug, Clone)]
pub struct Normalizer {
    stats: NormalizationStats,
    epsilon: f32,
}

impl Normalizer {
    pub fn new(stats: NormalizationStats, epsilon: f32) -> Result<Self, AppError> {
        if !epsilon.is_finite() || epsilon != 1e-8_f32 {
            return Err(AppError::ConfigValidation(
                "normalization epsilon must be finite and exactly 1e-8".to_owned(),
            ));
        }
        Ok(Self { stats, epsilon })
    }

    pub fn normalize(&self, dataset: &mut LobDataset) -> Result<(), AppError> {
        for segment in dataset.segments_mut() {
            for (row_index, row) in segment
                .features_mut()
                .chunks_exact_mut(FEATURE_COUNT)
                .enumerate()
            {
                for (feature_index, value) in row.iter_mut().enumerate() {
                    *value = (*value - self.stats.mean[feature_index])
                        / (self.stats.std[feature_index] + self.epsilon);
                    if !value.is_finite() {
                        return Err(AppError::DatasetInvariant {
                            path: segment.source().to_owned(),
                            details: format!(
                                "normalized value at row {row_index}, feature {feature_index} is not finite"
                            ),
                        });
                    }
                }
            }
        }
        Ok(())
    }
}
