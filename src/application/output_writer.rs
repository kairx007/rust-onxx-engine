use std::{
    fs::{self, File, OpenOptions},
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
};

use serde::Serialize;

use crate::{error::AppError, inference::Prediction};

static NEXT_TEMP_ID: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Serialize)]
pub struct PredictionRow {
    pub window_index: usize,
    pub source_file: String,
    pub segment_index: usize,
    pub start_row: usize,
    pub true_class: u8,
    pub predicted_class: usize,
    pub predicted_direction: &'static str,
    pub down_score: f32,
    pub stationary_score: f32,
    pub up_score: f32,
}

impl PredictionRow {
    pub fn new(
        window_index: usize,
        source_file: &Path,
        segment_index: usize,
        start_row: usize,
        true_class: u8,
        prediction: Prediction,
    ) -> Self {
        let scores = prediction.scores();
        let predicted_direction = match prediction.direction() {
            crate::inference::Direction::Down => "Down",
            crate::inference::Direction::Stationary => "Stationary",
            crate::inference::Direction::Up => "Up",
        };
        Self {
            window_index,
            source_file: source_file.display().to_string(),
            segment_index,
            start_row,
            true_class,
            predicted_class: prediction.class_index(),
            predicted_direction,
            down_score: scores[0],
            stationary_score: scores[1],
            up_score: scores[2],
        }
    }
}

#[derive(Debug)]
pub struct PredictionWriter {
    writer: Option<csv::Writer<File>>,
    destination: PathBuf,
    temporary: PathBuf,
    overwrite: bool,
    published: bool,
}

impl PredictionWriter {
    pub fn create(path: impl AsRef<Path>, overwrite: bool) -> Result<Self, AppError> {
        let destination = path.as_ref().to_owned();
        if destination.exists() && !overwrite {
            return Err(AppError::OutputExists(destination));
        }
        let parent = destination
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        fs::create_dir_all(parent).map_err(|source| AppError::OutputCreate {
            path: parent.to_owned(),
            source,
        })?;

        let file_name = destination
            .file_name()
            .unwrap_or_default()
            .to_string_lossy();
        let (temporary, file) = create_temp_file(parent, &file_name, &destination)?;
        let writer = csv::WriterBuilder::new().from_writer(file);
        Ok(Self {
            writer: Some(writer),
            destination,
            temporary,
            overwrite,
            published: false,
        })
    }

    pub fn write(&mut self, row: &PredictionRow) -> Result<(), AppError> {
        let writer = self.writer.as_mut().ok_or_else(|| AppError::OutputWrite {
            path: self.temporary.clone(),
            source: std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "prediction writer has already been finalized",
            ),
        })?;
        writer.serialize(row).map_err(|source| AppError::OutputCsv {
            path: self.temporary.clone(),
            source,
        })
    }

    pub fn finish(mut self) -> Result<(), AppError> {
        let Some(mut writer) = self.writer.take() else {
            return Err(AppError::OutputWrite {
                path: self.temporary.clone(),
                source: std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "prediction writer has already been finalized",
                ),
            });
        };
        writer.flush().map_err(|source| AppError::OutputWrite {
            path: self.temporary.clone(),
            source,
        })?;
        let file = writer.into_inner().map_err(|error| AppError::OutputWrite {
            path: self.temporary.clone(),
            source: error.into_error(),
        })?;
        file.sync_all().map_err(|source| AppError::OutputWrite {
            path: self.temporary.clone(),
            source,
        })?;
        drop(file);

        if self.overwrite {
            fs::rename(&self.temporary, &self.destination).map_err(|source| {
                AppError::OutputPublish {
                    path: self.destination.clone(),
                    source,
                }
            })?;
        } else {
            fs::hard_link(&self.temporary, &self.destination).map_err(|source| {
                if source.kind() == std::io::ErrorKind::AlreadyExists {
                    AppError::OutputExists(self.destination.clone())
                } else {
                    AppError::OutputPublish {
                        path: self.destination.clone(),
                        source,
                    }
                }
            })?;
            fs::remove_file(&self.temporary).map_err(|source| AppError::OutputPublish {
                path: self.temporary.clone(),
                source,
            })?;
        }
        self.published = true;
        Ok(())
    }
}

impl Drop for PredictionWriter {
    fn drop(&mut self) {
        if !self.published {
            let _ = fs::remove_file(&self.temporary);
        }
    }
}

fn create_temp_file(
    parent: &Path,
    file_name: &str,
    destination: &Path,
) -> Result<(PathBuf, File), AppError> {
    for _ in 0..100 {
        let id = NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed);
        let temporary = parent.join(format!(".{file_name}.{}.{}.tmp", std::process::id(), id));
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
        {
            Ok(file) => return Ok((temporary, file)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(source) => {
                return Err(AppError::OutputCreate {
                    path: destination.to_owned(),
                    source,
                });
            }
        }
    }
    Err(AppError::OutputCreate {
        path: destination.to_owned(),
        source: std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            "could not allocate a unique temporary output file",
        ),
    })
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::PathBuf,
        sync::atomic::{AtomicUsize, Ordering},
    };

    use crate::{error::AppError, inference::Prediction};

    use super::{PredictionRow, PredictionWriter};

    static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

    fn temp_output() -> PathBuf {
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "rust-engine-cli-{}-{id}/predictions.csv",
            std::process::id()
        ))
    }

    #[test]
    fn writes_stable_prediction_csv_schema_and_scores() {
        let path = temp_output();
        let mut writer = PredictionWriter::create(&path, false).unwrap();
        writer
            .write(&PredictionRow::new(
                4,
                PathBuf::from("segment.csv").as_path(),
                2,
                100,
                1,
                Prediction::from_scores([0.1, 0.2, 0.7]).unwrap(),
            ))
            .unwrap();
        writer.finish().unwrap();
        let contents = fs::read_to_string(&path).unwrap();
        assert!(contents.starts_with("window_index,source_file,segment_index,start_row,true_class,predicted_class,predicted_direction,down_score,stationary_score,up_score"));
        assert!(contents.contains("4,segment.csv,2,100,1,2,Up,0.1,0.2,0.7"));
        let _ = fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn refuses_to_overwrite_by_default_and_preserves_existing_output() {
        let path = temp_output();
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, "keep me").unwrap();
        assert!(matches!(
            PredictionWriter::create(&path, false),
            Err(AppError::OutputExists(_))
        ));
        assert_eq!(fs::read_to_string(&path).unwrap(), "keep me");
        let _ = fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn dropping_an_unfinished_writer_removes_temporary_output() {
        let path = temp_output();
        let parent = path.parent().unwrap();
        {
            let mut writer = PredictionWriter::create(&path, false).unwrap();
            writer
                .write(&PredictionRow::new(
                    0,
                    PathBuf::from("segment.csv").as_path(),
                    0,
                    0,
                    0,
                    Prediction::from_scores([1.0, 0.0, 0.0]).unwrap(),
                ))
                .unwrap();
        }
        assert!(!path.exists());
        assert_eq!(fs::read_dir(parent).unwrap().count(), 0);
        let _ = fs::remove_dir_all(parent);
    }

    #[test]
    fn overwrite_publishes_a_complete_replacement() {
        let path = temp_output();
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, "old output").unwrap();
        let mut writer = PredictionWriter::create(&path, true).unwrap();
        writer
            .write(&PredictionRow::new(
                0,
                PathBuf::from("segment.csv").as_path(),
                0,
                0,
                0,
                Prediction::from_scores([1.0, 0.0, 0.0]).unwrap(),
            ))
            .unwrap();
        writer.finish().unwrap();
        let output = fs::read_to_string(&path).unwrap();
        assert!(output.starts_with("window_index,"));
        assert!(!output.contains("old output"));
        let _ = fs::remove_dir_all(path.parent().unwrap());
    }
}
