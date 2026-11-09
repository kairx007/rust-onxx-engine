#![allow(dead_code)]

use std::{
    fs::{self, File},
    io::BufWriter,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

use npyz::WriterBuilder;
use rust_engine::config::Settings;

static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

pub struct TempDir(PathBuf);

impl TempDir {
    pub fn new() -> Self {
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("rust-engine-phase2-{}-{id}", std::process::id()));
        fs::create_dir_all(&path).expect("test temp directory should be created");
        Self(path)
    }

    pub fn path(&self) -> &std::path::Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

pub fn default_settings() -> Settings {
    Settings::parse(include_str!("../../configs/default.toml"))
        .expect("default settings should be valid")
}

pub fn write_csv_segment(path: &std::path::Path, rows: usize, label: u8) {
    let mut contents = String::new();
    for row in 0..rows {
        let fields = (0..=144)
            .map(|column| {
                if column == 144 {
                    label.to_string()
                } else {
                    (row * 100 + column).to_string()
                }
            })
            .collect::<Vec<_>>();
        contents.push_str(&fields.join(","));
        contents.push('\n');
    }
    fs::write(path, contents).expect("test CSV should be written");
}

pub fn write_npy<T>(path: &std::path::Path, values: &[T])
where
    T: Clone + npyz::AutoSerialize,
{
    write_npy_shape(path, &[values.len() as u64], values);
}

pub fn write_npy_shape<T>(path: &std::path::Path, shape: &[u64], values: &[T])
where
    T: Clone + npyz::AutoSerialize,
{
    let file = File::create(path).expect("test NPY file should be created");
    let mut writer = npyz::WriteOptions::new()
        .default_dtype()
        .shape(shape)
        .writer(BufWriter::new(file))
        .begin_nd()
        .expect("test NPY writer should initialize");
    writer
        .extend(values.iter().cloned())
        .expect("test NPY values should be written");
    writer.finish().expect("test NPY file should finish");
}
