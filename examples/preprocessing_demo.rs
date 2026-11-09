use rust_engine::{config::Settings, preprocessing::PreprocessingPipeline};

fn main() -> anyhow::Result<()> {
    let settings = Settings::load("configs/default.toml")?;
    let pipeline = PreprocessingPipeline::new(&settings)?;
    let prepared = pipeline.prepare()?;
    println!(
        "Loaded {} rows across {} segments; {} complete windows are ready.",
        prepared.total_rows(),
        prepared.segment_count(),
        prepared.windows().len()
    );
    if !prepared.windows().is_empty() {
        let sample = prepared.window(0)?;
        println!(
            "First tensor: shape={:?}, values={}, label={}",
            sample.tensor.shape(),
            sample.tensor.data().len(),
            sample.label
        );
    }
    Ok(())
}
