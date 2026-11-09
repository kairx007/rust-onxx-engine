use tracing_subscriber::{EnvFilter, fmt};

use crate::error::AppError;

pub fn init(level: &str) -> Result<(), AppError> {
    let filter = EnvFilter::try_new(level).map_err(|error| AppError::Logging(error.to_string()))?;
    fmt().with_env_filter(filter).try_init().map_err(|error| {
        AppError::Logging(format!(
            "a global tracing subscriber is already installed: {error}"
        ))
    })
}
