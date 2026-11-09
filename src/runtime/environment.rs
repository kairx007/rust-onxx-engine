use std::{fmt, sync::OnceLock};

use ort::{environment::Environment, ep, ep::ExecutionProviderDispatch};
use tracing::info;

use crate::error::AppError;

static ENVIRONMENT: OnceLock<Result<Environment, String>> = OnceLock::new();

pub struct Runtime {
    environment: Environment,
    execution_provider: &'static str,
}

impl fmt::Debug for Runtime {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Runtime")
            .field("execution_provider", &self.execution_provider)
            .finish_non_exhaustive()
    }
}

impl Runtime {
    pub fn initialize(provider: &str) -> Result<Self, AppError> {
        if !provider.eq_ignore_ascii_case("cpu") {
            return Err(AppError::UnsupportedExecutionProvider(provider.to_owned()));
        }

        let environment = ENVIRONMENT
            .get_or_init(|| {
                let cpu_provider: ExecutionProviderDispatch = ep::CPU::default().build();
                ort::init()
                    .with_name("RustLOB")
                    .with_telemetry(false)
                    .with_execution_providers([cpu_provider])
                    .build()
                    .map_err(|error| error.to_string())
            })
            .as_ref()
            .map_err(|message| AppError::RuntimeInitialization(message.clone()))?
            .clone();

        info!(
            execution_provider = "cpu",
            "ONNX Runtime environment initialized"
        );
        Ok(Self {
            environment,
            execution_provider: "cpu",
        })
    }

    pub fn execution_provider(&self) -> &'static str {
        self.execution_provider
    }

    pub(crate) fn environment(&self) -> &Environment {
        &self.environment
    }
}
