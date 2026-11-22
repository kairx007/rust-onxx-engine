use std::{fmt, str::FromStr, sync::OnceLock};

use ort::{
    environment::Environment,
    ep,
    ep::ExecutionProviderDispatch,
    session::builder::{AutoDevicePolicy, GraphOptimizationLevel, SessionBuilder},
};
use tracing::info;

use crate::{config::RuntimeSettings, error::AppError};

static ENVIRONMENT: OnceLock<Result<Environment, String>> = OnceLock::new();

/// ONNX Runtime graph optimization level, as exposed in configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphOptimization {
    Disable,
    Level1,
    Level2,
    Level3,
    All,
}

impl GraphOptimization {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Disable => "disable",
            Self::Level1 => "level1",
            Self::Level2 => "level2",
            Self::Level3 => "level3",
            Self::All => "all",
        }
    }

    fn to_ort(self) -> GraphOptimizationLevel {
        match self {
            Self::Disable => GraphOptimizationLevel::Disable,
            Self::Level1 => GraphOptimizationLevel::Level1,
            Self::Level2 => GraphOptimizationLevel::Level2,
            Self::Level3 => GraphOptimizationLevel::Level3,
            Self::All => GraphOptimizationLevel::All,
        }
    }
}

impl FromStr for GraphOptimization {
    type Err = AppError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "disable" => Ok(Self::Disable),
            "level1" => Ok(Self::Level1),
            "level2" => Ok(Self::Level2),
            "level3" => Ok(Self::Level3),
            "all" => Ok(Self::All),
            other => Err(AppError::ConfigValidation(format!(
                "unsupported runtime.graph_optimization '{other}' (expected disable, level1, level2, level3, or all)"
            ))),
        }
    }
}

/// ONNX Runtime automatic device-selection policy.
///
/// The `ort` crate (api-22+) sets `MaxEfficiency` on every new session builder.
/// On Apple silicon that lets ONNX Runtime place graph partitions on the CoreML
/// execution provider (GPU / Neural Engine, reduced precision) even though only
/// the CPU provider was registered. `Cpu` restores genuine CPU-only execution.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DevicePolicy {
    /// `AutoDevicePolicy::PreferCPU`: all nodes on the CPU execution provider.
    #[default]
    Cpu,
    /// `AutoDevicePolicy::MaxEfficiency`: the `ort` crate's implicit default.
    AutoEfficiency,
}

impl DevicePolicy {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::AutoEfficiency => "auto-efficiency",
        }
    }

    fn to_ort(self) -> AutoDevicePolicy {
        match self {
            Self::Cpu => AutoDevicePolicy::PreferCPU,
            Self::AutoEfficiency => AutoDevicePolicy::MaxEfficiency,
        }
    }
}

impl FromStr for DevicePolicy {
    type Err = AppError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "cpu" => Ok(Self::Cpu),
            "auto-efficiency" => Ok(Self::AutoEfficiency),
            other => Err(AppError::ConfigValidation(format!(
                "unsupported runtime.device_policy '{other}' (expected cpu or auto-efficiency)"
            ))),
        }
    }
}

/// Per-session ONNX Runtime options. `None` keeps the ONNX Runtime default.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SessionOptions {
    pub intra_op_threads: Option<usize>,
    pub inter_op_threads: Option<usize>,
    pub graph_optimization: Option<GraphOptimization>,
    pub intra_op_spinning: Option<bool>,
    pub device_policy: DevicePolicy,
}

impl SessionOptions {
    pub fn from_settings(settings: &RuntimeSettings) -> Result<Self, AppError> {
        Ok(Self {
            intra_op_threads: settings.intra_op_threads,
            inter_op_threads: settings.inter_op_threads,
            graph_optimization: settings
                .graph_optimization
                .as_deref()
                .map(str::parse)
                .transpose()?,
            intra_op_spinning: settings.intra_op_spinning,
            device_policy: settings
                .device_policy
                .as_deref()
                .map(str::parse)
                .transpose()?
                .unwrap_or_default(),
        })
    }

    pub(crate) fn apply(&self, builder: SessionBuilder) -> Result<SessionBuilder, ort::Error> {
        let mut builder = builder
            .with_auto_device(self.device_policy.to_ort())
            .map_err(erase_builder)?;
        if let Some(threads) = self.intra_op_threads {
            builder = builder.with_intra_threads(threads).map_err(erase_builder)?;
        }
        if let Some(threads) = self.inter_op_threads {
            builder = builder.with_inter_threads(threads).map_err(erase_builder)?;
        }
        if let Some(level) = self.graph_optimization {
            builder = builder
                .with_optimization_level(level.to_ort())
                .map_err(erase_builder)?;
        }
        if let Some(spinning) = self.intra_op_spinning {
            builder = builder
                .with_intra_op_spinning(spinning)
                .map_err(erase_builder)?;
        }
        Ok(builder)
    }

    /// Human-readable description recorded in benchmark reports.
    pub fn describe(&self) -> String {
        fn or_default<T: ToString>(value: Option<T>) -> String {
            value.map_or_else(|| "ort-default".to_owned(), |value| value.to_string())
        }
        format!(
            "intra_op_threads={}; inter_op_threads={}; graph_optimization={}; intra_op_spinning={}; device_policy={}",
            or_default(self.intra_op_threads),
            or_default(self.inter_op_threads),
            or_default(self.graph_optimization.map(GraphOptimization::as_str)),
            or_default(self.intra_op_spinning),
            self.device_policy.as_str(),
        )
    }
}

fn erase_builder(error: ort::Error<SessionBuilder>) -> ort::Error {
    error.into()
}

pub struct Runtime {
    environment: Environment,
    execution_provider: &'static str,
    session_options: SessionOptions,
}

impl fmt::Debug for Runtime {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Runtime")
            .field("execution_provider", &self.execution_provider)
            .field("session_options", &self.session_options)
            .finish_non_exhaustive()
    }
}

impl Runtime {
    /// Initializes the CPU runtime with ONNX Runtime default session options.
    pub fn initialize(provider: &str) -> Result<Self, AppError> {
        Self::initialize_with(provider, SessionOptions::default())
    }

    /// Initializes the runtime using the provider and session options from configuration.
    pub fn from_settings(settings: &RuntimeSettings) -> Result<Self, AppError> {
        Self::initialize_with(
            &settings.execution_provider,
            SessionOptions::from_settings(settings)?,
        )
    }

    pub fn initialize_with(
        provider: &str,
        session_options: SessionOptions,
    ) -> Result<Self, AppError> {
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
            session_options = %session_options.describe(),
            "ONNX Runtime environment initialized"
        );
        Ok(Self {
            environment,
            execution_provider: "cpu",
            session_options,
        })
    }

    pub fn execution_provider(&self) -> &'static str {
        self.execution_provider
    }

    pub fn session_options(&self) -> &SessionOptions {
        &self.session_options
    }

    /// Build information reported by the linked ONNX Runtime library.
    pub fn build_info(&self) -> &'static str {
        ort::info()
    }

    pub(crate) fn environment(&self) -> &Environment {
        &self.environment
    }
}

#[cfg(test)]
mod tests {
    use super::{DevicePolicy, GraphOptimization, SessionOptions};
    use crate::config::Settings;

    #[test]
    fn default_session_options_force_cpu_only_device_policy() {
        let settings = Settings::parse(include_str!("../../configs/default.toml")).unwrap();
        let options = SessionOptions::from_settings(&settings.runtime).unwrap();
        // Regression guard: the ort crate's implicit MaxEfficiency policy offloads to CoreML on macOS.
        assert_eq!(options.device_policy, DevicePolicy::Cpu);
        assert!(options.describe().contains("device_policy=cpu"));
        assert!(options.describe().contains("intra_op_threads=ort-default"));
    }

    #[test]
    fn parses_runtime_option_values_and_rejects_unknown_ones() {
        assert_eq!(
            "ALL".parse::<GraphOptimization>().unwrap(),
            GraphOptimization::All
        );
        assert_eq!(
            "auto-efficiency".parse::<DevicePolicy>().unwrap(),
            DevicePolicy::AutoEfficiency
        );
        assert!("fastest".parse::<GraphOptimization>().is_err());
        assert!("npu".parse::<DevicePolicy>().is_err());
    }

    #[test]
    fn configured_threads_and_policy_are_described() {
        let mut settings = Settings::parse(include_str!("../../configs/default.toml")).unwrap();
        settings.runtime.intra_op_threads = Some(1);
        settings.runtime.graph_optimization = Some("disable".to_owned());
        settings.runtime.device_policy = Some("auto-efficiency".to_owned());
        let described = SessionOptions::from_settings(&settings.runtime)
            .unwrap()
            .describe();
        assert_eq!(
            described,
            "intra_op_threads=1; inter_op_threads=ort-default; graph_optimization=disable; intra_op_spinning=ort-default; device_policy=auto-efficiency"
        );
        settings.runtime.intra_op_threads = Some(0);
        assert!(settings.validate().is_err());
    }
}
