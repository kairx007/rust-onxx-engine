use rust_engine::{error::AppError, runtime::Runtime};

#[test]
fn initializes_cpu_runtime_once_and_reports_provider() {
    let runtime = Runtime::initialize("cpu").expect("CPU runtime should initialize");
    assert_eq!(runtime.execution_provider(), "cpu");

    let second = Runtime::initialize("CPU").expect("runtime initialization should be reusable");
    assert_eq!(second.execution_provider(), "cpu");
}

#[test]
fn rejects_unsupported_provider_before_runtime_initialization() {
    assert!(matches!(
        Runtime::initialize("cuda"),
        Err(AppError::UnsupportedExecutionProvider(_))
    ));
}
