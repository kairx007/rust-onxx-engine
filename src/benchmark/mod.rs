//! Reproducible benchmark statistics, reports, and optional process metrics.

mod comparison;
mod report;
mod resources;
mod statistics;

pub use comparison::{ComparisonReport, compare_json_reports};
pub use report::{
    BenchmarkMetadata, BenchmarkReport, BenchmarkTrial, PreprocessingStageMetrics, ReportFormat,
};
pub use resources::{ResourceMetrics, ResourceSampler};
pub use statistics::{LatencyStatistics, calculate_latency_statistics, calculate_throughput};
