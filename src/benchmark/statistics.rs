use serde::{Deserialize, Serialize};

/// Summary of latency samples. Percentiles use the nearest-rank method;
/// standard deviation is population standard deviation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LatencyStatistics {
    pub sample_count: usize,
    pub mean_ms: f64,
    pub standard_deviation_ms: f64,
    pub minimum_ms: f64,
    pub maximum_ms: f64,
    pub p50_ms: f64,
    pub p90_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
}

pub fn calculate_latency_statistics(samples_ms: &[f64]) -> Option<LatencyStatistics> {
    if samples_ms.is_empty()
        || samples_ms
            .iter()
            .any(|sample| !sample.is_finite() || *sample < 0.0)
    {
        return None;
    }
    let mut sorted = samples_ms.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
    let variance = sorted
        .iter()
        .map(|sample| (sample - mean).powi(2))
        .sum::<f64>()
        / sorted.len() as f64;
    Some(LatencyStatistics {
        sample_count: sorted.len(),
        mean_ms: mean,
        standard_deviation_ms: variance.sqrt(),
        minimum_ms: sorted[0],
        maximum_ms: sorted[sorted.len() - 1],
        p50_ms: percentile(&sorted, 0.50),
        p90_ms: percentile(&sorted, 0.90),
        p95_ms: percentile(&sorted, 0.95),
        p99_ms: percentile(&sorted, 0.99),
    })
}

fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    let rank = (fraction * sorted.len() as f64).ceil().max(1.0) as usize;
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

pub fn calculate_throughput(completed_windows: usize, elapsed_seconds: f64) -> Option<f64> {
    if completed_windows == 0 || !elapsed_seconds.is_finite() || elapsed_seconds <= 0.0 {
        None
    } else {
        Some(completed_windows as f64 / elapsed_seconds)
    }
}

#[cfg(test)]
mod tests {
    use super::{calculate_latency_statistics, calculate_throughput};

    #[test]
    fn calculates_nearest_rank_percentiles_and_population_deviation() {
        let stats = calculate_latency_statistics(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(stats.mean_ms, 2.5);
        assert_eq!(stats.p50_ms, 2.0);
        assert_eq!(stats.p90_ms, 4.0);
        assert_eq!(stats.p95_ms, 4.0);
        assert_eq!(stats.p99_ms, 4.0);
        assert_eq!(stats.standard_deviation_ms, (1.25_f64).sqrt());
    }

    #[test]
    fn handles_empty_invalid_samples_and_throughput_inputs() {
        assert!(calculate_latency_statistics(&[]).is_none());
        assert!(calculate_latency_statistics(&[f64::NAN]).is_none());
        assert_eq!(calculate_throughput(20, 2.0), Some(10.0));
        assert!(calculate_throughput(0, 2.0).is_none());
        assert!(calculate_throughput(2, 0.0).is_none());
    }
}
