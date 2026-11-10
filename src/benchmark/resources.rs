use std::{
    process::Command,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::Duration,
};

use serde::{Deserialize, Serialize};

/// Process RSS measurements sampled by the platform `ps` utility. `None` means
/// this platform or process environment did not provide a usable measurement.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourceMetrics {
    pub baseline_rss_bytes: Option<u64>,
    pub peak_rss_bytes: Option<u64>,
    pub average_rss_bytes: Option<u64>,
    pub samples: usize,
    pub average_cpu_percent: Option<f64>,
    pub peak_cpu_percent: Option<f64>,
    pub method: String,
}

#[derive(Debug)]
pub struct ResourceSampler {
    stop: Arc<AtomicBool>,
    worker: Option<thread::JoinHandle<ResourceMetrics>>,
}

impl ResourceSampler {
    pub fn start(interval: Duration) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = Arc::clone(&stop);
        let pid = std::process::id();
        let worker = thread::spawn(move || {
            let baseline = read_metrics(pid).map(|metrics| metrics.0);
            let mut rss_samples = Vec::new();
            let mut cpu_samples = Vec::new();
            while !worker_stop.load(Ordering::Relaxed) {
                if let Some((rss, cpu)) = read_metrics(pid) {
                    if let Some(rss) = rss {
                        rss_samples.push(rss);
                    }
                    if let Some(cpu) = cpu {
                        cpu_samples.push(cpu);
                    }
                }
                thread::sleep(interval);
            }
            let peak = rss_samples.iter().copied().max();
            let average = (!rss_samples.is_empty())
                .then(|| rss_samples.iter().sum::<u64>() / rss_samples.len() as u64);
            let cpu_peak = cpu_samples.iter().copied().reduce(f64::max);
            let cpu_average = (!cpu_samples.is_empty())
                .then(|| cpu_samples.iter().sum::<f64>() / cpu_samples.len() as f64);
            ResourceMetrics {
                baseline_rss_bytes: baseline.flatten(),
                peak_rss_bytes: peak,
                average_rss_bytes: average,
                samples: rss_samples.len(),
                average_cpu_percent: cpu_average,
                peak_cpu_percent: cpu_peak,
                method: "ps sampled process RSS and CPU; sample collection runs concurrently"
                    .to_owned(),
            }
        });
        Self {
            stop,
            worker: Some(worker),
        }
    }

    pub fn finish(mut self) -> ResourceMetrics {
        self.stop.store(true, Ordering::Relaxed);
        self.worker
            .take()
            .expect("resource sampler worker exists")
            .join()
            .unwrap_or(ResourceMetrics {
                baseline_rss_bytes: None,
                peak_rss_bytes: None,
                average_rss_bytes: None,
                samples: 0,
                average_cpu_percent: None,
                peak_cpu_percent: None,
                method: "resource sampler failed".to_owned(),
            })
    }
}

fn read_metrics(pid: u32) -> Option<(Option<u64>, Option<f64>)> {
    let output = Command::new("ps")
        .args(["-o", "rss=,%cpu=", "-p", &pid.to_string()])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let output = String::from_utf8(output.stdout).ok()?;
    let mut fields = output.split_whitespace();
    let rss = fields.next()?.parse::<u64>().ok()?.saturating_mul(1024);
    let cpu = fields.next().and_then(|value| value.parse::<f64>().ok());
    Some((Some(rss), cpu))
}
