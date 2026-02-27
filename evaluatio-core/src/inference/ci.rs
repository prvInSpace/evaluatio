use crate::stats;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pyclass]
pub struct ConfidenceInterval {
    #[pyo3(get)]
    pub mean: f64,
    #[pyo3(get)]
    pub lower: f64,
    #[pyo3(get)]
    pub upper: f64,
}

#[cfg(not(feature = "python"))]
pub struct ConfidenceInterval {
    pub mean: f64,
    pub lower: f64,
    pub upper: f64,
}

pub fn confidence_interval(x: &[f64], iterations: usize, alpha: f64) -> ConfidenceInterval {
    let n = x.len();
    let mut bootstrap_means = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let sample_mean = (0..n).map(|_| x[fastrand::usize(0..n)]).sum::<f64>() / (n as f64);
        bootstrap_means.push(sample_mean);
    }

    let lower_idx = ((alpha / 2.0) * iterations as f64).floor() as usize;
    let upper_idx = ((1.0 - alpha / 2.0) * iterations as f64).floor() as usize;

    let lower = bootstrap_means[lower_idx.min(iterations - 1)];
    let upper = bootstrap_means[upper_idx.min(iterations - 1)];

    let mean = stats::mean(x);
    ConfidenceInterval { mean, lower, upper }
}
