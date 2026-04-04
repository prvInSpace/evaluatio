use evaluatio_core::{
    inference::ci::ConfidenceInterval,
    metrics::bleu::{self, BLEUSufficientStats},
};
use pyo3::{pyfunction, PyResult};

#[pyfunction(name = "bleu_bootstrap_test")]
pub fn bleu_bootstrap_test_py(
    stats_a: Vec<BLEUSufficientStats>,
    stats_b: Vec<BLEUSufficientStats>,
    iterations: usize,
) -> PyResult<f64> {
    Ok(bleu::bleu_bootstrap_test(&stats_a, &stats_b, iterations))
}

#[pyfunction(name = "bleu_ci")]
pub fn bleu_ci_py(
    stats: Vec<BLEUSufficientStats>,
    iterations: usize,
    alpha: f64,
) -> PyResult<ConfidenceInterval> {
    Ok(bleu::bleu_ci(&stats, iterations, alpha)?)
}
