use evaluatio_core::metrics::bleu::{self, BLEUSufficientStats};
use pyo3::{pyfunction, PyResult};

#[pyfunction(name = "bleu_bootstrap_test")]
pub fn bleu_bootstrap_test_py(
    stats_a: Vec<BLEUSufficientStats>,
    stats_b: Vec<BLEUSufficientStats>,
    iterations: usize,
) -> PyResult<f64> {
    Ok(bleu::bleu_bootstrap_test(&stats_a, &stats_b, iterations))
}
