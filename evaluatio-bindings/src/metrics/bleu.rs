use evaluatio_core::metrics::bleu::{self, BLEUSufficientStats};
use pyo3::{pyfunction, PyResult};

#[pyfunction(name = "bootstrap_bleu")]
pub fn bootstrap_bleu_py(
    stats_a: Vec<BLEUSufficientStats>,
    stats_b: Vec<BLEUSufficientStats>,
    iterations: usize,
) -> PyResult<f64> {
    Ok(bleu::bootstrap_bleu(&stats_a, &stats_b, iterations))
}
