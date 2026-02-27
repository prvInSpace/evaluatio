use evaluatio_core::inference::bootstrap;
use pyo3::{pyfunction, PyResult};

#[pyfunction(name = "paired_bootstrap_test")]
pub fn paired_bootstrap_test_py(x1: Vec<f64>, x2: Vec<f64>, iterations: usize) -> PyResult<f64> {
    Ok(bootstrap::paired_bootstrap_test(&x1, &x2, iterations))
}
