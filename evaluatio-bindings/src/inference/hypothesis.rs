use evaluatio_core::inference::hypothesis;
use pyo3::{pyfunction, PyResult};

#[pyfunction(name = "paired_bootstrap_test")]
pub fn paired_bootstrap_test_py(x1: Vec<f64>, x2: Vec<f64>, iterations: usize) -> PyResult<f64> {
    Ok(hypothesis::paired_bootstrap_test(&x1, &x2, iterations)?)
}

#[pyfunction(name = "paired_permutation_test")]
pub fn paired_permutation_test_py(
    x1: Vec<f64>,
    x2: Vec<f64>,
    iterations: usize,
    two_tailed: bool,
) -> PyResult<f64> {
    Ok(hypothesis::paired_permutation_test(
        &x1, &x2, iterations, two_tailed,
    )?)
}
