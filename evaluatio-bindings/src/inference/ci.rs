use evaluatio_core::inference::ci::{self as ci, ConfidenceInterval};
use pyo3::{pyfunction, PyResult};

#[pyfunction(name = "confidence_interval")]
pub fn confidence_interval_py(
    x1: Vec<f64>,
    iterations: usize,
    alpha: f64,
) -> PyResult<ConfidenceInterval> {
    Ok(ci::confidence_interval(&x1, iterations, alpha))
}
