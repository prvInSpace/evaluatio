use evaluatio_core::effect_size::cohen;
use pyo3::{pyfunction, PyResult};

#[pyfunction(name = "cohens_d_independent")]
pub fn cohens_d_independent_py(x1: Vec<f64>, x2: Vec<f64>) -> PyResult<f64> {
    Ok(cohen::cohens_d_independent(&x1, &x2))
}

#[pyfunction(name = "cohens_d_paired")]
pub fn cohens_d_paired_py(x1: Vec<f64>, x2: Vec<f64>) -> PyResult<f64> {
    Ok(cohen::cohens_d_paired(&x1, &x2))
}