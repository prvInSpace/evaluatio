use evaluatio_core::effect_size::cohen;
use pyo3::{pyfunction, PyResult};

#[pyfunction(name = "cohens_d")]
pub fn cohens_d_py(x1: Vec<f64>, x2: Vec<f64>) -> PyResult<f64> {
    Ok(cohen::cohens_d(&x1, &x2))
}
