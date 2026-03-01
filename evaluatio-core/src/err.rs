#[cfg(feature = "python")]
use pyo3::PyErr;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ValueError {
    #[error("variance requires at least two values")]
    NotEnoughValues,

    #[error("input arrays must have equal length")]
    UnequalLengths,
}

#[cfg(feature = "python")]
impl From<ValueError> for PyErr {
    fn from(err: ValueError) -> PyErr {
        pyo3::exceptions::PyValueError::new_err(err.to_string())
    }
}
