use evaluatio_core::metrics::wer;
use pyo3::{pyfunction, PyResult};

#[pyfunction(name = "word_error_rate_per_pair")]
pub fn word_error_rate_per_pair_py(
    predictions: Vec<String>,
    references: Vec<String>,
) -> PyResult<Vec<f64>> {
    let left_vec: Vec<&str> = predictions.iter().map(|x| x.as_str()).collect();
    let right_vec: Vec<&str> = references.iter().map(|x| x.as_str()).collect();
    let result = wer::word_error_rate_per_pair(&left_vec, &right_vec);
    Ok(result)
}

#[pyfunction(name = "word_edit_distance_per_pair")]
pub fn word_edit_distance_per_pair_py(
    predictions: Vec<String>,
    references: Vec<String>,
) -> PyResult<Vec<usize>> {
    let left_vec: Vec<&str> = predictions.iter().map(|x| x.as_str()).collect();
    let right_vec: Vec<&str> = references.iter().map(|x| x.as_str()).collect();
    let result = wer::word_edit_distance_per_pair(&left_vec, &right_vec);
    Ok(result)
}

#[pyfunction(name = "word_error_rate")]
pub fn word_error_rate_py(predictions: Vec<String>, references: Vec<String>) -> PyResult<f64> {
    let left_vec: Vec<&str> = predictions.iter().map(|x| x.as_str()).collect();
    let right_vec: Vec<&str> = references.iter().map(|x| x.as_str()).collect();
    let result = wer::word_error_rate(&left_vec, &right_vec);
    Ok(result)
}
