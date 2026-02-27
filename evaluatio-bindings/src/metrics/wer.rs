use evaluatio_core::{inference::ci::ConfidenceInterval, metrics::wer};
use pyo3::{pyfunction, PyResult};

#[pyfunction(name = "word_error_rate_per_pair")]
pub fn word_error_rate_per_pair_py(
    references: Vec<String>,
    hypotheses: Vec<String>,
) -> PyResult<Vec<f64>> {
    let left_vec: Vec<&str> = references.iter().map(|x| x.as_str()).collect();
    let right_vec: Vec<&str> = hypotheses.iter().map(|x| x.as_str()).collect();
    let result = wer::word_error_rate_per_pair(&left_vec, &right_vec);
    Ok(result)
}

#[pyfunction(name = "word_edit_distance_per_pair")]
pub fn word_edit_distance_per_pair_py(
    references: Vec<String>,
    hypotheses: Vec<String>,
) -> PyResult<Vec<usize>> {
    let left_vec: Vec<&str> = references.iter().map(|x| x.as_str()).collect();
    let right_vec: Vec<&str> = hypotheses.iter().map(|x| x.as_str()).collect();
    let result = wer::word_edit_distance_per_pair(&left_vec, &right_vec);
    Ok(result)
}

#[pyfunction(name = "word_error_rate")]
pub fn word_error_rate_py(references: Vec<String>, hypotheses: Vec<String>) -> PyResult<f64> {
    let left_vec: Vec<&str> = references.iter().map(|x| x.as_str()).collect();
    let right_vec: Vec<&str> = hypotheses.iter().map(|x| x.as_str()).collect();
    let result = wer::word_error_rate(&left_vec, &right_vec);
    Ok(result)
}

#[pyfunction(name = "word_error_rate_ci")]
pub fn word_error_rate_ci_py(
    references: Vec<String>,
    hypotheses: Vec<String>,
    iterations: usize,
    alpha: f64,
) -> PyResult<ConfidenceInterval> {
    let left_vec: Vec<&str> = references.iter().map(|x| x.as_str()).collect();
    let right_vec: Vec<&str> = hypotheses.iter().map(|x| x.as_str()).collect();
    let result = wer::word_error_rate_ci(&left_vec, &right_vec, iterations, alpha);
    Ok(result)
}
