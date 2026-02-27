use evaluatio_core::metrics::cer;
use pyo3::{pyfunction, PyResult};

#[pyfunction(name = "character_error_rate_per_pair")]
pub fn character_error_rate_per_pair_py(
    references: Vec<String>,
    hypotheses: Vec<String>,
) -> PyResult<Vec<f64>> {
    let left_vec: Vec<&str> = references.iter().map(|x| x.as_str()).collect();
    let right_vec: Vec<&str> = hypotheses.iter().map(|x| x.as_str()).collect();
    let result = cer::character_error_rate_per_pair(&left_vec, &right_vec);
    Ok(result)
}

#[pyfunction(name = "character_edit_distance_per_pair")]
pub fn character_edit_distance_per_pair_py(
    references: Vec<String>,
    hypotheses: Vec<String>,
) -> PyResult<Vec<usize>> {
    let left_vec: Vec<&str> = references.iter().map(|x| x.as_str()).collect();
    let right_vec: Vec<&str> = hypotheses.iter().map(|x| x.as_str()).collect();
    let result = cer::character_edit_distance_per_pair(&left_vec, &right_vec);
    Ok(result)
}

#[pyfunction(name = "character_error_rate")]
pub fn character_error_rate_py(references: Vec<String>, hypotheses: Vec<String>) -> PyResult<f64> {
    let left_vec: Vec<&str> = references.iter().map(|x| x.as_str()).collect();
    let right_vec: Vec<&str> = hypotheses.iter().map(|x| x.as_str()).collect();
    let result = cer::character_error_rate(&left_vec, &right_vec);
    Ok(result)
}
