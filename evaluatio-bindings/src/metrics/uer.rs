use evaluatio_core::metrics::uer;
use pyo3::{prelude::*, types::PyList};

use crate::base::{convert_to_nested_edit_distance_item_vec, EditDistanceItem};

#[pyfunction(name = "universal_error_rate_per_pair")]
pub fn universal_error_rate_per_pair_py(
    references: &Bound<PyList>,
    hypotheses: &Bound<PyList>,
) -> PyResult<Vec<f64>> {
    // Create vectors to store the converted data
    let pred_vecs: Vec<Vec<EditDistanceItem>> =
        convert_to_nested_edit_distance_item_vec(references)?;
    let ref_vecs: Vec<Vec<EditDistanceItem>> =
        convert_to_nested_edit_distance_item_vec(hypotheses)?;

    // Create the vectors of hypotheses to vectors that the edit_distance function expects
    let pred_vec_refs: Vec<&Vec<EditDistanceItem>> = pred_vecs.iter().collect();
    let ref_vec_refs: Vec<&Vec<EditDistanceItem>> = ref_vecs.iter().collect();

    // Call a modified edit_distance implementation that works with EditDistanceItem
    let result = uer::universal_error_rate_per_pair(&pred_vec_refs, &ref_vec_refs);

    Ok(result)
}

#[pyfunction(name = "universal_edit_distance")]
pub fn universal_edit_distance_per_pair_py(
    references: &Bound<PyList>,
    hypotheses: &Bound<PyList>,
) -> PyResult<Vec<usize>> {
    // Create vectors to store the converted data
    let pred_vecs: Vec<Vec<EditDistanceItem>> =
        convert_to_nested_edit_distance_item_vec(references)?;
    let ref_vecs: Vec<Vec<EditDistanceItem>> =
        convert_to_nested_edit_distance_item_vec(hypotheses)?;

    // Create the vectors of hypotheses to vectors that the edit_distance function expects
    let pred_vec_refs: Vec<&Vec<EditDistanceItem>> = pred_vecs.iter().collect();
    let ref_vec_refs: Vec<&Vec<EditDistanceItem>> = ref_vecs.iter().collect();

    // Call a modified edit_distance implementation that works with EditDistanceItem
    let result = uer::universal_edit_distance_per_pair(&pred_vec_refs, &ref_vec_refs);

    Ok(result)
}
