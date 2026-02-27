use evaluatio_core::metrics::alignment;
use pyo3::{pyfunction, Bound, PyAny, PyResult};

use crate::base::{convert_to_edit_distance_vec, EditDistanceItem};

#[pyfunction(name = "optimal_alignment")]
pub fn optimial_alignment_py(
    references: Vec<Bound<PyAny>>,
    hypotheses: Vec<Bound<PyAny>>,
) -> PyResult<Vec<alignment::Alignment>> {
    let pred: Vec<EditDistanceItem> = convert_to_edit_distance_vec(&references)?;
    let ref_: Vec<EditDistanceItem> = convert_to_edit_distance_vec(&hypotheses)?;

    let result = alignment::optimial_aligment(&pred, &ref_);
    Ok(result)
}
