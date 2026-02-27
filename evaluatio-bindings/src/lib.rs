use pyo3::prelude::*;
mod base;
mod inference;
use inference::*;
mod effect_size;
use effect_size::*;
mod metrics;
use metrics::*;

/// A Python module implemented in Rust.
#[pymodule]
fn _bindings(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Character error rate
    m.add_function(wrap_pyfunction!(
        cer::character_edit_distance_per_pair_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(cer::character_error_rate_py, m)?)?;
    m.add_function(wrap_pyfunction!(cer::character_error_rate_per_pair_py, m)?)?;

    // Cohen's d
    m.add_function(wrap_pyfunction!(cohen::cohens_d_independent_py, m)?)?;
    m.add_function(wrap_pyfunction!(cohen::cohens_d_paired_py, m)?)?;

    // Confidence interval
    m.add_function(wrap_pyfunction!(ci::confidence_interval_py, m)?)?;
    m.add_class::<evaluatio_core::inference::ci::ConfidenceInterval>()?;

    // Match error rate
    //m.add_function(wrap_pyfunction!(mer::match_error_rate_per_pair_py, m)?)?;
    //m.add_function(wrap_pyfunction!(mer::hits_per_pair_py, m)?)?;

    // Optimal alignment
    m.add_function(wrap_pyfunction!(alignment::optimial_alignment_py, m)?)?;
    m.add_class::<evaluatio_core::metrics::alignment::Alignment>()?;

    // Paired Bootstrap Test
    m.add_function(wrap_pyfunction!(bootstrap::paired_bootstrap_test_py, m)?)?;

    // Points of interest error rate
    m.add_function(wrap_pyfunction!(pier::poi_edit_distance_py, m)?)?;
    m.add_function(wrap_pyfunction!(pier::poi_error_rate_py, m)?)?;

    // Universal error rate
    m.add_function(wrap_pyfunction!(
        uer::universal_edit_distance_per_pair_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(uer::universal_error_rate_per_pair_py, m)?)?;

    // Word error rate
    m.add_function(wrap_pyfunction!(wer::word_edit_distance_per_pair_py, m)?)?;
    m.add_function(wrap_pyfunction!(wer::word_error_rate_py, m)?)?;
    m.add_function(wrap_pyfunction!(wer::word_error_rate_ci_py, m)?)?;
    m.add_function(wrap_pyfunction!(wer::word_error_rate_per_pair_py, m)?)?;

    Ok(())
}
