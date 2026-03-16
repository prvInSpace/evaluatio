use crate::{err::ValueError, stats};

#[cfg(feature = "python")]
use pyo3::prelude::*;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

#[cfg(feature = "python")]
#[pyclass]
#[derive(Debug, PartialEq)]
pub struct ConfidenceInterval {
    #[pyo3(get)]
    pub mean: f64,
    #[pyo3(get)]
    pub lower: f64,
    #[pyo3(get)]
    pub upper: f64,
}

#[cfg(not(feature = "python"))]
#[derive(Debug, PartialEq)]
pub struct ConfidenceInterval {
    pub mean: f64,
    pub lower: f64,
    pub upper: f64,
}

pub fn confidence_interval(
    x: &[f64],
    iterations: usize,
    alpha: f64,
) -> Result<ConfidenceInterval, ValueError> {
    if iterations < 1 {
        return Err(ValueError::AtLeastOneIterationRequired);
    }
    if !(0.0..=1.0).contains(&alpha) {
        return Err(ValueError::InvalidAlphaValue);
    }
    let n: usize = x.len();
    let mut bootstrap_means: Vec<f64> = (0..iterations)
        .into_par_iter()
        .map(|_| (0..n).map(|_| x[fastrand::usize(0..n)]).sum::<f64>() / (n as f64))
        .collect();

    bootstrap_means.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let lower_idx = ((alpha / 2.0) * iterations as f64).floor() as usize;
    let upper_idx = ((1.0 - alpha / 2.0) * iterations as f64).floor() as usize;

    let lower = bootstrap_means[lower_idx.min(iterations - 1)];
    let upper = bootstrap_means[upper_idx.min(iterations - 1)];

    let mean = stats::mean(x)?;
    Ok(ConfidenceInterval { mean, lower, upper })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ensure_that_it_does_what_it_should() {
        let res = confidence_interval(&[1.0], 1, 0.05).unwrap();
        let expected = ConfidenceInterval {
            mean: 1.0,
            lower: 1.0,
            upper: 1.0,
        };
        assert_eq!(res, expected)
    }

    #[test]
    fn need_at_least_one_bootstrap() {
        let res = confidence_interval(&[1.0], 0, 0.05);
        assert!(res.is_err())
    }

    #[test]
    fn word_error_rate_ci_should_not_allow_wrong_alpha_above_1() {
        let reference = vec![1.0, 1.0];
        let result = confidence_interval(&reference, 1, 1.01);
        assert!(result.is_err());
        // Should not fail
        let _ = confidence_interval(&reference, 1, 1.00);
    }

    #[test]
    fn word_error_rate_ci_should_not_allow_wrong_alpha_below_0() {
        let reference = vec![1.0, 1.0];
        let result = confidence_interval(&reference, 1, -0.01);
        assert!(result.is_err());
        // Should not fail
        let _ = confidence_interval(&reference, 1, -0.00);
    }
}
