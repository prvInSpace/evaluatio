use crate::{err::ValueError, stats};
use fastrand;

pub fn paired_bootstrap_test(x1: &[f64], x2: &[f64], iterations: usize) -> Result<f64, ValueError> {
    if x1.len() != x2.len() {
        return Err(ValueError::UnequalLengths);
    }
    if x1.is_empty() {
        return Err(ValueError::NotEnoughValues);
    }
    if iterations == 0 {
        return Err(ValueError::AtLeastOneIterationRequired);
    }
    let diffs: Vec<f64> = x1.iter().zip(x2).map(|(a, b)| a - b).collect();
    let n = diffs.len();
    let diff_mean = stats::mean(&diffs)?;
    let diffs_centered: Vec<f64> = diffs.iter().map(|di| di - diff_mean).collect();
    let t_obs = diff_mean.abs();

    let mut extreme_or_equal = 0usize;
    for _ in 0..iterations {
        let sample_mean = (0..n)
            .map(|_| diffs_centered[fastrand::usize(0..n)])
            .sum::<f64>()
            / (n as f64);

        if sample_mean.abs() >= t_obs {
            extreme_or_equal += 1
        }
    }

    Ok((extreme_or_equal as f64 + 1.0) / (iterations as f64 + 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ensure_that_it_does_what_it_should() {
        let res = paired_bootstrap_test(&[1.0], &[1.0], 1).unwrap();
        assert_eq!(res, 1.0);
    }

    #[test]
    fn need_at_least_one_bootstrap() {
        let res = paired_bootstrap_test(&[1.0], &[1.0], 0);
        assert!(res.is_err())
    }

        #[test]
    fn lists_should_be_of_equal_lengths() {
        let res = paired_bootstrap_test(&[1.0], &[1.0, 2.0], 1);
        assert!(res.is_err())
    }

    #[test]
    fn lists_require_at_least_one_value() {
        let res = paired_bootstrap_test(&[], &[], 1);
        assert!(res.is_err())
    }
}
