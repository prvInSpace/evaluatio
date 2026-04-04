use crate::err::ValueError;
use rayon::prelude::*;

pub fn paired_permutation_test(
    a: &[f64],
    b: &[f64],
    iterations: usize,
    two_tailed: bool,
) -> Result<f64, ValueError> {
    if a.len() != b.len() {
        return Err(ValueError::UnequalLengths);
    }
    if a.is_empty() {
        return Err(ValueError::NotEnoughValues);
    }
    if iterations == 0 {
        return Err(ValueError::AtLeastOneIterationRequired);
    }

    let diffs: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x - y).collect();
    let observed = diffs.iter().sum::<f64>() / diffs.len() as f64;
    let n = diffs.len() as f64;

    let extreme_count: usize = (0..iterations)
        .into_par_iter()
        .map_init(
            || fastrand::Rng::new(), // independently seeded per thread
            |rng, _| {
                let perm_stat: f64 = diffs
                    .iter()
                    .map(|&d| if rng.bool() { d } else { -d })
                    .sum::<f64>()
                    / n;

                let is_extreme = if two_tailed {
                    perm_stat.abs() >= observed.abs()
                } else {
                    perm_stat >= observed
                };

                is_extreme as usize
            },
        )
        .sum();

    Ok((extreme_count + 1) as f64 / (iterations + 1) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clearly_significant() {
        // Large consistent effect — p-value should be very small
        let a = vec![10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0];
        let b = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let p = paired_permutation_test(&a, &b, 9999, true).unwrap();
        assert!(p < 0.01, "Expected p < 0.01, got {}", p);
    }

    #[test]
    fn test_clearly_not_significant() {
        // Identical series — p-value should be large (near 1.0)
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let p = paired_permutation_test(&a, &b, 9999, true).unwrap();
        assert!(p > 0.5, "Expected p > 0.5, got {}", p);
    }

    #[test]
    fn test_p_value_bounds() {
        // p-value must always be in (0, 1] — the +1 correction ensures
        // it can never be exactly 0
        let a = vec![100.0; 10];
        let b = vec![0.0; 10];
        let p = paired_permutation_test(&a, &b, 999, true).unwrap();
        assert!(p > 0.0, "p-value must be > 0");
        assert!(p <= 1.0, "p-value must be <= 1");
    }

    #[test]
    fn test_two_tailed_vs_one_tailed() {
        // Two-tailed p should be roughly double the one-tailed p
        // for a symmetric test statistic
        let a = vec![3.0, 5.0, 2.0, 4.0, 6.0, 3.0, 5.0, 2.0, 4.0, 6.0];
        let b = vec![1.0, 2.0, 1.0, 2.0, 3.0, 1.0, 2.0, 1.0, 2.0, 3.0];
        let p_two = paired_permutation_test(&a, &b, 9999, true).unwrap();
        let p_one = paired_permutation_test(&a, &b, 9999, false).unwrap();
        // Two-tailed should be larger
        assert!(p_two > p_one, "Two-tailed p should exceed one-tailed p");
        // And roughly double, with generous Monte Carlo tolerance
        assert!(
            p_two < p_one * 3.0,
            "Two-tailed p should be roughly 2x one-tailed"
        );
    }

    #[test]
    fn test_symmetry() {
        // Swapping a and b should give the same two-tailed p-value
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0];
        let p_ab = paired_permutation_test(&a, &b, 9999, true).unwrap();
        let p_ba = paired_permutation_test(&b, &a, 9999, true).unwrap();
        assert!(
            (p_ab - p_ba).abs() < 0.05,
            "Two-tailed test should be symmetric: {} vs {}",
            p_ab,
            p_ba
        );
    }

    #[test]
    fn test_minimum_p_value() {
        // With n_perms=999, minimum possible p is 1/1000 = 0.001
        let a = vec![100.0; 50];
        let b = vec![0.0; 50];
        let p = paired_permutation_test(&a, &b, 999, true).unwrap();
        assert!(
            (p - 0.001).abs() < 1e-10,
            "Expected minimum p of 0.001, got {}",
            p
        );
    }

    #[test]
    fn test_unequal_lengths_errors() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0];
        assert!(paired_permutation_test(&a, &b, 999, true).is_err());
    }

    #[test]
    fn test_not_enough_iterations() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0];
        assert!(paired_permutation_test(&a, &b, 0, true).is_err());
    }

    #[test]
    fn test_empty_input_errors() {
        let a: Vec<f64> = vec![];
        let b: Vec<f64> = vec![];
        assert!(paired_permutation_test(&a, &b, 999, true).is_err());
    }

    #[test]
    fn test_single_pair() {
        // Not good, but shouldn't cause a panic
        let a = vec![1.0];
        let b = vec![0.0];
        let result = paired_permutation_test(&a, &b, 999, true);
        assert!(result.is_ok());
    }
}
