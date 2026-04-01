use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};

use crate::{err::ValueError, inference::ci::ConfidenceInterval};

/// Universally applicable error rates and distances
pub fn universal_error_rate_per_pair<T: PartialEq>(
    references: &[&[T]],
    hypotheses: &[&[T]],
) -> Result<Vec<f64>, ValueError> {
    if references.len() != hypotheses.len() {
        return Err(ValueError::UnequalLengths);
    }
    Ok(universal_edit_distance_per_pair(references, hypotheses)?
        .iter()
        .zip(references)
        .map(|(distance, reference)| *distance as f64 / reference.len() as f64)
        .collect())
}

pub fn universal_edit_distance_per_pair<T: PartialEq>(
    references: &[&[T]],
    hypotheses: &[&[T]],
) -> Result<Vec<usize>, ValueError> {
    if references.len() != hypotheses.len() {
        return Err(ValueError::UnequalLengths);
    }
    Ok(references
        .iter()
        .zip(hypotheses)
        .map(|(a, b)| universal_edit_distance(a, b))
        .collect())
}

pub fn universal_error_rate<T: PartialEq + Send + Sync>(
    references: &[&[T]],
    hypotheses: &[&[T]],
) -> Result<f64, ValueError> {
    // This is the equivalent to the jiwer and evaluatio package in Python
    // Takes the sum of the edit distances and divides it by total length of
    // the hypotheses.
    if references.len() != hypotheses.len() {
        return Err(ValueError::UnequalLengths);
    }

    let distance: usize = references
        .par_iter()
        .zip(hypotheses)
        .map(|(reference, hypothesis)| universal_edit_distance(reference, hypothesis))
        .sum();

    let total: usize = references.par_iter().map(|reference| reference.len()).sum();

    Ok((distance as f64) / (total as f64))
}

/// An actual implementation of the Levenshtein distance
pub fn universal_edit_distance<T: PartialEq>(left: &[T], right: &[T]) -> usize {
    if left.is_empty() {
        return right.len();
    }
    if right.is_empty() {
        return left.len();
    }

    if left.len() < right.len() {
        return universal_edit_distance(right, left);
    }

    let len_right = right.len() + 1;
    let mut current_row = vec![0; len_right];
    for (i, v) in current_row.iter_mut().enumerate().skip(1) {
        *v = i;
    }

    let mut pre;
    let mut tmp;

    for (i, left_element) in left.iter().enumerate() {
        // get first column for this row
        pre = current_row[0];
        current_row[0] = i + 1;
        for (j, right_element) in right.iter().enumerate() {
            tmp = current_row[j + 1];
            current_row[j + 1] = std::cmp::min(
                // deletion
                tmp + 1,
                std::cmp::min(
                    // insertion
                    current_row[j] + 1,
                    // match or substitution
                    pre + if left_element == right_element { 0 } else { 1 },
                ),
            );
            pre = tmp;
        }
    }

    current_row[len_right - 1]
}

pub fn universal_error_rate_ci<T: PartialEq>(
    references: &[&[T]],
    hypotheses: &[&[T]],
    iterations: usize,
    alpha: f64,
) -> Result<ConfidenceInterval, ValueError> {
    let edit_distances = universal_edit_distance_per_pair(references, hypotheses)?;
    let ref_lengths: Vec<usize> = references.iter().map(|a| a.len()).collect();
    error_rate_ci(&edit_distances, &ref_lengths, iterations, alpha)
}

pub fn error_rate_ci(
    edit_distances: &[usize],
    ref_lengths: &[usize],
    iterations: usize,
    alpha: f64,
) -> Result<ConfidenceInterval, ValueError> {
    if !(0.0..=1.0).contains(&alpha) {
        return Err(ValueError::InvalidAlphaValue);
    }

    if iterations < 1 {
        return Err(ValueError::AtLeastOneIterationRequired);
    }

    if edit_distances.is_empty() {
        return Err(ValueError::NotEnoughValues);
    }

    let n = edit_distances.len();

    // Compute observed corpus-level WER
    let total_edits: usize = edit_distances.iter().sum();
    let total_refs: usize = ref_lengths.iter().sum();
    let mean = total_edits as f64 / total_refs as f64;

    let mut bootstrapped: Vec<f64> = (0..iterations)
        .into_par_iter()
        .map(|_| {
            let mut sum_edits = 0;
            let mut sum_refs = 0;

            for _ in 0..n {
                let idx = fastrand::usize(0..n);
                sum_edits += edit_distances[idx];
                sum_refs += ref_lengths[idx];
            }

            sum_edits as f64 / sum_refs as f64
        })
        .collect();

    bootstrapped.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let lower_idx = ((alpha / 2.0) * iterations as f64).floor() as usize;
    let upper_idx = ((1.0 - alpha / 2.0) * iterations as f64).floor() as usize;

    let lower = bootstrapped[lower_idx.min(iterations - 1)];
    let upper = bootstrapped[upper_idx.min(iterations - 1)];

    Ok(ConfidenceInterval { mean, lower, upper })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edit_distance_works_both_ways() {
        let reference = vec!["h", "e", "l", "l", "o"];
        let prediction = vec!["h", "e", "l", "o"];
        let result = universal_edit_distance(&reference, &prediction);
        assert_eq!(result, 1);
        let result = universal_edit_distance(&prediction, &reference);
        assert_eq!(result, 1);
        let result = universal_edit_distance(&vec![&reference], &vec![&prediction]);
        assert_eq!(1, result);
    }

    #[test]
    fn error_rate_should_return_correct_value() {
        let reference = vec!["h", "e", "l", "l", "o"];
        let prediction = vec!["h", "e", "l", "o"];
        let result = universal_error_rate(&[&reference], &[&prediction]).unwrap();
        assert_eq!(0.2, result);
    }

    #[test]
    fn edit_distance_should_be_length_of_left_if_right_is_empty() {
        let left = vec!["h", "e", "l", "l", "o"];
        let right = vec![];
        let result = universal_edit_distance(&left, &right);
        assert_eq!(5, result);
    }

    #[test]
    fn edit_distance_should_be_length_of_right_if_left_is_empty() {
        let left = vec![];
        let right = vec!["h", "e", "l", "l", "o"];
        let result = universal_edit_distance(&left, &right);
        assert_eq!(5, result);
    }

    #[test]
    fn universal_error_rate_ci_should_work() {
        let reference = vec![0, 0];
        let prediction = vec![0, 1];
        let result = universal_error_rate_ci(&[&reference], &[&prediction], 1, 0.05).unwrap();
        let expected = ConfidenceInterval {
            mean: 1.0 / 2.0,
            lower: 1.0 / 2.0,
            upper: 1.0 / 2.0,
        };
        println!("{:?}", result);
        assert_eq!(result, expected)
    }

    #[test]
    fn universal_error_rate_ci_should_not_allow_lists_of_different_lengths() {
        let reference = vec![0];
        let prediction = vec![0, 1];
        let result = universal_error_rate_ci(&[&reference, &reference], &[&prediction], 1, 0.05);
        assert!(result.is_err())
    }

    #[test]
    fn universal_error_rate_ci_should_require_at_least_one_iteration() {
        let reference = vec![0];
        let prediction = vec![0];
        let result = universal_error_rate_ci(&[&reference], &[&prediction], 0, 0.05);
        assert!(result.is_err())
    }

    #[test]
    fn universal_error_rate_ci_should_not_allow_wrong_alpha_above_1() {
        let reference = vec![0];
        let prediction = vec![0];
        let result = universal_error_rate_ci(&[&reference], &[&prediction], 1, 1.01);
        assert!(result.is_err());
        // Should not fail
        let _ = universal_error_rate_ci(&[&reference], &[&prediction], 1, 1.0).unwrap();
    }

    #[test]
    fn universal_error_rate_ci_should_not_allow_wrong_alpha_below_0() {
        let reference = vec![0];
        let prediction = vec![0];
        let result = universal_error_rate_ci(&[&reference], &[&prediction], 1, -0.01);
        assert!(result.is_err());
        // Should not fail
        let _ = universal_error_rate_ci(&[&reference], &[&prediction], 1, -0.0);
    }

    #[test]
    fn universal_error_rate_ci_should_require_at_least_one_pair() {
        let reference: Vec<&[i32]> = vec![];
        let prediction: Vec<&[i32]> = vec![];
        let result = universal_error_rate_ci(&reference, &prediction, 1, 0.05);
        assert!(result.is_err())
    }
}
