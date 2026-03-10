use crate::{err::ValueError, inference::ci::ConfidenceInterval, metrics::uer};
use rayon::prelude::*;

pub(crate) fn split_strings_into_word_vec<'a>(list: &Vec<&'a str>) -> Vec<Vec<&'a str>> {
    list.iter()
        .map(|x| x.split_whitespace().filter(|x| !x.is_empty()).collect())
        .collect::<Vec<Vec<&str>>>()
}

pub fn word_error_rate_per_pair(
    references: &Vec<&str>,
    hypotheses: &Vec<&str>,
) -> Result<Vec<f64>, ValueError> {
    let references_split = split_strings_into_word_vec(references);
    let hypotheses_split = split_strings_into_word_vec(hypotheses);
    uer::universal_error_rate_per_pair(
        &references_split.iter().collect(),
        &hypotheses_split.iter().collect(),
    )
}

pub fn word_edit_distance_per_pair(
    references: &Vec<&str>,
    hypotheses: &Vec<&str>,
) -> Result<Vec<usize>, ValueError> {
    let references_split = split_strings_into_word_vec(references);
    let hypotheses_split = split_strings_into_word_vec(hypotheses);
    uer::universal_edit_distance_per_pair(
        &references_split.iter().collect(),
        &hypotheses_split.iter().collect(),
    )
}

pub fn word_error_rate(references: &Vec<&str>, hypotheses: &Vec<&str>) -> Result<f64, ValueError> {
    let references_split = split_strings_into_word_vec(references);
    let hypotheses_split = split_strings_into_word_vec(hypotheses);
    uer::universal_error_rate(
        &references_split.iter().collect(),
        &hypotheses_split.iter().collect(),
    )
}

pub fn word_error_rate_ci(
    references: &Vec<&str>,
    hypotheses: &Vec<&str>,
    iterations: usize,
    alpha: f64,
) -> Result<ConfidenceInterval, ValueError> {
    if !(0.0..=1.0).contains(&alpha) {
        return Err(ValueError::InvalidAlphaValue);
    }

    if iterations < 1 {
        return Err(ValueError::AtLeastOneIterationRequired);
    }

    let edit_distances = word_edit_distance_per_pair(references, hypotheses)?;
    let ref_lengths: Vec<usize> = split_strings_into_word_vec(references)
        .iter()
        .map(|a| a.len())
        .collect();

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
    fn word_error_rate_should_return_correct_value() {
        let reference = vec!["hello world"];
        let prediction = vec!["helo world"];
        let result = word_error_rate(&reference, &prediction).unwrap();
        assert_eq!(0.5, result);
    }

    #[test]
    fn word_error_rate_should_return_error_if_lengths_differ() {
        let reference = vec!["hello world"];
        let prediction = vec![];
        let result = word_error_rate(&reference, &prediction);
        assert!(result.is_err());
    }

    #[test]
    fn word_error_rate_per_pair_should_return_correct_value() {
        let reference = vec!["hello world"];
        let prediction = vec!["helo world"];
        let result = word_error_rate_per_pair(&reference, &prediction).unwrap();
        assert_eq!(vec![0.5], result);
    }

    #[test]
    fn word_error_rate_per_pair_should_return_error_if_lengths_differ() {
        let reference = vec!["hello world"];
        let prediction = vec![];
        let result = word_error_rate_per_pair(&reference, &prediction);
        assert!(result.is_err());
    }

    #[test]
    fn word_edit_distance_per_pair_should_return_correct_value() {
        let reference = vec!["hello world"];
        let prediction = vec!["helo world"];
        let result = word_edit_distance_per_pair(&reference, &prediction).unwrap();
        assert_eq!(vec![1], result);
    }

    #[test]
    fn word_edit_distance_per_pair_should_return_error_if_lengths_differ() {
        let reference = vec!["hello world"];
        let prediction = vec![];
        let result = word_edit_distance_per_pair(&reference, &prediction);
        assert!(result.is_err());
    }

    #[test]
    fn word_error_rate_ci_should_work() {
        let reference = vec!["hello world"];
        let prediction = vec!["helo world"];
        let result = word_error_rate_ci(&reference, &prediction, 1, 0.05).unwrap();
        let expected = ConfidenceInterval {
            mean: 0.5,
            lower: 0.5,
            upper: 0.5,
        };
        assert_eq!(result, expected)
    }

    #[test]
    fn word_error_rate_ci_should_not_allow_lists_of_different_lengths() {
        let reference = vec!["hello world"];
        let prediction = vec!["helo world", "tada!"];
        let result = word_error_rate_ci(&reference, &prediction, 1, 0.05);
        assert!(result.is_err())
    }

    #[test]
    fn word_error_rate_ci_should_require_at_least_one_iteration() {
        let reference = vec!["hello world"];
        let prediction = vec!["helo world"];
        let result = word_error_rate_ci(&reference, &prediction, 0, 0.05);
        assert!(result.is_err())
    }

    #[test]
    fn word_error_rate_ci_should_not_allow_wrong_alpha_above_1() {
        let reference = vec!["hello world"];
        let prediction = vec!["helo world"];
        let result = word_error_rate_ci(&reference, &prediction, 1, 1.01);
        assert!(result.is_err());
        // Should not fail
        let _ = word_error_rate_ci(&reference, &prediction, 1, 1.0).unwrap();
    }

    #[test]
    fn word_error_rate_ci_should_not_allow_wrong_alpha_below_0() {
        let reference = vec!["hello world"];
        let prediction = vec!["helo world"];
        let result = word_error_rate_ci(&reference, &prediction, 1, -0.01);
        assert!(result.is_err());
        // Should not fail
        let _ = word_error_rate_ci(&reference, &prediction, 1, -0.0);
    }

    #[test]
    fn word_error_rate_ci_should_require_at_least_one_pair() {
        let reference = vec![];
        let prediction = vec![];
        let result = word_error_rate_ci(&reference, &prediction, 1, 0.05);
        assert!(result.is_err())
    }
}
