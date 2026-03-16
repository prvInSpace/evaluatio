use crate::{
    err::ValueError,
    inference::ci::ConfidenceInterval,
    metrics::uer::{self, error_rate_ci},
};

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
    let edit_distances = word_edit_distance_per_pair(references, hypotheses)?;
    let ref_lengths: Vec<usize> = split_strings_into_word_vec(references)
        .iter()
        .map(|a| a.len())
        .collect();
    error_rate_ci(&edit_distances, &ref_lengths, iterations, alpha)
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
