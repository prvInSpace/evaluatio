use crate::{err::ValueError, inference::ci::ConfidenceInterval, metrics::uer};

fn split_strings_into_char_vec(list: &[&str]) -> Vec<Vec<char>> {
    list.iter().map(|x| x.chars().collect()).collect()
}

fn convert_list_reference(list: &[Vec<char>]) -> Vec<&[char]> {
    list.iter().map(|v| v.as_slice()).collect()
}

pub fn character_error_rate_per_pair(
    references: &Vec<&str>,
    hypotheses: &Vec<&str>,
) -> Result<Vec<f64>, ValueError> {
    let references_split = split_strings_into_char_vec(references);
    let hypotheses_split = split_strings_into_char_vec(hypotheses);
    uer::universal_error_rate_per_pair(
        &convert_list_reference(&references_split),
        &convert_list_reference(&hypotheses_split),
    )
}

pub fn character_edit_distance_per_pair(
    references: &Vec<&str>,
    hypotheses: &Vec<&str>,
) -> Result<Vec<usize>, ValueError> {
    let references_split = split_strings_into_char_vec(references);
    let hypotheses_split = split_strings_into_char_vec(hypotheses);
    uer::universal_edit_distance_per_pair(
        &convert_list_reference(&references_split),
        &convert_list_reference(&hypotheses_split),
    )
}

pub fn character_error_rate(
    references: &Vec<&str>,
    hypotheses: &Vec<&str>,
) -> Result<f64, ValueError> {
    let references_split = split_strings_into_char_vec(references);
    let hypotheses_split = split_strings_into_char_vec(hypotheses);
    uer::universal_error_rate(
        &convert_list_reference(&references_split),
        &convert_list_reference(&hypotheses_split),
    )
}

pub fn character_error_rate_ci(
    references: &Vec<&str>,
    hypotheses: &Vec<&str>,
    iterations: usize,
    alpha: f64,
) -> Result<ConfidenceInterval, ValueError> {
    let edit_distances = character_edit_distance_per_pair(references, hypotheses)?;
    let ref_lengths: Vec<usize> = split_strings_into_char_vec(references)
        .iter()
        .map(|a| a.len())
        .collect();
    uer::error_rate_ci(&edit_distances, &ref_lengths, iterations, alpha)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn character_error_rate_should_return_correct_value() {
        let reference = vec!["hello world"];
        let prediction = vec!["helo world"];
        let result = character_error_rate(&reference, &prediction).unwrap();
        assert_eq!(1.0 / 11.0, result);
    }

    #[test]
    fn character_error_rate_should_return_error_if_lengths_differ() {
        let reference = vec!["hello world"];
        let prediction = vec![];
        let result = character_error_rate(&reference, &prediction);
        assert!(result.is_err());
    }

    #[test]
    fn character_error_rate_per_pair_should_return_correct_value() {
        let reference = vec!["hello world"];
        let prediction = vec!["helo world"];
        let result = character_error_rate_per_pair(&reference, &prediction).unwrap();
        assert_eq!(vec![1.0 / 11.0], result);
    }

    #[test]
    fn character_error_rate_per_pair_should_return_error_if_lengths_differ() {
        let reference = vec!["hello world"];
        let prediction = vec![];
        let result = character_error_rate_per_pair(&reference, &prediction);
        assert!(result.is_err());
    }

    #[test]
    fn character_edit_distance_per_pair_should_return_correct_value() {
        let reference = vec!["hello world"];
        let prediction = vec!["helo word"];
        let result = character_edit_distance_per_pair(&reference, &prediction).unwrap();
        assert_eq!(vec![2], result);
    }

    #[test]
    fn character_edit_distance_per_pair_should_return_error_if_lengths_differ() {
        let reference = vec!["hello world"];
        let prediction = vec![];
        let result = character_edit_distance_per_pair(&reference, &prediction);
        assert!(result.is_err());
    }

    #[test]
    fn character_error_rate_ci_should_work() {
        let reference = vec!["hello world"];
        let prediction = vec!["helo world"];
        let result = character_error_rate_ci(&reference, &prediction, 1, 0.05).unwrap();
        let expected = ConfidenceInterval {
            mean: 1.0 / 11.0,
            lower: 1.0 / 11.0,
            upper: 1.0 / 11.0,
        };
        assert_eq!(result, expected)
    }

    #[test]
    fn character_error_rate_ci_should_not_allow_lists_of_different_lengths() {
        let reference = vec!["hello world"];
        let prediction = vec!["helo world", "tada!"];
        let result = character_error_rate_ci(&reference, &prediction, 1, 0.05);
        assert!(result.is_err())
    }

    #[test]
    fn character_error_rate_ci_should_require_at_least_one_iteration() {
        let reference = vec!["hello world"];
        let prediction = vec!["helo world"];
        let result = character_error_rate_ci(&reference, &prediction, 0, 0.05);
        assert!(result.is_err())
    }

    #[test]
    fn character_error_rate_ci_should_not_allow_wrong_alpha_above_1() {
        let reference = vec!["hello world"];
        let prediction = vec!["helo world"];
        let result = character_error_rate_ci(&reference, &prediction, 1, 1.01);
        assert!(result.is_err());
        // Should not fail
        let _ = character_error_rate_ci(&reference, &prediction, 1, 1.0).unwrap();
    }

    #[test]
    fn character_error_rate_ci_should_not_allow_wrong_alpha_below_0() {
        let reference = vec!["hello world"];
        let prediction = vec!["helo world"];
        let result = character_error_rate_ci(&reference, &prediction, 1, -0.01);
        assert!(result.is_err());
        // Should not fail
        let _ = character_error_rate_ci(&reference, &prediction, 1, -0.0);
    }

    #[test]
    fn character_error_rate_ci_should_require_at_least_one_pair() {
        let reference = vec![];
        let prediction = vec![];
        let result = character_error_rate_ci(&reference, &prediction, 1, 0.05);
        assert!(result.is_err())
    }
}
