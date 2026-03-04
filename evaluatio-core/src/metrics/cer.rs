use crate::{err::ValueError, metrics::uer};

fn split_strings_into_char_vec(list: &[&str]) -> Vec<Vec<char>> {
    list.iter().map(|x| x.chars().collect()).collect()
}

pub fn character_error_rate_per_pair(
    references: &Vec<&str>,
    hypotheses: &Vec<&str>,
) -> Result<Vec<f64>, ValueError> {
    let references_split = split_strings_into_char_vec(references);
    let hypotheses_split = split_strings_into_char_vec(hypotheses);
    uer::universal_error_rate_per_pair(
        &references_split.iter().collect(),
        &hypotheses_split.iter().collect(),
    )
}

pub fn character_edit_distance_per_pair(
    references: &Vec<&str>,
    hypotheses: &Vec<&str>,
) -> Result<Vec<usize>, ValueError> {
    let references_split = split_strings_into_char_vec(references);
    let hypotheses_split = split_strings_into_char_vec(hypotheses);
    uer::universal_edit_distance_per_pair(
        &references_split.iter().collect(),
        &hypotheses_split.iter().collect(),
    )
}

pub fn character_error_rate(
    references: &Vec<&str>,
    hypotheses: &Vec<&str>,
) -> Result<f64, ValueError> {
    let references_split = split_strings_into_char_vec(references);
    let hypotheses_split = split_strings_into_char_vec(hypotheses);
    uer::universal_error_rate(
        &references_split.iter().collect(),
        &hypotheses_split.iter().collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn character_error_rate_should_return_correct_value() {
        let reference = vec!["hello world"];
        let prediction = vec!["helo world"];
        let result = character_error_rate(&reference, &prediction).unwrap();
        assert_eq!(1.0/11.0, result);
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
        assert_eq!(vec![1.0/11.0], result);
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
}
