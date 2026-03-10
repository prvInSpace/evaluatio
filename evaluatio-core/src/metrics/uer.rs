use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::err::ValueError;

/// Universally applicable error rates and distances
pub fn universal_error_rate_per_pair<T: PartialEq>(
    references: &Vec<&Vec<T>>,
    hypotheses: &Vec<&Vec<T>>,
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
    references: &Vec<&Vec<T>>,
    hypotheses: &Vec<&Vec<T>>,
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
    references: &Vec<&Vec<T>>,
    hypotheses: &Vec<&Vec<T>>,
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
pub fn universal_edit_distance<T: PartialEq>(left: &Vec<T>, right: &Vec<T>) -> usize {
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
        let result = universal_error_rate(&vec![&reference], &vec![&prediction]).unwrap();
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
}
