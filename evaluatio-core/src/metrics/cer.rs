use crate::metrics::uer;

fn split_strings_into_char_vec(list: &[&str]) -> Vec<Vec<char>> {
    list.iter().map(|x| x.chars().collect()).collect()
}

pub fn character_error_rate_per_pair(predictions: &Vec<&str>, references: &Vec<&str>) -> Vec<f64> {
    let predictions_split = split_strings_into_char_vec(predictions);
    let references_split = split_strings_into_char_vec(references);
    uer::universal_error_rate_per_pair(
        &predictions_split.iter().collect(),
        &references_split.iter().collect(),
    )
}

pub fn character_edit_distance_per_pair(
    predictions: &Vec<&str>,
    references: &Vec<&str>,
) -> Vec<usize> {
    let predictions_split = split_strings_into_char_vec(predictions);
    let references_split = split_strings_into_char_vec(references);
    uer::universal_edit_distance_per_pair(
        &predictions_split.iter().collect(),
        &references_split.iter().collect(),
    )
}

pub fn character_error_rate(predictions: &Vec<&str>, references: &Vec<&str>) -> f64 {
    let predictions_split = split_strings_into_char_vec(predictions);
    let references_split = split_strings_into_char_vec(references);
    uer::universal_error_rate(
        &predictions_split.iter().collect(),
        &references_split.iter().collect(),
    )
}
