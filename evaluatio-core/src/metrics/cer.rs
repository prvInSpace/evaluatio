use crate::metrics::uer;

fn split_strings_into_char_vec(list: &[&str]) -> Vec<Vec<char>> {
    list.iter().map(|x| x.chars().collect()).collect()
}

pub fn character_error_rate_per_pair(references: &Vec<&str>, hypotheses: &Vec<&str>) -> Vec<f64> {
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
) -> Vec<usize> {
    let references_split = split_strings_into_char_vec(references);
    let hypotheses_split = split_strings_into_char_vec(hypotheses);
    uer::universal_edit_distance_per_pair(
        &references_split.iter().collect(),
        &hypotheses_split.iter().collect(),
    )
}

pub fn character_error_rate(references: &Vec<&str>, hypotheses: &Vec<&str>) -> f64 {
    let references_split = split_strings_into_char_vec(references);
    let hypotheses_split = split_strings_into_char_vec(hypotheses);
    uer::universal_error_rate(
        &references_split.iter().collect(),
        &hypotheses_split.iter().collect(),
    )
}
