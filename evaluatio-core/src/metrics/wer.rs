use crate::metrics::uer;

pub(crate) fn split_strings_into_word_vec<'a>(list: &Vec<&'a str>) -> Vec<Vec<&'a str>> {
    list.iter()
        .map(|x| x.split(" ").filter(|x| !x.is_empty()).collect())
        .collect::<Vec<Vec<&str>>>()
}

pub fn word_error_rate_per_pair(references: &Vec<&str>, hypotheses: &Vec<&str>) -> Vec<f64> {
    let references_split = split_strings_into_word_vec(references);
    let hypotheses_split = split_strings_into_word_vec(hypotheses);
    uer::universal_error_rate_per_pair(
        &references_split.iter().collect(),
        &hypotheses_split.iter().collect(),
    )
}

pub fn word_edit_distance_per_pair(references: &Vec<&str>, hypotheses: &Vec<&str>) -> Vec<usize> {
    let references_split = split_strings_into_word_vec(references);
    let hypotheses_split = split_strings_into_word_vec(hypotheses);
    uer::universal_edit_distance_per_pair(
        &references_split.iter().collect(),
        &hypotheses_split.iter().collect(),
    )
}

pub fn word_error_rate(references: &Vec<&str>, hypotheses: &Vec<&str>) -> f64 {
    let references_split = split_strings_into_word_vec(references);
    let hypotheses_split = split_strings_into_word_vec(hypotheses);
    uer::universal_error_rate(
        &references_split.iter().collect(),
        &hypotheses_split.iter().collect(),
    )
}
