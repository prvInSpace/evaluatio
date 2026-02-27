use crate::{inference::ci::ConfidenceInterval, metrics::uer};

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

pub fn word_error_rate_ci(
    references: &Vec<&str>,
    hypotheses: &Vec<&str>,
    iterations: usize,
    alpha: f64,
) -> ConfidenceInterval {
    let edit_distances = word_edit_distance_per_pair(references, hypotheses);
    let ref_lengths: Vec<usize> = split_strings_into_word_vec(references)
        .iter()
        .map(|a| a.len())
        .collect();

    let n = edit_distances.len();
    assert!(n == ref_lengths.len(), "Lengths must match");
    assert!(n > 0, "Cannot compute CI for empty slice");
    assert!(alpha > 0.0 && alpha < 1.0, "Alpha must be in (0,1)");

    // Compute observed corpus-level WER
    let total_edits: usize = edit_distances.iter().sum();
    let total_refs: usize = ref_lengths.iter().sum();
    let mean = total_edits as f64 / total_refs as f64;

    let mut bootstrapped = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        let mut sum_edits = 0;
        let mut sum_refs = 0;

        for _ in 0..n {
            let idx = fastrand::usize(0..n);
            sum_edits += edit_distances[idx];
            sum_refs += ref_lengths[idx];
        }

        bootstrapped.push(sum_edits as f64 / sum_refs as f64);
    }

    bootstrapped.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let lower_idx = ((alpha / 2.0) * iterations as f64).floor() as usize;
    let upper_idx = ((1.0 - alpha / 2.0) * iterations as f64).floor() as usize;

    let lower = bootstrapped[lower_idx.min(iterations - 1)];
    let upper = bootstrapped[upper_idx.min(iterations - 1)];

    ConfidenceInterval { mean, lower, upper }
}
