use itertools::izip;

use crate::err::ValueError;

pub fn poi_error_rate<T: PartialEq>(
    references: &[&[T]],
    hypotheses: &[&[T]],
    point_of_interests: &[&[bool]],
) -> Result<f64, ValueError> {
    if references.len() != hypotheses.len() {
        return Err(ValueError::UnequalLengths);
    }
    let total: usize = point_of_interests
        .iter()
        .map(|x| x.iter().filter(|b| **b).count())
        .sum();

    let distance: Result<usize, ValueError> = izip!(references, hypotheses, point_of_interests)
        .map(|(prediction, reference, pois)| poi_edit_distance(reference, prediction, pois))
        .sum();

    Ok((distance? as f64) / (total as f64))
}

pub fn poi_edit_distance<T: PartialEq>(
    ref_: &[T],
    hyp: &[T],
    poi: &[bool],
) -> Result<usize, ValueError> {
    if ref_.len() != poi.len() {
        return Err(ValueError::UnequalLengths);
    }

    // Edge case where the hypothesis is empty
    if hyp.is_empty() {
        return Ok(poi.iter().filter(|b| **b).count());
    }

    let hyp_len = hyp.len() + 1;
    let ref_len = ref_.len() + 1;

    let mut table = vec![0usize; hyp_len * ref_len];

    // Pre-fill
    // This looks slightly ugly but it makes clippy happy. It does exactly the same
    // as the loop below just using iterators.
    for (i, v) in table.iter_mut().enumerate().take(ref_len).skip(1) {
        *v = i;
    }
    for i in 1..hyp_len {
        table[i * ref_len] = i;
    }

    // Fill in the table
    for r in 1..ref_len {
        for h in 1..hyp_len {
            let cost = if hyp[h - 1] == ref_[r - 1] { 0 } else { 1 };
            table[r + ref_len * h] = *[
                get_value(&table, ref_len, h as isize - 1, r as isize) + 1, // deletion
                get_value(&table, ref_len, h as isize, r as isize - 1) + 1, // insertion
                get_value(&table, ref_len, h as isize - 1, r as isize - 1) + cost, // substitution/match
            ]
            .iter()
            .min()
            .unwrap();
        }
    }

    // Backtracking
    let mut h = hyp_len - 1;
    let mut r = ref_len - 1;
    let mut edit_distance = 0;

    while h > 0 || r > 0 {
        if poi[r - 1] && ref_[r - 1] != hyp[h - 1] {
            edit_distance += 1;
        }
        let del = get_value(&table, ref_len, h as isize - 1, r as isize);
        let ins = get_value(&table, ref_len, h as isize, r as isize - 1);
        let sub = get_value(&table, ref_len, h as isize - 1, r as isize - 1);

        let lowest = del.min(ins).min(sub);

        if del == lowest {
            h -= 1;
        } else if ins == lowest {
            r -= 1;
        } else {
            h = (h as isize - 1).max(0) as usize;
            r = (r as isize - 1).max(0) as usize;
        }
    }

    Ok(edit_distance)
}

fn get_value(table: &[usize], ref_len: usize, h: isize, r: isize) -> usize {
    // Helper function for indexing
    let h = h.max(0) as usize;
    let r = r.max(0) as usize;
    table[r + ref_len * h]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poi_empty_mask_should_be_zero() {
        let ref_: Vec<char> = "aaaa".chars().collect();
        let hyp: Vec<char> = "bbbb".chars().collect();
        let res = poi_edit_distance(&ref_, &hyp, &[false, false, false, false]).unwrap();
        assert_eq!(res, 0)
    }

    #[test]
    fn poi_full_mask_should_be_len() {
        let ref_: Vec<char> = "aaaa".chars().collect();
        let hyp: Vec<char> = "bbbb".chars().collect();
        let res = poi_edit_distance(&ref_, &hyp, &[true, true, true, true]).unwrap();
        assert_eq!(res, 4)
    }

    #[test]
    fn poi_should_include_outliers() {
        let ref_: Vec<char> = "aaaa".chars().collect();
        let hyp: Vec<char> = "baaaab".chars().collect();
        let res = poi_edit_distance(&ref_, &hyp, &[true, false, false, true]).unwrap();
        assert_eq!(res, 2)
    }

    #[test]
    fn poi_should_only_count_when_mask_is_true() {
        let ref_: Vec<char> = "aaba".chars().collect();
        let hyp: Vec<char> = "caca".chars().collect();
        let res = poi_edit_distance(&ref_, &hyp, &[false, false, true, false]).unwrap();
        assert_eq!(res, 1)
    }

    #[test]
    fn poi_should_fail_if_ref_and_poi_differ() {
        let ref_: Vec<char> = "aaaa".chars().collect();
        let hyp: Vec<char> = "aaaa".chars().collect();
        let res = poi_edit_distance(&ref_, &hyp, &[false, false, false]);
        assert!(res.is_err())
    }

    #[test]
    fn poi_should_be_sum_of_mask_if_hyp_is_empty() {
        let ref_: Vec<char> = "aaaa".chars().collect();
        let hyp: Vec<char> = "".chars().collect();
        let res = poi_edit_distance(&ref_, &hyp, &[false, true, true, false]).unwrap();
        assert_eq!(res, 2)
    }

    #[test]
    fn pier_should_return_correct_value() {
        let ref_: Vec<char> = "aaaa".chars().collect();
        let hyp: Vec<char> = "abaa".chars().collect();
        let poi = vec![false, true, true, false];
        let res = poi_error_rate(&[&ref_], &[&hyp], &[&poi]).unwrap();
        assert_eq!(res, 0.5)
    }

    #[test]
    fn pier_should_fail_if_ref_and_poi_differ() {
        let ref_: Vec<char> = "aaaa".chars().collect();
        let hyp: Vec<char> = "aaaa".chars().collect();
        let poi = vec![false, false, false];
        let res = poi_error_rate(&[&ref_], &[&hyp], &[&poi]);
        assert!(res.is_err())
    }

    #[test]
    fn pier_should_fail_if_ref_and_hyp_differ() {
        let res = poi_error_rate(&[&[0]], &[&[0], &[1]], &[&[true]]);
        assert!(res.is_err())
    }
}
