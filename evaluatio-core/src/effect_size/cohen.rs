use crate::{err::ValueError, stats};

pub fn cohens_d_independent(x1: &[f64], x2: &[f64]) -> Result<f64, ValueError> {
    let x1_len = x1.len() as f64;
    let x2_len = x2.len() as f64;
    let x1_mean = stats::mean(x1)?;
    let x2_mean = stats::mean(x2)?;
    let x1_variance = stats::variance(x1, x1_mean)?;
    let x2_variance = stats::variance(x2, x2_mean)?;

    let s1: f64 = (x1_len - 1.0) * x1_variance;
    let s2: f64 = (x2_len - 1.0) * x2_variance;
    let s: f64 = ((s1 + s2) / (x1_len + x2_len - 2.0)).sqrt();
    Ok((x1_mean - x2_mean) / s)
}

pub fn cohens_d_paired(x1: &[f64], x2: &[f64]) -> Result<f64, ValueError> {
    let diff: Vec<f64> = x1.iter().zip(x2).map(|(a, b)| a - b).collect();
    let diff_mean = stats::mean(&diff)?;
    let diff_std = stats::variance(&diff, diff_mean)?.sqrt();
    Ok(diff_mean / diff_std)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cohens_d_independent() {
        let result = cohens_d_independent(&[1.0, 2.0, 3.0, 4.0], &[2.0, 3.0, 4.0, 5.0]).unwrap();
        assert_eq!(result, -0.7745966692414834);
    }

    #[test]
    fn test_paired_cohens_d() {
        let left = [1.0, 2.0, 3.0, 4.0];
        let right = [0.0, 3.0, 4.0, 5.0];
        let result = cohens_d_paired(&left, &right).unwrap();
        let ind = cohens_d_independent(&left, &right).unwrap();
        assert_eq!(result, -0.5);
        // It is a more powerful tests, so it should be bigger than the independent test
        assert!(result.abs() > ind.abs())
    }
}
