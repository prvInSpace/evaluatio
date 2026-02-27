pub fn cohens_d_independent(x1: &[f64], x2: &[f64]) -> f64 {
    let x1_len = x1.len() as f64;
    let x2_len = x2.len() as f64;
    let x1_mean = _mean(x1);
    let x2_mean = _mean(x2);
    let x1_variance = _variance(x1, x1_mean);
    let x2_variance = _variance(x2, x2_mean);

    let s1: f64 = (x1_len - 1.0) * x1_variance;
    let s2: f64 = (x2_len - 1.0) * x2_variance;
    let s: f64 = ((s1 + s2) / (x1_len + x2_len - 2.0)).sqrt();
    (x1_mean - x2_mean) / s
}

pub fn cohens_d_paired(x1: &[f64], x2: &[f64]) -> f64 {
    let diff: Vec<f64> = x1.iter().zip(x2).map(|(a, b)| a - b).collect();
    let diff_mean = _mean(&diff);
    let diff_std = _variance(&diff, diff_mean).sqrt();
    diff_mean / diff_std
}

fn _mean(numbers: &[f64]) -> f64 {
    let sum: f64 = numbers.iter().sum();
    let count: f64 = numbers.len() as f64;
    sum / count
}

fn _variance(x: &[f64], x_mean: f64) -> f64 {
    let sum: f64 = x.iter().map(|xi| (xi - x_mean).powf(2.0)).sum();
    sum / (x.len() - 1) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = cohens_d_independent(&[1.0, 2.0, 3.0, 4.0], &[2.0, 3.0, 4.0, 5.0]);
        assert_eq!(result, -0.7745966692414834);
    }
}
