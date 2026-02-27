pub(crate) fn mean(numbers: &[f64]) -> f64 {
    let sum: f64 = numbers.iter().sum();
    let count: f64 = numbers.len() as f64;
    sum / count
}

pub(crate) fn variance(x: &[f64], x_mean: f64) -> f64 {
    let sum: f64 = x.iter().map(|xi| (xi - x_mean).powf(2.0)).sum();
    sum / (x.len() - 1) as f64
}
