use crate::err::ValueError;

pub(crate) fn mean(numbers: &[f64]) -> f64 {
    let sum: f64 = numbers.iter().sum();
    let count: f64 = numbers.len() as f64;
    sum / count
}

pub(crate) fn variance(x: &[f64], x_mean: f64) -> Result<f64, ValueError> {
    if x.len() < 2 {
        return Err(ValueError::NotEnoughValues);
    }
    let sum: f64 = x.iter().map(|xi| (xi - x_mean).powi(2)).sum();
    Ok(sum / (x.len() - 1) as f64)
}
