use crate::err::ValueError;

pub(crate) fn mean(numbers: &[f64]) -> Result<f64, ValueError> {
    if numbers.is_empty() {
        return Err(ValueError::NotEnoughValues);
    }
    let sum: f64 = numbers.iter().sum();
    let count: f64 = numbers.len() as f64;
    Ok(sum / count)
}

pub(crate) fn variance(x: &[f64], x_mean: f64) -> Result<f64, ValueError> {
    if x.len() < 2 {
        return Err(ValueError::NotEnoughValues);
    }
    let sum: f64 = x.iter().map(|xi| (xi - x_mean).powi(2)).sum();
    Ok(sum / (x.len() - 1) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_raises_not_enough_values() {
        let var = mean(&[]);
        assert!(var.is_err())
    }

    #[test]
    fn variance_raises_not_enough_values() {
        let var = variance(&[1.0], 1.0);
        assert!(var.is_err())
    }
}
