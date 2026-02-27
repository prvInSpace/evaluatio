use crate::stats;
use fastrand;

pub fn paired_bootstrap_test(x1: &[f64], x2: &[f64], iterations: usize) -> f64 {
    let diffs: Vec<f64> = x1.iter().zip(x2).map(|(a, b)| a - b).collect();
    let n = diffs.len();
    let diff_mean = stats::mean(&diffs);
    let diffs_centered: Vec<f64> = diffs.iter().map(|di| di - diff_mean).collect();
    let t_obs = diff_mean.abs();

    let mut extreme_or_equal = 0usize;
    for _ in 0..iterations {
        let sample_mean = (0..n)
            .map(|_| diffs_centered[fastrand::usize(0..n)])
            .sum::<f64>()
            / (n as f64);

        if sample_mean.abs() >= t_obs {
            extreme_or_equal += 1
        }
    }

    (extreme_or_equal as f64 + 1.0) / (iterations as f64 + 1.0)
}
