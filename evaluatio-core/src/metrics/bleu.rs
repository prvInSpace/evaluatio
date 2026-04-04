use rayon::prelude::*;

// Default object if Python is not used
#[cfg(not(feature = "python"))]
#[derive(Copy, Clone)]
pub struct BLEUSufficientStats {
    pub counts: [u32; 4], // clipped matches per n-gram order
    pub totals: [u32; 4], // total n-grams per order
    pub sys_len: u32,
    pub ref_len: u32,
}

// Python version of the object with initialiser

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pyclass]
#[derive(Copy, Clone)]
pub struct BLEUSufficientStats {
    #[pyo3(get)]
    pub counts: [u32; 4], // clipped matches per n-gram order
    #[pyo3(get)]
    pub totals: [u32; 4], // total n-grams per order
    #[pyo3(get)]
    pub sys_len: u32,
    #[pyo3(get)]
    pub ref_len: u32,
}

#[cfg(feature = "python")]
#[pymethods]
impl BLEUSufficientStats {
    #[new]
    pub fn new(counts: [u32; 4], totals: [u32; 4], sys_len: u32, ref_len: u32) -> Self {
        Self {
            counts,
            totals,
            sys_len,
            ref_len,
        }
    }
}

pub fn bleu_from_stats(stats: &[BLEUSufficientStats]) -> f64 {
    let counts: [u32; 4] = std::array::from_fn(|i| stats.iter().map(|s| s.counts[i]).sum());
    let totals: [u32; 4] = std::array::from_fn(|i| stats.iter().map(|s| s.totals[i]).sum());
    let sys_len: u32 = stats.iter().map(|s| s.sys_len).sum();
    let ref_len: u32 = stats.iter().map(|s| s.ref_len).sum();

    let bp = if sys_len >= ref_len {
        1.0
    } else {
        (1.0 - ref_len as f64 / sys_len as f64).exp()
    };

    let log_avg = (0..4)
        .map(|i| (counts[i] as f64 / totals[i] as f64).ln())
        .sum::<f64>()
        / 4.0;

    bp * log_avg.exp() * 100.0
}

pub fn bootstrap_bleu(
    stats_a: &[BLEUSufficientStats],
    stats_b: &[BLEUSufficientStats],
    iterations: usize,
) -> f64 {
    let observed_a = bleu_from_stats(stats_a);
    let observed_b = bleu_from_stats(stats_b);
    let better = if observed_a >= observed_b {
        (stats_a, stats_b)
    } else {
        (stats_b, stats_a)
    };

    let extreme_count: usize = (0..iterations)
        .into_par_iter()
        .map_init(
            || fastrand::Rng::new(),
            |rng, _| {
                let indices: Vec<usize> = (0..stats_a.len())
                    .map(|_| rng.usize(0..stats_a.len()))
                    .collect();
                let sample_a: Vec<_> = indices.iter().map(|&i| better.0[i]).collect();
                let sample_b: Vec<_> = indices.iter().map(|&i| better.1[i]).collect();
                (bleu_from_stats(&sample_a) <= bleu_from_stats(&sample_b)) as usize
            },
        )
        .sum();

    (extreme_count + 1) as f64 / (iterations + 1) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_stats(
        counts: [u32; 4],
        totals: [u32; 4],
        sys_len: u32,
        ref_len: u32,
    ) -> BLEUSufficientStats {
        BLEUSufficientStats {
            counts,
            totals,
            sys_len,
            ref_len,
        }
    }

    #[test]
    fn test_perfect_bleu() {
        // All n-grams match, lengths equal -> BP=1, BLEU=100
        let stats = vec![make_stats([10, 9, 8, 7], [10, 9, 8, 7], 10, 10)];
        let score = bleu_from_stats(&stats);
        assert!(
            (score - 100.0).abs() < 1e-6,
            "Expected 100.0, got {}",
            score
        );
    }

    #[test]
    fn test_brevity_penalty_applied() {
        // sys_len < ref_len -> BP < 1 -> BLEU < perfect score
        let stats = vec![make_stats([5, 4, 3, 2], [5, 4, 3, 2], 5, 10)];
        let score = bleu_from_stats(&stats);
        assert!(score < 100.0, "Expected BP to reduce score below 100");
        assert!(score > 0.0);
    }

    #[test]
    fn test_additivity() {
        // Splitting stats across two entries should give same result as one combined entry
        let combined = vec![make_stats([20, 18, 16, 14], [40, 38, 36, 34], 40, 40)];
        let split = vec![
            make_stats([10, 9, 8, 7], [20, 19, 18, 17], 20, 20),
            make_stats([10, 9, 8, 7], [20, 19, 18, 17], 20, 20),
        ];
        let score_combined = bleu_from_stats(&combined);
        let score_split = bleu_from_stats(&split);
        assert!(
            (score_combined - score_split).abs() < 1e-6,
            "Additivity violated: {} vs {}",
            score_combined,
            score_split
        );
    }

    #[test]
    fn test_clearly_significant() {
        // System A has near-perfect matches, B has poor matches
        let stats_a: Vec<_> = (0..30)
            .map(|_| make_stats([10, 9, 8, 7], [10, 9, 8, 7], 10, 10))
            .collect();
        let stats_b: Vec<_> = (0..30)
            .map(|_| make_stats([3, 2, 1, 0], [10, 9, 8, 7], 10, 10))
            .collect();
        let p = bootstrap_bleu(&stats_a, &stats_b, 4999);
        assert!(p < 0.01, "Expected p < 0.01, got {}", p);
    }

    #[test]
    fn test_identical_systems_not_significant() {
        let stats: Vec<_> = (0..30)
            .map(|_| make_stats([8, 7, 6, 5], [10, 9, 8, 7], 10, 10))
            .collect();
        let p = bootstrap_bleu(&stats, &stats, 4999);
        assert!(p > 0.5, "Expected p > 0.5 for identical systems, got {}", p);
    }

    #[test]
    fn test_p_value_bounds() {
        let stats_a: Vec<_> = (0..10)
            .map(|_| make_stats([10, 9, 8, 7], [10, 9, 8, 7], 10, 10))
            .collect();
        let stats_b: Vec<_> = (0..10)
            .map(|_| make_stats([1, 1, 1, 1], [10, 9, 8, 7], 10, 10))
            .collect();
        let p = bootstrap_bleu(&stats_a, &stats_b, 999);
        assert!(p > 0.0, "p-value must be > 0");
        assert!(p <= 1.0, "p-value must be <= 1");
    }

    #[test]
    fn test_minimum_p_value() {
        // With 999 iterations, minimum p is 1/1000
        let stats_a: Vec<_> = (0..50)
            .map(|_| make_stats([10, 9, 8, 7], [10, 9, 8, 7], 10, 10))
            .collect();
        let stats_b: Vec<_> = (0..50)
            .map(|_| make_stats([0, 0, 0, 0], [10, 9, 8, 7], 10, 10))
            .collect();
        let p = bootstrap_bleu(&stats_a, &stats_b, 999);
        assert!(
            (p - 0.001).abs() < 1e-10,
            "Expected minimum p of 0.001, got {}",
            p
        );
    }

    #[test]
    fn test_symmetry() {
        // Swapping a and b should give the same p-value since we always
        // orient better vs worse internally
        let stats_a: Vec<_> = (0..20)
            .map(|_| make_stats([9, 8, 7, 6], [10, 9, 8, 7], 10, 10))
            .collect();
        let stats_b: Vec<_> = (0..20)
            .map(|_| make_stats([6, 5, 4, 3], [10, 9, 8, 7], 10, 10))
            .collect();
        let p_ab = bootstrap_bleu(&stats_a, &stats_b, 9999);
        let p_ba = bootstrap_bleu(&stats_b, &stats_a, 9999);
        assert!(
            (p_ab - p_ba).abs() < 0.05,
            "Expected symmetric p-values, got {} vs {}",
            p_ab,
            p_ba
        );
    }
}
