//! Deterministic pseudo-random numbers (SplitMix64 + Box-Muller). No rand crate.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_repeats() {
        let mut a = SplitMix64::new(42);
        let mut b = SplitMix64::new(42);
        for _ in 0..10 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn gaussian_mean_and_variance() {
        let mut rng = SplitMix64::new(1);
        let vals: Vec<f64> = (0..200_000).map(|_| rng.gaussian()).collect();
        let n = vals.len() as f64;
        let mean = vals.iter().sum::<f64>() / n;
        let var = vals.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
        assert!(mean.abs() < 0.01, "mean {mean}");
        assert!((var - 1.0).abs() < 0.02, "var {var}");
    }

    #[test]
    fn uniforms_in_unit_interval() {
        let mut rng = SplitMix64::new(9);
        for _ in 0..1000 {
            let u = rng.next_f64();
            assert!((0.0..1.0).contains(&u), "u {u}");
        }
    }
}
