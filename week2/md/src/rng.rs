//! Deterministic pseudo-random numbers (SplitMix64 + Box-Muller). No rand crate.

/// A deterministic 64-bit SplitMix64 generator.
///
/// The same seed always produces the same stream, so runs are reproducible.
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Create a generator from a 64-bit seed.
    pub fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }

    /// Next raw 64-bit value.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Next uniform value in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    /// Standard normal deviate (Box-Muller), mean 0, variance 1.
    pub fn gaussian(&mut self) -> f64 {
        let u = self.next_f64().max(1e-300);
        let v = self.next_f64();
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

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
