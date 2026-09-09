//! Run pipeline for the two-dimensional Lennard-Jones fluid.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::Vec2;

    fn temp_of(vs: &[Vec2]) -> f64 {
        let sum: f64 = vs.iter().map(|v| v[0] * v[0] + v[1] * v[1]).sum();
        sum / (2.0 * vs.len() as f64)
    }

    #[test]
    fn default_config_is_reference_run() {
        let c = RunConfig::default();
        assert_eq!(
            (c.atoms, c.rho, c.temperature, c.seed, c.dt, c.cutoff),
            (100, 0.8, 0.5, 42, 0.005, 2.5)
        );
        let expected = (100.0 / 0.8).sqrt();
        assert!((box_length(&c) - expected).abs() < 1e-9);
    }

    #[test]
    fn run_is_deterministic_and_emits_expected_frames() {
        let c = RunConfig {
            atoms: 64,
            rho: 0.8,
            eq: 200,
            steps: 400,
            save_every: 20,
            ..RunConfig::default()
        };
        let r1 = run_fluid(&c);
        let r2 = run_fluid(&c);
        assert_eq!(r1.frames.len(), 20);
        assert_eq!(r1.frames.len(), c.steps / c.save_every);
        for (a, b) in r1.frames.iter().zip(&r2.frames) {
            assert_eq!(a.positions, b.positions);
            assert_eq!(a.velocities, b.velocities);
        }
        assert!(r1.frames.last().unwrap().t > 0.0);
    }

    #[test]
    fn production_temperature_stays_near_target() {
        let c = RunConfig {
            atoms: 64,
            temperature: 0.5,
            eq: 400,
            steps: 300,
            save_every: 25,
            ..RunConfig::default()
        };
        let r = run_fluid(&c);
        let t = temp_of(&r.frames.last().unwrap().velocities);
        assert!((t - 0.5).abs() < 0.15, "production temperature {t}");
    }
}
