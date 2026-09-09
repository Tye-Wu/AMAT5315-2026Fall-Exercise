//! Observables measured from a saved trajectory: speed-distribution fit and
//! the secular energy drift.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::SplitMix64;
    use crate::store::FrameRecord;

    fn synth_frame(vx: Vec<f64>, vy: Vec<f64>) -> FrameRecord {
        let n = vx.len();
        let k: f64 = vx.iter().zip(&vy).map(|(a, b)| 0.5 * (a * a + b * b)).sum();
        FrameRecord {
            frame: 0,
            t: 0.0,
            x: (0..n).map(|i| i as f64 * 0.1).collect(),
            y: vec![0.0; n],
            vx,
            vy,
            k,
            u: 0.0,
            e: k,
            tkin: k / n as f64,
        }
    }

    #[test]
    fn speed_fit_recovers_temperature_of_maxwell_speeds() {
        // 2D Maxwell-Boltzmann at T = 0.5: Gaussian components, var per axis T.
        let mut rng = SplitMix64::new(11);
        let n = 4000;
        let s = 0.5f64.sqrt();
        let mut vx = Vec::with_capacity(n);
        let mut vy = Vec::with_capacity(n);
        for _ in 0..n {
            vx.push(rng.gaussian() * s);
            vy.push(rng.gaussian() * s);
        }
        let frames = vec![synth_frame(vx, vy)];
        let (t_fit, chi2) = fit_speed_temperature(&frames, 40);
        assert!((t_fit - 0.5).abs() < 0.05, "T_fit {t_fit}");
        assert!(chi2 < 3.0, "chi2/dof {chi2}");
    }

    #[test]
    fn secular_drift_measures_slope_over_time() {
        let t = vec![0.0, 1.0, 2.0, 3.0];
        let e = vec![0.0, 0.001, 0.002, 0.003]; // per-atom energy rising 1e-3 per t
        let d = secular_drift(&t, &e);
        assert!((d - 0.003).abs() < 1e-6, "drift {d}");
    }
}
