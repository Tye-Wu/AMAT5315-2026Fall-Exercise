//! Observables measured from a saved trajectory: speed-distribution fit and
//! the secular energy drift.

use crate::store::FrameRecord;

/// Fit the pooled speed distribution of `frames` to the 2D Maxwell-Boltzmann
/// `f(v) = (v/T) exp(-v^2/(2T))` and return `(T_fit, chi2/dof)`.
pub fn fit_speed_temperature(frames: &[FrameRecord], bins: usize) -> (f64, f64) {
    let mut speeds: Vec<f64> = Vec::new();
    for f in frames {
        for i in 0..f.vx.len() {
            speeds.push((f.vx[i] * f.vx[i] + f.vy[i] * f.vy[i]).sqrt());
        }
    }
    if speeds.is_empty() {
        return (0.0, f64::INFINITY);
    }
    let vmax = speeds.iter().cloned().fold(0.0, f64::max) * 1.05;
    let h = vmax / bins as f64;
    let mut count = vec![0usize; bins];
    for &v in &speeds {
        let b = ((v / h) as usize).min(bins - 1);
        count[b] += 1;
    }
    let total = speeds.len() as f64;

    let mut best = (f64::INFINITY, 0.5);
    let mut t = 0.10;
    while t < 1.5 {
        let mut chi2 = 0.0;
        let mut dof = 0usize;
        for b in 0..bins {
            let vlo = b as f64 * h;
            let vhi = vlo + h;
            let expected = (mb_cdf(vhi, t) - mb_cdf(vlo, t)) * total;
            if expected > 5.0 {
                let diff = count[b] as f64 - expected;
                chi2 += diff * diff / expected;
                dof += 1;
            }
        }
        let chi2_dof = chi2 / ((dof as f64) - 1.0).max(1.0);
        if chi2_dof < best.0 {
            best = (chi2_dof, t);
        }
        t += 0.002;
    }
    (best.1, best.0)
}

/// P(v <= v) for the 2D Maxwell-Boltzmann with per-component variance T:
/// `1 - exp(-v^2 / (2T))`.
fn mb_cdf(v: f64, t: f64) -> f64 {
    if t <= 0.0 {
        return 0.0;
    }
    1.0 - (-v * v / (2.0 * t)).exp()
}

/// Linear-trend drift of a per-atom energy series over time:
/// `|slope| * (t_last - t_first)`, slope from ordinary least squares.
pub fn secular_drift(t: &[f64], e: &[f64]) -> f64 {
    let n = t.len().min(e.len());
    if n < 2 {
        return 0.0;
    }
    let nf = n as f64;
    let mt = t[..n].iter().sum::<f64>() / nf;
    let me = e[..n].iter().sum::<f64>() / nf;
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..n {
        num += (t[i] - mt) * (e[i] - me);
        den += (t[i] - mt) * (t[i] - mt);
    }
    let slope = if den > 0.0 { num / den } else { 0.0 };
    (slope * (t[n - 1] - t[0])).abs()
}

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
