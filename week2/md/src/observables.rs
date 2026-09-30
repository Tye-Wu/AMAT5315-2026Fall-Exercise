//! Independent checks recomputed from saved positions and velocities.

use crate::store::{FrameRecord, RunJson};
use crate::system::{BoxConfig, ForceMethod, System, interactions_with_method, kinetic_energy};

#[derive(Clone, Debug)]
pub struct CheckReport {
    pub secular_drift: f64,
    pub t_speed: f64,
    pub chi2_per_22: f64,
    pub stored_energy_max_error: f64,
}

impl CheckReport {
    pub fn passes(&self, target_temperature: f64) -> bool {
        self.secular_drift < 2e-3
            && (self.t_speed - target_temperature).abs() < 0.05
            && self.chi2_per_22 < 2.0
            && self.stored_energy_max_error < 1e-8
    }
}

pub fn speed_temperature(frames: &[FrameRecord]) -> f64 {
    let mut sum_v2 = 0.0;
    let mut count = 0usize;
    for frame in frames {
        for v in &frame.vel {
            sum_v2 += v[0] * v[0] + v[1] * v[1];
            count += 1;
        }
    }
    sum_v2 / (2.0 * count as f64)
}

/// The sheet's 24 equal-probability bins at the measured speed temperature.
pub fn speed_shape_chi2_per_22(frames: &[FrameRecord], t_speed: f64) -> f64 {
    let mut counts = [0usize; 24];
    let mut total = 0usize;
    for frame in frames {
        for v in &frame.vel {
            let v2 = v[0] * v[0] + v[1] * v[1];
            let cdf = 1.0 - (-v2 / (2.0 * t_speed)).exp();
            let bin = ((24.0 * cdf).floor() as usize).min(23);
            counts[bin] += 1;
            total += 1;
        }
    }
    let expected = total as f64 / 24.0;
    let chi2: f64 = counts
        .iter()
        .map(|&observed| {
            let difference = observed as f64 - expected;
            difference * difference / expected
        })
        .sum();
    chi2 / 22.0
}

pub fn windowed_secular_drift(energies: &[f64]) -> f64 {
    let k = (energies.len() / 10).max(1);
    let first = energies[..k].iter().sum::<f64>() / k as f64;
    let last = energies[energies.len() - k..].iter().sum::<f64>() / k as f64;
    (last - first).abs() / energies[0].abs()
}

pub fn evaluate_saved_run(meta: &RunJson, frames: &[FrameRecord]) -> Result<CheckReport, String> {
    if frames.is_empty() {
        return Err("trajectory contains no frames".to_string());
    }
    if frames.len() != meta.steps / meta.sample_every {
        return Err(format!(
            "expected {} frames, found {}",
            meta.steps / meta.sample_every,
            frames.len()
        ));
    }
    let bc = BoxConfig {
        lengths: meta.box_lengths,
        cutoff: meta.cutoff,
    };
    let mut energies = Vec::with_capacity(frames.len());
    let mut max_stored_error = 0.0_f64;
    for frame in frames {
        if frame.pos.len() != meta.n || frame.vel.len() != meta.n {
            return Err(format!("frame {} has the wrong atom count", frame.step));
        }
        let expected_t = frame.step as f64 * meta.dt;
        if (frame.t - expected_t).abs() > 1e-10 {
            return Err(format!("frame {} has inconsistent time", frame.step));
        }
        if frame.pos.iter().any(|p| {
            !p[0].is_finite()
                || !p[1].is_finite()
                || !(0.0..meta.box_lengths[0]).contains(&p[0])
                || !(0.0..meta.box_lengths[1]).contains(&p[1])
        }) || frame.vel.iter().flatten().any(|x| !x.is_finite())
        {
            return Err(format!("frame {} contains malformed state", frame.step));
        }
        let system = System::with_box_and_force(
            frame.pos.clone(),
            frame.vel.clone(),
            bc,
            ForceMethod::Naive,
        );
        let recomputed_potential =
            interactions_with_method(&system, ForceMethod::Naive).potential_energy;
        let recomputed_kinetic = kinetic_energy(&system);
        let total = recomputed_potential + recomputed_kinetic;
        let stored = frame.e_pot + frame.e_kin;
        max_stored_error = max_stored_error.max((stored - total).abs() / total.abs().max(1.0));
        energies.push(total);
    }
    let t_speed = speed_temperature(frames);
    Ok(CheckReport {
        secular_drift: windowed_secular_drift(&energies),
        t_speed,
        chi2_per_22: speed_shape_chi2_per_22(frames, t_speed),
        stored_energy_max_error: max_stored_error,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::SplitMix64;

    fn synthetic_frame(vel: Vec<[f64; 2]>) -> FrameRecord {
        FrameRecord {
            step: 1,
            t: 0.01,
            pos: vec![[0.0, 0.0]; vel.len()],
            vel,
            e_pot: 0.0,
            e_kin: 0.0,
        }
    }

    #[test]
    fn measured_temperature_and_equal_probability_shape_recover_gaussian_sample() {
        let mut rng = SplitMix64::new(11);
        let velocities = (0..24_000)
            .map(|_| {
                [
                    0.5_f64.sqrt() * rng.gaussian(),
                    0.5_f64.sqrt() * rng.gaussian(),
                ]
            })
            .collect();
        let frames = vec![synthetic_frame(velocities)];
        let temperature = speed_temperature(&frames);
        let shape = speed_shape_chi2_per_22(&frames, temperature);
        assert!((temperature - 0.5).abs() < 0.02, "{temperature}");
        assert!(shape < 2.0, "{shape}");
    }

    #[test]
    fn drift_uses_first_and_last_ten_percent_means() {
        let energies: Vec<f64> = (0..20).map(|i| 100.0 + i as f64 * 0.01).collect();
        let expected = ((100.185_f64) - 100.005).abs() / 100.0;
        assert!((windowed_secular_drift(&energies) - expected).abs() < 1e-12);
    }
}
