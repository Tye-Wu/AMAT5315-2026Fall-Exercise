//! Run pipeline for the two-dimensional Lennard-Jones fluid.

use crate::integrator::{advance, VelocityVerlet};
use crate::lattice::triangular_lattice;
use crate::rng::SplitMix64;
use crate::system::{BoxConfig, System, Vec2};

/// Everything needed to run the fluid simulation.
#[derive(Clone, Debug)]
pub struct RunConfig {
    pub atoms: usize,
    pub rho: f64,
    pub temperature: f64,
    pub seed: u64,
    pub dt: f64,
    pub cutoff: f64,
    pub eq: usize,
    pub steps: usize,
    pub save_every: usize,
}

impl Default for RunConfig {
    fn default() -> Self {
        RunConfig {
            atoms: 100,
            rho: 0.8,
            temperature: 0.5,
            seed: 42,
            dt: 0.005,
            cutoff: 2.5,
            eq: 4000,
            steps: 5000,
            save_every: 25,
        }
    }
}

/// Periodic box side for a run: `L = sqrt(N / rho)`.
pub fn box_length(cfg: &RunConfig) -> f64 {
    (cfg.atoms as f64 / cfg.rho).sqrt()
}

/// Instantaneous kinetic temperature `T = <v^2> / d` with d = 2 (m = 1).
pub fn kinetic_temperature(sys: &System) -> f64 {
    let sum: f64 = sys.velocities.iter().map(|v| v[0] * v[0] + v[1] * v[1]).sum();
    sum / (2.0 * sys.n_atoms() as f64)
}

/// Rescale every velocity so the kinetic temperature equals `target`.
pub fn rescale_velocities(sys: &mut System, target: f64) {
    let t = kinetic_temperature(sys);
    if t > 0.0 {
        let alpha = (target / t).sqrt();
        for v in &mut sys.velocities {
            v[0] *= alpha;
            v[1] *= alpha;
        }
    }
}

/// Overwrite velocities with Gaussian draws, remove net momentum, then rescale
/// to the target temperature.
pub fn seed_velocities(sys: &mut System, rng: &mut SplitMix64, target: f64) {
    let n = sys.n_atoms();
    let mut vx = Vec::with_capacity(n);
    let mut vy = Vec::with_capacity(n);
    for _ in 0..n {
        vx.push(rng.gaussian());
        vy.push(rng.gaussian());
    }
    let mx = vx.iter().sum::<f64>() / n as f64;
    let my = vy.iter().sum::<f64>() / n as f64;
    for i in 0..n {
        sys.velocities[i] = [vx[i] - mx, vy[i] - my];
    }
    rescale_velocities(sys, target);
}

/// One saved production snapshot.
#[derive(Clone, Debug)]
pub struct FrameData {
    pub t: f64,
    pub positions: Vec<Vec2>,
    pub velocities: Vec<Vec2>,
}

/// The outcome of a fluid run: config, box, initial energy, saved frames.
#[derive(Clone, Debug)]
pub struct RunRecord {
    pub cfg: RunConfig,
    pub box_len: f64,
    pub e0: f64,
    pub frames: Vec<FrameData>,
}

/// Run the equilibration + production pipeline (velocity-Verlet).
///
/// Equilibration rescales velocities to `T` every 50 steps; production has no
/// thermostat. One `FrameData` is saved every `save_every` production steps,
/// so `cfg.steps / cfg.save_every` frames are produced (200 by default).
pub fn run_fluid(cfg: &RunConfig) -> RunRecord {
    let l = box_length(cfg);
    let side = (cfg.atoms as f64).sqrt() as usize;
    let mut pos = triangular_lattice(side.max(1), side.max(1), l);
    while pos.len() < cfg.atoms {
        let i = pos.len() as f64;
        pos.push([(i * 0.137).rem_euclid(l), (i * 0.79).rem_euclid(l)]);
    }
    pos.truncate(cfg.atoms);

    let vel = vec![[0.0; 2]; cfg.atoms];
    let mut sys = System::with_box(pos, vel, BoxConfig { length: l, cutoff: cfg.cutoff });
    let mut rng = SplitMix64::new(cfg.seed);
    seed_velocities(&mut sys, &mut rng, cfg.temperature);

    let mut step = 0usize;
    while step < cfg.eq {
        advance(&VelocityVerlet, &mut sys, cfg.dt);
        step += 1;
        if step % 50 == 0 {
            rescale_velocities(&mut sys, cfg.temperature);
        }
    }

    let e0 = crate::system::total_energy(&sys);
    let mut frames = Vec::with_capacity(cfg.steps / cfg.save_every);
    step = 0;
    while step < cfg.steps {
        advance(&VelocityVerlet, &mut sys, cfg.dt);
        step += 1;
        if step % cfg.save_every == 0 {
            frames.push(FrameData {
                t: step as f64 * cfg.dt,
                positions: sys.positions.clone(),
                velocities: sys.velocities.clone(),
            });
        }
    }
    RunRecord {
        cfg: cfg.clone(),
        box_len: l,
        e0,
        frames,
    }
}

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
        let expected = (100.0_f64 / 0.8).sqrt();
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

