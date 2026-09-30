//! Equilibration and production pipeline for the two-dimensional LJ fluid.

use crate::integrator::{VelocityVerlet, advance};
use crate::lattice::triangular_lattice;
use crate::rng::SplitMix64;
use crate::system::{BoxConfig, ForceMethod, System, Vec2, interactions, kinetic_energy};

#[derive(Clone, Debug)]
pub struct RunConfig {
    pub n: usize,
    pub rho: f64,
    pub temperature: f64,
    pub dt: f64,
    pub eq_steps: usize,
    pub steps: usize,
    pub sample_every: usize,
    pub seed: u64,
    pub cutoff: f64,
    pub force_method: ForceMethod,
    pub ramp_to: Option<f64>,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            n: 100,
            rho: 0.8,
            temperature: 0.5,
            dt: 0.01,
            eq_steps: 2_000,
            steps: 10_000,
            sample_every: 50,
            seed: 2026,
            cutoff: 2.5,
            force_method: ForceMethod::Cells,
            ramp_to: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct FrameData {
    pub step: usize,
    pub t: f64,
    pub positions: Vec<Vec2>,
    pub velocities: Vec<Vec2>,
    pub e_pot: f64,
    pub e_kin: f64,
}

#[derive(Clone, Debug)]
pub struct RunRecord {
    pub cfg: RunConfig,
    pub box_lengths: Vec2,
    pub frames: Vec<FrameData>,
}

/// Thermostat temperature after removing the two centre-of-mass components.
pub fn thermostat_temperature(system: &System) -> f64 {
    let dof = 2 * system.n_atoms() - 2;
    2.0 * kinetic_energy(system) / dof as f64
}

pub fn rescale_velocities(system: &mut System, target: f64) {
    let current = thermostat_temperature(system);
    assert!(target >= 0.0 && current > 0.0);
    let factor = (target / current).sqrt();
    for v in &mut system.velocities {
        v[0] *= factor;
        v[1] *= factor;
    }
}

pub fn seed_velocities(system: &mut System, rng: &mut SplitMix64, target: f64) {
    let scale = target.sqrt();
    for v in &mut system.velocities {
        v[0] = scale * rng.gaussian();
        v[1] = scale * rng.gaussian();
    }
    let n = system.n_atoms() as f64;
    let mean = system.velocities.iter().fold([0.0, 0.0], |mut sum, v| {
        sum[0] += v[0];
        sum[1] += v[1];
        sum
    });
    for v in &mut system.velocities {
        v[0] -= mean[0] / n;
        v[1] -= mean[1] / n;
    }
    rescale_velocities(system, target);
}

/// Target temperature at a production step. Step zero is the starting
/// temperature and `cfg.steps` is exactly `ramp_to`.
pub fn production_target(cfg: &RunConfig, step: usize) -> Option<f64> {
    cfg.ramp_to.map(|final_temperature| {
        let fraction = step.min(cfg.steps) as f64 / cfg.steps.max(1) as f64;
        cfg.temperature + fraction * (final_temperature - cfg.temperature)
    })
}

fn grid_side(n: usize) -> usize {
    let side = (n as f64).sqrt() as usize;
    assert_eq!(side * side, n, "--n must be a perfect square");
    assert_eq!(side % 2, 0, "sqrt(--n) must be even for periodic rows");
    side
}

pub fn run_fluid(cfg: &RunConfig) -> RunRecord {
    assert!(cfg.n >= 4);
    assert!(cfg.sample_every > 0);
    let side = grid_side(cfg.n);
    let (positions, box_lengths) = triangular_lattice(side, side, cfg.rho);
    let velocities = vec![[0.0; 2]; cfg.n];
    let bc = BoxConfig {
        lengths: box_lengths,
        cutoff: cfg.cutoff,
    };
    let mut system = System::with_box_and_force(positions, velocities, bc, cfg.force_method);
    let mut rng = SplitMix64::new(cfg.seed);
    seed_velocities(&mut system, &mut rng, cfg.temperature);

    for step in 1..=cfg.eq_steps {
        advance(&VelocityVerlet, &mut system, cfg.dt);
        if step % 50 == 0 {
            rescale_velocities(&mut system, cfg.temperature);
        }
    }

    let mut frames = Vec::with_capacity(cfg.steps / cfg.sample_every);
    for step in 1..=cfg.steps {
        advance(&VelocityVerlet, &mut system, cfg.dt);
        if cfg.ramp_to.is_some() && (step % 50 == 0 || step == cfg.steps) {
            rescale_velocities(&mut system, production_target(cfg, step).unwrap());
        }
        if step % cfg.sample_every == 0 {
            let interaction = interactions(&system);
            frames.push(FrameData {
                step,
                t: step as f64 * cfg.dt,
                positions: system.positions.clone(),
                velocities: system.velocities.clone(),
                e_pot: interaction.potential_energy,
                e_kin: kinetic_energy(&system),
            });
        }
    }

    RunRecord {
        cfg: cfg.clone(),
        box_lengths,
        frames,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_the_contract_run() {
        let c = RunConfig::default();
        assert_eq!(c.n, 100);
        assert_eq!(c.rho, 0.8);
        assert_eq!(c.temperature, 0.5);
        assert_eq!(c.dt, 0.01);
        assert_eq!(c.eq_steps, 2_000);
        assert_eq!(c.steps, 10_000);
        assert_eq!(c.sample_every, 50);
        assert_eq!(c.seed, 2026);
        assert_eq!(c.force_method, ForceMethod::Cells);
        assert_eq!(c.ramp_to, None);
    }

    #[test]
    fn seeding_removes_centre_of_mass_and_sets_thermostat_temperature() {
        let mut system = System::new(vec![[0.0; 2]; 100], vec![[0.0; 2]; 100]);
        seed_velocities(&mut system, &mut SplitMix64::new(2026), 0.5);
        let mean = system.velocities.iter().fold([0.0, 0.0], |mut sum, v| {
            sum[0] += v[0];
            sum[1] += v[1];
            sum
        });
        assert!(mean[0].abs() < 1e-12 && mean[1].abs() < 1e-12);
        assert!((thermostat_temperature(&system) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn heating_schedule_has_exact_endpoints_and_midpoint() {
        let c = RunConfig {
            temperature: 0.2,
            ramp_to: Some(1.2),
            steps: 20_000,
            ..RunConfig::default()
        };
        assert_eq!(production_target(&c, 0), Some(0.2));
        assert!((production_target(&c, 10_000).unwrap() - 0.7).abs() < 1e-12);
        assert_eq!(production_target(&c, 20_000), Some(1.2));
    }

    #[test]
    fn sampling_excludes_step_zero_and_writes_exact_count() {
        let c = RunConfig {
            n: 36,
            eq_steps: 50,
            steps: 100,
            sample_every: 10,
            ..RunConfig::default()
        };
        let record = run_fluid(&c);
        assert_eq!(record.frames.len(), 10);
        assert_eq!(record.frames.first().unwrap().step, 10);
        assert_eq!(record.frames.last().unwrap().step, 100);
        assert_eq!(record.frames.last().unwrap().t, 1.0);
    }
}
