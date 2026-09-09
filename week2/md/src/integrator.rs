//! Time integrators for a `System`, sharing one `Integrator` trait.
//!
//! A concrete integrator implements a single time step given the current state
//! and a step `dt`; `advance` is the generic driver that works with any of
//! them, and `run_experiment` steps a state for many steps while recording the
//! total-energy error. Integrators are zero-sized so `&self` suffices — the
//! velocity-Verlet method recomputes the acceleration at both ends of a step
//! rather than storing state between calls.

use crate::system::{accelerations, total_energy, System};

/// Any object that can advance a `System` by one time step of length `dt`.
pub trait Integrator {
    fn step(&self, system: &mut System, dt: f64);
}

/// Drive `system` forward one step of length `dt` with any `Integrator`.
pub fn advance(method: &impl Integrator, system: &mut System, dt: f64) {
    method.step(system, dt);
}

/// A point atom with no forces: velocity stays constant, `x += v * dt`.
pub struct FreeFlight;

impl Integrator for FreeFlight {
    fn step(&self, system: &mut System, dt: f64) {
        for (x, v) in system.positions.iter_mut().zip(&system.velocities) {
            x[0] += dt * v[0];
            x[1] += dt * v[1];
        }
    }
}

/// Explicit (forward) Euler with Lennard-Jones accelerations.
///
/// `x_new = x + v * dt`, then `v_new = v + a(x) * dt`, with `a` evaluated at
/// the old positions. Not symplectic: total energy drifts upward over time.
pub struct Euler;

impl Integrator for Euler {
    fn step(&self, system: &mut System, dt: f64) {
        let a = accelerations(system);
        let n = system.n_atoms();
        for i in 0..n {
            for c in 0..2 {
                // Position step uses the *old* velocities (explicit Euler).
                system.positions[i][c] += dt * system.velocities[i][c];
            }
        }
        for i in 0..n {
            for c in 0..2 {
                system.velocities[i][c] += dt * a[i][c];
            }
        }
    }
}

/// Velocity-Verlet with Lennard-Jones accelerations.
///
/// `x_new = x + v * dt + (1/2) a(x) dt^2`, then
/// `v_new = v + (a(x) + a(x_new)) dt / 2`. Symplectic: the total energy error
/// stays bounded for a bound orbit.
pub struct VelocityVerlet;

impl Integrator for VelocityVerlet {
    fn step(&self, system: &mut System, dt: f64) {
        let a0 = accelerations(system);
        let dt2 = dt * dt;
        let n = system.n_atoms();
        for i in 0..n {
            for c in 0..2 {
                system.positions[i][c] +=
                    dt * system.velocities[i][c] + 0.5 * a0[i][c] * dt2;
            }
        }
        let a1 = accelerations(system);
        for i in 0..n {
            for c in 0..2 {
                system.velocities[i][c] += 0.5 * (a0[i][c] + a1[i][c]) * dt;
            }
        }
    }
}

/// Step a system for `steps` time steps of length `dt` with the given
/// integrator, recording `(t, E(t) - E(0))` after every step.
///
/// `E(0)` is the total energy of the initial state, so the first returned
/// entry is always `(0.0, 0.0)`.
pub fn run_experiment(
    method: &impl Integrator,
    system: &mut System,
    dt: f64,
    steps: usize,
) -> Vec<(f64, f64)> {
    let e0 = total_energy(system);
    let mut out = Vec::with_capacity(steps + 1);
    out.push((0.0, 0.0));
    for k in 1..=steps {
        advance(method, system, dt);
        let t = k as f64 * dt;
        out.push((t, total_energy(system) - e0));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::System;

    // The lesson's canonical free-flight test, verbatim in spirit.
    #[test]
    fn free_particle() {
        let mut system = System {
            positions: vec![[0.0, 0.0]],
            velocities: vec![[1.0, -2.0]],
        };
        advance(&FreeFlight, &mut system, 0.5); // Use the shared driver.
        assert_eq!(system.positions[0], [0.5, -1.0]);
        assert_eq!(system.velocities[0], [1.0, -2.0]); // Velocity unchanged.
    }

    /// The dimer experiment: two atoms, m = 1, released from rest at
    /// separation r = 1.2 (E0 = U(1.2)), dt = 0.01.
    fn dimer_initial() -> System {
        System::new(
            vec![[-0.6, 0.0], [0.6, 0.0]],
            vec![[0.0, 0.0], [0.0, 0.0]],
        )
    }

    fn max_abs(errors: &[(f64, f64)]) -> f64 {
        errors.iter().map(|(_, de)| de.abs()).fold(0.0, f64::max)
    }

    #[test]
    fn e0_is_potential_energy_at_separation_1_2() {
        let sys = dimer_initial();
        let expected = 4.0 * (1.2f64.powi(-12) - 1.2f64.powi(-6));
        assert!((total_energy(&sys) - expected).abs() < 1e-12);
    }

    #[test]
    fn velocity_verlet_conserves_energy_over_500_steps() {
        let mut sys = dimer_initial();
        let err = run_experiment(&VelocityVerlet, &mut sys, 0.01, 500);
        let max = max_abs(&err);
        assert!(
            max < 1e-3,
            "Verlet max energy error over 500 steps must be < 1e-3, got {max:.3e}"
        );
    }

    #[test]
    fn velocity_verlet_conserves_energy_over_5000_steps() {
        let mut sys = dimer_initial();
        let err = run_experiment(&VelocityVerlet, &mut sys, 0.01, 5000);
        let max = max_abs(&err);
        assert!(
            max < 1e-3,
            "Verlet max energy error over 5000 steps must stay < 1e-3, got {max:.3e}"
        );
    }

    #[test]
    fn euler_drifts_energy_over_500_steps() {
        let mut sys = dimer_initial();
        let err = run_experiment(&Euler, &mut sys, 0.01, 500);
        let final_err = err.last().expect("at least the t=0 entry").1;
        assert!(
            final_err > 0.5,
            "Euler final energy error over 500 steps must exceed 0.5, got {final_err:.3e}"
        );
    }
}
