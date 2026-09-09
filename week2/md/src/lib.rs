//! Library for the `md` project.
//!
//! Lennard-Jones pair interactions in reduced units (`lj_energy`, `lj_force`)
//! and a two-atom molecular-dynamics core: the `System` state, an `Integrator`
//! trait shared by `Euler` and `VelocityVerlet`, and `run_experiment` for the
//! total-energy-error diagnostic.

pub mod fluid;
pub mod integrator;
pub mod lattice;
pub mod rng;
pub mod store;
pub mod system;

pub use integrator::{
    advance, run_experiment, Euler, FreeFlight, Integrator, VelocityVerlet,
};
pub use lattice::triangular_lattice;
pub use rng::SplitMix64;
pub use system::{accelerations, total_energy, BoxConfig, System, Vec2};

/// Returns the greeting printed by the binary.
///
/// Kept in the library so it can be unit-tested directly.
pub fn greeting() -> &'static str {
    "Hello, world!"
}

/// Lennard-Jones pair energy in reduced units.
///
/// `r` is the dimensionless separation in units of sigma; the returned value
/// is the pair potential energy in units of epsilon:
///
/// ```text
/// V(r) = 4 (r^-12 - r^-6)
/// ```
pub fn lj_energy(r: f64) -> f64 {
    4.0 * (r.powi(-12) - r.powi(-6))
}

/// Lennard-Jones pair force magnitude in reduced units.
///
/// `r` is the dimensionless separation in units of sigma; the returned value
/// is the radial force magnitude in units of epsilon/sigma, positive when
/// repulsive:
///
/// ```text
/// F(r) = -V'(r) = 4 (12 r^-13 - 6 r^-7)
/// ```
pub fn lj_force(r: f64) -> f64 {
    4.0 * (12.0 * r.powi(-13) - 6.0 * r.powi(-7))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greeting_is_hello_world() {
        assert_eq!(greeting(), "Hello, world!");
    }

    // Lennard-Jones pair interactions, in reduced units.
    //
    // Lengths are in units of sigma and energies in units of epsilon, so the
    // pair potential between two particles separated by `r` is
    //     V(r) = 4 (r^-12 - r^-6)
    // and the (outward, repulsive-positive) force magnitude is
    //     F(r) = -V'(r) = 4 (12 r^-13 - 6 r^-7).
    //
    // The well minimum sits at the equilibrium separation r_eq = 2^(1/6),
    // where V = -1 (i.e. -epsilon) and the force vanishes.

    #[test]
    fn lj_pair_energy_is_minus_one_at_equilibrium_separation() {
        let r_eq = 2f64.powf(1.0 / 6.0);
        let v = lj_energy(r_eq);
        assert!(
            (v + 1.0).abs() < 1e-12,
            "expected V(2^(1/6)) = -1 (in units of epsilon), got {v}"
        );
    }

    #[test]
    fn lj_force_matches_numerical_derivative_of_energy() {
        // Check the force against the numerical derivative of the energy,
        // the standard finite-difference identity (see "Testing a derivative"):
        //     F(r) = -dE/dr  ≈  -(E(r + h) - E(r - h)) / (2 h).
        //
        // Chosen parameters (reduced units):
        //   r0        = 2^(1/6)                  potential minimum
        //   separations r = r0 +/- {0.05, 0.15, 0.30}   (straddle r0, so the
        //                                              sign change of F is exercised)
        //   step h    = 1e-7                     central-difference step
        //   tolerance = 1e-6                     absolute, in units of epsilon/sigma
        let r0 = 2f64.powf(1.0 / 6.0);
        let offsets = [0.05, 0.15, 0.30];
        let h = 1e-7;
        let tol = 1e-6;

        for &delta in &offsets {
            for r in [r0 - delta, r0 + delta] {
                let d_energy_dr = (lj_energy(r + h) - lj_energy(r - h)) / (2.0 * h);
                let f = lj_force(r);
                let dev = (f + d_energy_dr).abs(); // F = -dE/dr
                assert!(
                    dev < tol,
                    "at r = {r}: |lj_force + dE/dr| = {dev:.3e} exceeds {tol:.1e} \
                     (lj_force = {f:.6}, dE/dr = {d_energy_dr:.6})"
                );
            }
        }
    }
}
