//! Library for the `md` project.

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
///     V(r) = 4 (r^-12 - r^-6)
pub fn lj_energy(r: f64) -> f64 {
    4.0 * (r.powi(-12) - r.powi(-6))
}

/// Lennard-Jones pair force magnitude in reduced units.
///
/// `r` is the dimensionless separation in units of sigma; the returned value
/// is the radial force magnitude in units of epsilon/sigma, positive when
/// repulsive:
///
///     F(r) = -V'(r) = 4 (12 r^-13 - 6 r^-7)
///
/// Not yet implemented (see the failing force test).
pub fn lj_force(_r: f64) -> f64 {
    todo!("lj_force not yet implemented")
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
    fn lj_force_vanishes_at_equilibrium_and_is_repulsive_inside() {
        let r_eq = 2f64.powf(1.0 / 6.0);
        let f_eq = lj_force(r_eq);
        assert!(
            f_eq.abs() < 1e-9,
            "expected F(2^(1/6)) = 0, got {f_eq}"
        );

        // At r = 1 (one sigma), F = 4(12 - 6) = 24 in units of epsilon/sigma.
        let f_one = lj_force(1.0);
        assert!(
            (f_one - 24.0).abs() < 1e-9,
            "expected F(1) = 24 (epsilon/sigma), got {f_one}"
        );
    }
}
