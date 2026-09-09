//! The state of a two-dimensional point-atom system and the Lennard-Jones
//! two-body physics that acts on it.
//!
//! `System` is deliberately plain data — positions and velocities — so any
//! integrator can drive it. The physics (accelerations, total energy) are free
//! functions over that state, built on `lj_energy` and `lj_force`.

use crate::{lj_energy, lj_force};

/// A 2D vector, used for both positions and velocities.
pub type Vec2 = [f64; 2];

/// A set of equal-mass point atoms moving in the plane.
///
/// Mass is 1 (reduced units), so force equals acceleration.
pub struct System {
    /// Positions, one `[x, y]` entry per atom.
    pub positions: Vec<Vec2>,
    /// Velocities, one `[vx, vy]` entry per atom, parallel to `positions`.
    pub velocities: Vec<Vec2>,
}

impl System {
    /// Build a system from parallel position and velocity lists.
    pub fn new(positions: Vec<Vec2>, velocities: Vec<Vec2>) -> Self {
        assert_eq!(
            positions.len(),
            velocities.len(),
            "positions and velocities must have the same length"
        );
        System {
            positions,
            velocities,
        }
    }

    /// Number of atoms in the system.
    pub fn n_atoms(&self) -> usize {
        self.positions.len()
    }
}

/// Squared distance between two position vectors.
fn dist2(a: Vec2, b: Vec2) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    dx * dx + dy * dy
}

/// Pairwise Lennard-Jones potential energy of the whole system
/// (one `lj_energy` term per pair of atoms).
fn potential_energy(system: &System) -> f64 {
    let n = system.n_atoms();
    let mut pe = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            pe += lj_energy(dist2(system.positions[i], system.positions[j]).sqrt());
        }
    }
    pe
}

/// Total energy `E = KE + U`: kinetic energy of all atoms plus the pairwise
/// Lennard-Jones potential, in reduced units (m = 1, epsilon = 1).
pub fn total_energy(system: &System) -> f64 {
    let ke: f64 = system
        .velocities
        .iter()
        .map(|v| 0.5 * (v[0] * v[0] + v[1] * v[1]))
        .sum();
    ke + potential_energy(system)
}

/// Acceleration of every atom (equal to force, since m = 1), from the
/// Lennard-Jones pair forces between all pairs.
///
/// The LJ force magnitude `lj_force(r)` is positive when repulsive, so atom `j`
/// is pushed away from atom `i` along the vector `positions[j] - positions[i]`,
/// and atom `i` feels the equal and opposite push.
pub fn accelerations(system: &System) -> Vec<Vec2> {
    let n = system.n_atoms();
    let mut acc = vec![[0.0; 2]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = system.positions[j][0] - system.positions[i][0];
            let dy = system.positions[j][1] - system.positions[i][1];
            let r = (dx * dx + dy * dy).sqrt();
            let f = lj_force(r) / r; // force magnitude times the unit vector
            acc[j][0] += f * dx;
            acc[j][1] += f * dy;
            acc[i][0] -= f * dx;
            acc[i][1] -= f * dy;
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r0() -> f64 {
        2f64.powf(1.0 / 6.0)
    }

    fn rest_pair(sep: f64) -> System {
        System::new(
            vec![[-sep / 2.0, 0.0], [sep / 2.0, 0.0]],
            vec![[0.0, 0.0], [0.0, 0.0]],
        )
    }

    #[test]
    fn accelerations_vanish_at_equilibrium_separation() {
        // At r0 the pair force is zero, so a rest pair stays put.
        let sys = rest_pair(r0());
        let a = accelerations(&sys);
        for ai in &a {
            assert!(ai[0].abs() < 1e-9 && ai[1].abs() < 1e-9, "accel {ai:?}");
        }
        // ...and the total energy is the well depth: V(r0) = -1.
        assert!((total_energy(&sys) + 1.0).abs() < 1e-9);
    }

    #[test]
    fn accelerations_satisfy_newtons_third_law_and_repel_inside_r0() {
        // At r = 0.9 < r0 the force is repulsive: each atom is pushed away
        // from the other, with equal and opposite accelerations.
        let sys = rest_pair(0.9);
        let a = accelerations(&sys);
        assert!(a[0][0] < 0.0, "left atom should be pushed left, got {:?}", a[0]);
        assert!(a[1][0] > 0.0, "right atom should be pushed right, got {:?}", a[1]);
        assert!((a[0][0] + a[1][0]).abs() < 1e-12); // momentum conserved
        assert!(a[0][1].abs() < 1e-12 && a[1][1].abs() < 1e-12);
    }
}
