//! The state of a two-dimensional point-atom system and the Lennard-Jones
//! physics that acts on it.
//!
//! `System` is deliberately plain data — positions, velocities, and an
//! *optional* periodic box. With no box (`periodic: None`) the physics are the
//! exact all-pairs interactions used by the two-atom dimer; with a box they
//! use minimum-image periodic boundaries and a cut-and-shifted potential at
//! the cutoff. The physics (accelerations, total energy) are free functions
//! over that state, built on `lj_energy` and `lj_force`.

use crate::{lj_energy, lj_force};

/// A 2D vector, used for both positions and velocities.
pub type Vec2 = [f64; 2];

/// Periodic simulation box: side `length` and interaction `cutoff`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoxConfig {
    /// Box side length (reduced units of sigma).
    pub length: f64,
    /// Shifted interaction cutoff r_c.
    pub cutoff: f64,
}

/// A set of equal-mass point atoms moving in the plane.
///
/// Mass is 1 (reduced units), so force equals acceleration.
#[derive(Clone, Debug)]
pub struct System {
    /// Positions, one `[x, y]` entry per atom.
    pub positions: Vec<Vec2>,
    /// Velocities, one `[vx, vy]` entry per atom, parallel to `positions`.
    pub velocities: Vec<Vec2>,
    /// `Some(box)` enables periodic, minimum-image interactions; `None` keeps
    /// the exact all-pairs behaviour used by the two-atom dimer.
    pub periodic: Option<BoxConfig>,
}

impl Default for System {
    fn default() -> Self {
        System {
            positions: Vec::new(),
            velocities: Vec::new(),
            periodic: None,
        }
    }
}

impl System {
    /// Build a box-less system from parallel position and velocity lists.
    pub fn new(positions: Vec<Vec2>, velocities: Vec<Vec2>) -> Self {
        assert_eq!(
            positions.len(),
            velocities.len(),
            "positions and velocities must have the same length"
        );
        System {
            positions,
            velocities,
            periodic: None,
        }
    }

    /// Build a system with periodic boundary conditions from parallel lists.
    pub fn with_box(positions: Vec<Vec2>, velocities: Vec<Vec2>, bc: BoxConfig) -> Self {
        assert_eq!(
            positions.len(),
            velocities.len(),
            "positions and velocities must have the same length"
        );
        System {
            positions,
            velocities,
            periodic: Some(bc),
        }
    }

    /// Number of atoms in the system.
    pub fn n_atoms(&self) -> usize {
        self.positions.len()
    }
}

/// Minimum-image displacement from `a` to `b` inside a periodic box of side
/// `len`, wrapped into `(-len/2, len/2]`.
fn displacement(a: Vec2, b: Vec2, len: f64) -> Vec2 {
    let mut dx = b[0] - a[0];
    let mut dy = b[1] - a[1];
    dx -= len * (dx / len).round();
    dy -= len * (dy / len).round();
    [dx, dy]
}

/// Total energy `E = KE + U`: kinetic energy of all atoms plus the pairwise
/// Lennard-Jones potential, in reduced units (m = 1, epsilon = 1).
///
/// With a periodic box the pair potential is cut-and-shifted at `r_c`:
/// `V_shift(r) = V(r) - V(r_c)` for `r < r_c`, else 0. Without a box the
/// exact `lj_energy` is used for every pair (the dimer behaviour).
pub fn total_energy(system: &System) -> f64 {
    let ke: f64 = system
        .velocities
        .iter()
        .map(|v| 0.5 * (v[0] * v[0] + v[1] * v[1]))
        .sum();
    let n = system.n_atoms();
    let mut pe = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            let r = match system.periodic {
                Some(bc) => {
                    let d = displacement(system.positions[i], system.positions[j], bc.length);
                    (d[0] * d[0] + d[1] * d[1]).sqrt()
                }
                None => {
                    let dx = system.positions[j][0] - system.positions[i][0];
                    let dy = system.positions[j][1] - system.positions[i][1];
                    (dx * dx + dy * dy).sqrt()
                }
            };
            match system.periodic {
                Some(bc) if r < bc.cutoff => pe += lj_energy(r) - lj_energy(bc.cutoff),
                Some(_) => {}
                None => pe += lj_energy(r),
            }
        }
    }
    ke + pe
}

/// Acceleration of every atom (equal to force, since m = 1), from the
/// Lennard-Jones pair forces between all pairs.
///
/// The LJ force magnitude `lj_force(r)` is positive when repulsive, so atom `j`
/// is pushed away from atom `i` along the pair vector, and atom `i` feels the
/// equal and opposite push. With a periodic box the pair vector is the
/// minimum image and pairs beyond `r_c` are omitted.
pub fn accelerations(system: &System) -> Vec<Vec2> {
    let n = system.n_atoms();
    let mut acc = vec![[0.0; 2]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let (dx, dy) = match system.periodic {
                Some(bc) => {
                    let d = displacement(system.positions[i], system.positions[j], bc.length);
                    (d[0], d[1])
                }
                None => (
                    system.positions[j][0] - system.positions[i][0],
                    system.positions[j][1] - system.positions[i][1],
                ),
            };
            let r = (dx * dx + dy * dy).sqrt();
            let inside = match system.periodic {
                Some(bc) => r > 0.0 && r < bc.cutoff,
                None => r > 0.0,
            };
            if inside {
                let f = lj_force(r) / r; // force magnitude times the unit vector
                acc[j][0] += f * dx;
                acc[j][1] += f * dy;
                acc[i][0] -= f * dx;
                acc[i][1] -= f * dy;
            }
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

    // --- Periodic box + shifted cutoff (Task 1) ---

    fn box_cfg() -> BoxConfig {
        BoxConfig { length: 10.0, cutoff: 2.5 }
    }

    #[test]
    fn net_force_is_zero_under_periodic_boundaries() {
        let mut pos = Vec::new();
        for i in 0..12u32 {
            let x = (f64::from(i) * 1.31).rem_euclid(10.0);
            let y = (f64::from(i) * 0.77).rem_euclid(10.0);
            pos.push([x, y]);
        }
        let vel = vec![[0.0; 2]; pos.len()];
        let sys = System::with_box(pos, vel, box_cfg());
        let a = accelerations(&sys);
        let sx: f64 = a.iter().map(|v| v[0]).sum();
        let sy: f64 = a.iter().map(|v| v[1]).sum();
        assert!(sx.abs() < 1e-9 && sy.abs() < 1e-9, "net accel {sx} {sy}");
    }

    #[test]
    fn potential_is_continuous_just_inside_cutoff() {
        // Probe just inside rc: with the cut-and-shift the pair energy ~ 0 there.
        let r = 2.5 - 1e-6;
        let sys = System::with_box(
            vec![[0.0, 0.0], [r, 0.0]],
            vec![[0.0, 0.0], [0.0, 0.0]],
            box_cfg(),
        );
        let e = total_energy(&sys); // KE = 0, so this is the pair energy
        assert!(e.abs() < 1e-4, "energy at rc-1e-6 should be ~0 (shifted), got {e}");
    }

    #[test]
    fn shifted_potential_matches_exact_far_below_cutoff() {
        let sys = System::with_box(
            vec![[0.0, 0.0], [1.2, 0.0]],
            vec![[0.0, 0.0], [0.0, 0.0]],
            box_cfg(),
        );
        let shifted = total_energy(&sys);
        let exact = 4.0 * (1.2f64.powi(-12) - 1.2f64.powi(-6));
        let shift = 4.0 * (2.5f64.powi(-12) - 2.5f64.powi(-6));
        assert!((shifted - (exact - shift)).abs() < 1e-9);
    }

    #[test]
    fn no_box_keeps_exact_two_body_energy() {
        let sys = rest_pair(1.2);
        let expected = 4.0 * (1.2f64.powi(-12) - 1.2f64.powi(-6));
        assert!((total_energy(&sys) - expected).abs() < 1e-9);
    }
}
