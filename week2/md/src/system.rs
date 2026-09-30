//! Particle state and Lennard-Jones interactions in two dimensions.
//!
//! Open systems use the plain Lennard-Jones potential. Periodic systems use
//! the minimum image and a cut-and-shifted potential. The same interaction
//! kernel can enumerate pairs naively or through a periodic cell list.

use crate::{lj_energy, lj_force};

pub type Vec2 = [f64; 2];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ForceMethod {
    Naive,
    Cells,
}

impl ForceMethod {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Naive => "naive",
            Self::Cells => "cells",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "naive" => Some(Self::Naive),
            "cells" => Some(Self::Cells),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoxConfig {
    pub lengths: Vec2,
    pub cutoff: f64,
}

#[derive(Clone, Debug)]
pub struct System {
    pub positions: Vec<Vec2>,
    pub velocities: Vec<Vec2>,
    pub periodic: Option<BoxConfig>,
    pub force_method: ForceMethod,
    pub(crate) acceleration_cache: Option<Vec<Vec2>>,
}

impl Default for System {
    fn default() -> Self {
        Self {
            positions: Vec::new(),
            velocities: Vec::new(),
            periodic: None,
            force_method: ForceMethod::Naive,
            acceleration_cache: None,
        }
    }
}

impl System {
    pub fn new(positions: Vec<Vec2>, velocities: Vec<Vec2>) -> Self {
        assert_eq!(positions.len(), velocities.len());
        Self {
            positions,
            velocities,
            ..Self::default()
        }
    }

    pub fn with_box(positions: Vec<Vec2>, velocities: Vec<Vec2>, bc: BoxConfig) -> Self {
        Self::with_box_and_force(positions, velocities, bc, ForceMethod::Naive)
    }

    pub fn with_box_and_force(
        positions: Vec<Vec2>,
        velocities: Vec<Vec2>,
        bc: BoxConfig,
        force_method: ForceMethod,
    ) -> Self {
        assert_eq!(positions.len(), velocities.len());
        assert!(bc.lengths[0] > 2.0 * bc.cutoff);
        assert!(bc.lengths[1] > 2.0 * bc.cutoff);
        Self {
            positions,
            velocities,
            periodic: Some(bc),
            force_method,
            acceleration_cache: None,
        }
    }

    pub fn n_atoms(&self) -> usize {
        self.positions.len()
    }

    pub fn wrap_positions(&mut self) {
        if let Some(bc) = self.periodic {
            for p in &mut self.positions {
                p[0] = p[0].rem_euclid(bc.lengths[0]);
                p[1] = p[1].rem_euclid(bc.lengths[1]);
            }
        }
    }

    pub fn set_force_method(&mut self, method: ForceMethod) {
        self.force_method = method;
        self.acceleration_cache = None;
    }
}

#[derive(Clone, Debug)]
pub struct InteractionResult {
    pub accelerations: Vec<Vec2>,
    pub potential_energy: f64,
}

pub fn minimum_image_delta(a: Vec2, b: Vec2, lengths: Vec2) -> Vec2 {
    let mut d = [b[0] - a[0], b[1] - a[1]];
    for c in 0..2 {
        d[c] -= lengths[c] * (d[c] / lengths[c]).round();
    }
    d
}

fn accumulate_pair(
    result: &mut InteractionResult,
    i: usize,
    j: usize,
    d: Vec2,
    cutoff: Option<f64>,
) {
    let r2 = d[0] * d[0] + d[1] * d[1];
    if r2 == 0.0 {
        return;
    }
    if let Some(rc) = cutoff {
        if r2 >= rc * rc {
            return;
        }
    }
    let r = r2.sqrt();
    let scaled_force = lj_force(r) / r;
    let fx = scaled_force * d[0];
    let fy = scaled_force * d[1];
    result.accelerations[i][0] -= fx;
    result.accelerations[i][1] -= fy;
    result.accelerations[j][0] += fx;
    result.accelerations[j][1] += fy;
    result.potential_energy += match cutoff {
        Some(rc) => lj_energy(r) - lj_energy(rc),
        None => lj_energy(r),
    };
}

fn naive_interactions(system: &System) -> InteractionResult {
    let n = system.n_atoms();
    let mut result = InteractionResult {
        accelerations: vec![[0.0; 2]; n],
        potential_energy: 0.0,
    };
    for i in 0..n {
        for j in (i + 1)..n {
            let (d, cutoff) = match system.periodic {
                Some(bc) => (
                    minimum_image_delta(system.positions[i], system.positions[j], bc.lengths),
                    Some(bc.cutoff),
                ),
                None => (
                    [
                        system.positions[j][0] - system.positions[i][0],
                        system.positions[j][1] - system.positions[i][1],
                    ],
                    None,
                ),
            };
            accumulate_pair(&mut result, i, j, d, cutoff);
        }
    }
    result
}

fn cell_interactions(system: &System, bc: BoxConfig) -> InteractionResult {
    let n = system.n_atoms();
    let nx = ((bc.lengths[0] / bc.cutoff).floor() as usize).max(1);
    let ny = ((bc.lengths[1] / bc.cutoff).floor() as usize).max(1);
    let wx = bc.lengths[0] / nx as f64;
    let wy = bc.lengths[1] / ny as f64;
    debug_assert!(wx >= bc.cutoff && wy >= bc.cutoff);

    let mut cells = vec![Vec::<usize>::new(); nx * ny];
    let mut atom_cells = Vec::with_capacity(n);
    for (i, p) in system.positions.iter().enumerate() {
        let cx = ((p[0].rem_euclid(bc.lengths[0]) / wx).floor() as usize).min(nx - 1);
        let cy = ((p[1].rem_euclid(bc.lengths[1]) / wy).floor() as usize).min(ny - 1);
        let cid = cy * nx + cx;
        cells[cid].push(i);
        atom_cells.push((cx, cy));
    }

    let mut result = InteractionResult {
        accelerations: vec![[0.0; 2]; n],
        potential_energy: 0.0,
    };
    for i in 0..n {
        let (cx, cy) = atom_cells[i];
        let mut neighbours = Vec::with_capacity(9);
        for oy in -1isize..=1 {
            for ox in -1isize..=1 {
                let x = (cx as isize + ox).rem_euclid(nx as isize) as usize;
                let y = (cy as isize + oy).rem_euclid(ny as isize) as usize;
                let cid = y * nx + x;
                if !neighbours.contains(&cid) {
                    neighbours.push(cid);
                }
            }
        }
        for cid in neighbours {
            for &j in &cells[cid] {
                if j <= i {
                    continue;
                }
                let d = minimum_image_delta(system.positions[i], system.positions[j], bc.lengths);
                accumulate_pair(&mut result, i, j, d, Some(bc.cutoff));
            }
        }
    }
    result
}

pub fn interactions_with_method(system: &System, method: ForceMethod) -> InteractionResult {
    match (system.periodic, method) {
        (Some(bc), ForceMethod::Cells) => cell_interactions(system, bc),
        _ => naive_interactions(system),
    }
}

pub fn interactions(system: &System) -> InteractionResult {
    interactions_with_method(system, system.force_method)
}

pub fn accelerations(system: &System) -> Vec<Vec2> {
    interactions(system).accelerations
}

pub fn potential_energy(system: &System) -> f64 {
    interactions(system).potential_energy
}

pub fn kinetic_energy(system: &System) -> f64 {
    system
        .velocities
        .iter()
        .map(|v| 0.5 * (v[0] * v[0] + v[1] * v[1]))
        .sum()
}

pub fn total_energy(system: &System) -> f64 {
    kinetic_energy(system) + potential_energy(system)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn box_config(lengths: Vec2) -> BoxConfig {
        BoxConfig {
            lengths,
            cutoff: 2.5,
        }
    }

    fn rest_pair(separation: f64) -> System {
        System::new(
            vec![[-separation / 2.0, 0.0], [separation / 2.0, 0.0]],
            vec![[0.0; 2]; 2],
        )
    }

    #[test]
    fn equilibrium_pair_has_well_depth_and_zero_acceleration() {
        let system = rest_pair(2f64.powf(1.0 / 6.0));
        assert!((total_energy(&system) + 1.0).abs() < 1e-12);
        assert!(
            accelerations(&system)
                .iter()
                .flatten()
                .all(|x| x.abs() < 1e-10)
        );
    }

    #[test]
    fn internal_forces_sum_to_zero() {
        let positions = vec![[0.1, 0.2], [1.3, 0.4], [4.9, 5.8], [7.8, 0.3], [9.8, 5.9]];
        let system = System::with_box(
            positions.clone(),
            vec![[0.0; 2]; positions.len()],
            box_config([10.0, 6.0]),
        );
        let a = accelerations(&system);
        let sum = a.iter().fold([0.0, 0.0], |mut s, v| {
            s[0] += v[0];
            s[1] += v[1];
            s
        });
        assert!(sum[0].abs() < 1e-9 && sum[1].abs() < 1e-9, "{sum:?}");
    }

    #[test]
    fn shifted_potential_is_continuous_just_inside_cutoff() {
        let rc = 2.5;
        let system = System::with_box(
            vec![[0.0, 0.0], [rc - 1e-7, 0.0]],
            vec![[0.0; 2]; 2],
            box_config([10.0, 8.0]),
        );
        assert!(potential_energy(&system).abs() < 1e-7);
    }

    fn assert_methods_match(positions: Vec<Vec2>, lengths: Vec2) {
        let mut system = System::with_box(
            positions.clone(),
            vec![[0.0; 2]; positions.len()],
            box_config(lengths),
        );
        let naive = interactions_with_method(&system, ForceMethod::Naive);
        system.set_force_method(ForceMethod::Cells);
        let cells = interactions(&system);
        assert!((naive.potential_energy - cells.potential_energy).abs() < 1e-10);
        for (a, b) in naive.accelerations.iter().zip(&cells.accelerations) {
            assert!((a[0] - b[0]).abs() < 1e-9, "{a:?} != {b:?}");
            assert!((a[1] - b[1]).abs() < 1e-9, "{a:?} != {b:?}");
        }
    }

    #[test]
    fn cells_match_naive_for_perturbed_boundary_and_cutoff_cases() {
        assert_methods_match(
            vec![
                [0.05, 0.2],
                [9.95, 0.25],
                [2.55, 0.2],
                [5.01, 5.9],
                [5.06, 0.1],
                [7.41, 3.0],
            ],
            [10.0, 6.0],
        );
    }

    #[test]
    fn cells_match_naive_in_two_cell_wide_box() {
        assert_methods_match(
            vec![[0.1, 0.1], [2.0, 0.2], [4.9, 0.15], [0.2, 4.8], [4.8, 4.9]],
            [5.1, 5.1],
        );
    }
}
