//! Triangular lattices at a prescribed two-dimensional number density.

use crate::system::Vec2;

/// Return a `rows x cols` triangular lattice and its rectangular periodic box.
///
/// The horizontal spacing is `a = sqrt(2 / (sqrt(3) rho))`, the row spacing
/// is `h = sqrt(3) a / 2`, and alternate rows are shifted by `a / 2`.
pub fn triangular_lattice(rows: usize, cols: usize, rho: f64) -> (Vec<Vec2>, Vec2) {
    assert!(rows > 0 && cols > 0);
    assert!(rho > 0.0);
    assert_eq!(
        rows % 2,
        0,
        "an even row count is required across periodic y"
    );
    let a = (2.0 / (3.0_f64.sqrt() * rho)).sqrt();
    let h = 0.5 * 3.0_f64.sqrt() * a;
    let lengths = [cols as f64 * a, rows as f64 * h];
    let mut positions = Vec::with_capacity(rows * cols);
    for j in 0..rows {
        for i in 0..cols {
            positions.push([(i as f64 + 0.5 * (j % 2) as f64) * a, j as f64 * h]);
        }
    }
    (positions, lengths)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ten_by_ten_contract_lattice_matches_sheet() {
        let (positions, lengths) = triangular_lattice(10, 10, 0.8);
        assert_eq!(positions.len(), 100);
        assert!((lengths[0] - 12.014_057_070_7).abs() < 1e-9);
        assert!((lengths[1] - 10.404_478_625_7).abs() < 1e-9);
        assert!((positions[10][0] - lengths[0] / 20.0).abs() < 1e-12);
        assert!(
            positions.iter().all(|p| {
                (0.0..lengths[0]).contains(&p[0]) && (0.0..lengths[1]).contains(&p[1])
            })
        );
    }

    #[test]
    fn larger_grids_keep_density_and_spacing() {
        let (_, small) = triangular_lattice(10, 10, 0.8);
        let (_, large) = triangular_lattice(20, 20, 0.8);
        assert!((large[0] / small[0] - 2.0).abs() < 1e-12);
        assert!((large[1] / small[1] - 2.0).abs() < 1e-12);
        assert!((400.0 / (large[0] * large[1]) - 0.8).abs() < 1e-12);
    }
}
