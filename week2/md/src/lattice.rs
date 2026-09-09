//! Deterministic initial configurations: a triangular lattice generator.

use crate::system::Vec2;

/// Generate a `rows` x `cols` triangular lattice wrapped into a periodic box of
/// side `length`.
///
/// Row `iy` is offset by half a column spacing so neighbouring rows nest, the
/// classic triangular (hexagonal) arrangement; every point is wrapped into
/// `[0, length)^2` so it is a valid periodic-box configuration.
pub fn triangular_lattice(rows: usize, cols: usize, length: f64) -> Vec<Vec2> {
    let a = length / cols as f64; // column spacing, ~1.12 for the defaults
    let dy = a * 0.866_025_403_784_438_6; // row spacing = a * sqrt(3)/2
    let mut out = Vec::with_capacity(rows * cols);
    for iy in 0..rows {
        for ix in 0..cols {
            let x = ((ix as f64 + 0.5 * (iy % 2) as f64) * a).rem_euclid(length);
            let y = (iy as f64 * dy).rem_euclid(length);
            out.push([x, y]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn min_pair_dist(pts: &[[f64; 2]]) -> f64 {
        let mut dmin = f64::INFINITY;
        for i in 0..pts.len() {
            for j in (i + 1)..pts.len() {
                let d2 = (pts[i][0] - pts[j][0]).powi(2) + (pts[i][1] - pts[j][1]).powi(2);
                dmin = dmin.min(d2);
            }
        }
        dmin.sqrt()
    }

    #[test]
    fn ten_by_ten_lattice_has_100_points_in_box() {
        let len = 11.1803398875; // sqrt(100 / 0.8)
        let pts = triangular_lattice(10, 10, len);
        assert_eq!(pts.len(), 100);
        for p in &pts {
            assert!((0.0..len).contains(&p[0]), "x {}", p[0]);
            assert!((0.0..len).contains(&p[1]), "y {}", p[1]);
        }
        // Spacing about L/10 ~ 1.12: comfortably above the LJ core.
        assert!(min_pair_dist(&pts) > 0.9, "min dist {}", min_pair_dist(&pts));
    }

    #[test]
    fn eight_by_eight_lattice_has_64_points() {
        let pts = triangular_lattice(8, 8, 8.0);
        assert_eq!(pts.len(), 64);
    }
}
