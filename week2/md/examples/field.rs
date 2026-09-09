//! Dump the Lennard-Jones pair energy and force on a grid around one atom.
//!
//! The central atom sits at the origin. For each grid point `(x, y)` — the
//! position of a second atom at separation `r` — the example prints
//!
//!     x, y, V, Fx, Fy
//!
//! where `V = md::lj_energy(r)` and `Fx, Fy` are the components of the pair
//! force `md::lj_force(r)` along the radial direction (pointing away from the
//! central atom when repulsive). All quantities are in reduced units
//! (epsilon = sigma = 1). `week2/plot_field.py` consumes this on stdout.
//!
//! Run it directly with:
//!
//!     cargo run --quiet --example field --manifest-path week2/md/Cargo.toml

use md::{lj_energy, lj_force};

fn main() {
    // Plot window: the square [-half, half] x [-half, half].
    let half = 2.4_f64;
    // Points per side of the grid.
    let n = 301_u32;

    println!("x,y,V,Fx,Fy");
    for iy in 0..n {
        let y = -half + (2.0 * half) * f64::from(iy) / f64::from(n - 1);
        for ix in 0..n {
            let x = -half + (2.0 * half) * f64::from(ix) / f64::from(n - 1);
            let r = (x * x + y * y).sqrt();

            let (v, fx, fy) = if r > 0.0 {
                // Radial unit vector (x / r, y / r) times the force magnitude.
                (
                    lj_energy(r),
                    lj_force(r) * (x / r),
                    lj_force(r) * (y / r),
                )
            } else {
                // Exact centre: separation zero is a singularity; mark it so
                // the plotter can mask the core.
                (0.0, 0.0, 0.0)
            };

            println!("{x:.10},{y:.10},{v:.10},{fx:.10},{fy:.10}");
        }
    }
}
