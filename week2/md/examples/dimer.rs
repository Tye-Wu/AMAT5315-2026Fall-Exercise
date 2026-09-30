//! Run the two-atom LJ dimer experiment and print the energy error over time.
//!
//! Writes CSV to stdout with one row per step: `run,t,dE`, where
//! `dE = (E(t) - E(0)) / |E(0)|` is the relative total-energy error.
//! The three cases share the same initial state and time step:
//!
//!   run        steps   integrator
//!   euler500   500     Euler (explicit forward Euler)
//!   verlet500  500     VelocityVerlet
//!   verlet5000 5000    VelocityVerlet (extended run)
//!
//! Initial state: two m = 1 atoms at rest, separated by r = 1.2, so
//! E(0) = U(1.2). dt = 0.01.
//!
//! Run directly with:
//!
//!     cargo run --quiet --example dimer --manifest-path week2/md/Cargo.toml
//!
//! `week2/plot_dimer.py` runs this example and draws the two-panel figure
//! `week2/dimer.png`.

use md::{Euler, System, VelocityVerlet, run_experiment};

fn main() {
    let dt = 0.01;

    // Two atoms at rest, 1.2 sigma apart (outside r0), straddling the origin.
    let initial = || System::new(vec![[-0.6, 0.0], [0.6, 0.0]], vec![[0.0, 0.0], [0.0, 0.0]]);

    let euler500 = run_experiment(&Euler, &mut initial(), dt, 500);
    let verlet500 = run_experiment(&VelocityVerlet, &mut initial(), dt, 500);
    let verlet5000 = run_experiment(&VelocityVerlet, &mut initial(), dt, 5000);

    println!("run,t,dE");
    emit("euler500", &euler500);
    emit("verlet500", &verlet500);
    emit("verlet5000", &verlet5000);
}

fn emit(run: &str, series: &[(f64, f64)]) {
    for (t, de) in series {
        println!("{run},{t:.6},{de:.9e}");
    }
}
