//! CLI for the `md` crate: simulate a 2D Lennard-Jones fluid and save it,
//! check the physics of a saved trajectory, and render a motion video.
//!
//! Subcommands (with `--dir`, default `artifacts/`):
//! - (no subcommand, or `run`) run the simulation, writing `run.json` +
//!   `traj.jsonl`;
//! - `check <dir>` — print the three acceptance metrics and PASS/FAIL;
//! - `video <dir> --out fluid.mp4` — render atoms + g(r) to an mp4.

use std::env;
use std::path::PathBuf;

use md::fluid::{run_fluid, RunConfig};
use md::observables::{fit_speed_temperature, secular_drift};
use md::store::{self, FrameRecord};
use md::system::{total_energy, BoxConfig, System, Vec2};

fn fail(msg: &str) -> ! {
    eprintln!("{msg}");
    std::process::exit(2);
}

fn val<'a>(args: &'a [String], i: usize, flag: &str) -> &'a str {
    args.get(i + 1)
        .map(String::as_str)
        .unwrap_or_else(|| fail(&format!("{flag} needs a value")))
}

fn parse_usize(args: &[String], i: usize, flag: &str) -> usize {
    val(args, i, flag)
        .parse()
        .unwrap_or_else(|_| fail(&format!("{flag} must be an integer")))
}

fn parse_u64(args: &[String], i: usize, flag: &str) -> u64 {
    val(args, i, flag)
        .parse()
        .unwrap_or_else(|_| fail(&format!("{flag} must be an integer")))
}

fn parse_f64(args: &[String], i: usize, flag: &str) -> f64 {
    val(args, i, flag)
        .parse()
        .unwrap_or_else(|_| fail(&format!("{flag} must be a number")))
}

/// Parse `md` run flags into a `RunConfig` and an output directory.
fn parse_run(args: &[String]) -> (RunConfig, PathBuf) {
    let mut c = RunConfig::default();
    let mut dir = PathBuf::from("artifacts");
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--atoms" => c.atoms = parse_usize(args, i, "--atoms"),
            "--rho" => c.rho = parse_f64(args, i, "--rho"),
            "--temperature" => c.temperature = parse_f64(args, i, "--temperature"),
            "--seed" => c.seed = parse_u64(args, i, "--seed"),
            "--dt" => c.dt = parse_f64(args, i, "--dt"),
            "--cutoff" => c.cutoff = parse_f64(args, i, "--cutoff"),
            "--eq" => c.eq = parse_usize(args, i, "--eq"),
            "--steps" => c.steps = parse_usize(args, i, "--steps"),
            "--save-every" => c.save_every = parse_usize(args, i, "--save-every"),
            "--dir" => dir = PathBuf::from(val(args, i, "--dir")),
            other => fail(&format!("unknown flag {other}")),
        }
        i += 2;
    }
    (c, dir)
}

fn cmd_run(args: &[String]) {
    let (cfg, dir) = parse_run(args);
    let rec = run_fluid(&cfg);
    store::write_output(&dir, &rec).expect("write outputs");
    println!(
        "wrote {} frames (box {:.5}) to {}",
        rec.frames.len(),
        rec.box_len,
        dir.display()
    );
}

/// Recompute per-atom total energy from a saved frame's positions/velocities.
fn per_atom_energy(f: &FrameRecord, box_len: f64, cutoff: f64) -> f64 {
    let pos: Vec<Vec2> = f.x.iter().zip(&f.y).map(|(&a, &b)| [a, b]).collect();
    let vel: Vec<Vec2> = f.vx.iter().zip(&f.vy).map(|(&a, &b)| [a, b]).collect();
    let sys = System::with_box(pos, vel, BoxConfig { length: box_len, cutoff });
    total_energy(&sys) / f.x.len() as f64
}

fn cmd_check(dir: &PathBuf) {
    let (meta, frames) = store::read_run(dir).unwrap_or_else(|e| fail(&format!("read {:?}: {e}", dir)));
    if frames.is_empty() {
        fail("no frames found in trajectory");
    }
    let ts: Vec<f64> = frames.iter().map(|f| f.t).collect();
    let es: Vec<f64> = frames
        .iter()
        .map(|f| per_atom_energy(f, meta.box_len, meta.cutoff))
        .collect();
    let drift = secular_drift(&ts, &es);
    let (t_speed, chi2) = fit_speed_temperature(&frames, 40);

    println!("secular drift  = {drift:.6e}");
    println!("T_speed        = {t_speed:.4}");
    println!("chi2/dof       = {chi2:.4}");
    let pass = drift < 2e-3 && (t_speed - 0.5).abs() < 0.05 && chi2 < 2.0;
    println!("{}", if pass { "PASS" } else { "FAIL" });
    if !pass {
        std::process::exit(1);
    }
}

fn main() {
    let argv: Vec<String> = env::args().skip(1).collect();
    if argv.is_empty() {
        cmd_run(&argv); // bare `md` = default run into artifacts/
        return;
    }
    match argv[0].as_str() {
        "run" => cmd_run(&argv[1..]),
        "check" => {
            let dir = argv
                .get(1)
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("artifacts"));
            cmd_check(&dir);
        }
        "video" => {
            eprintln!("video subcommand not implemented yet");
            std::process::exit(1);
        }
        other if other.starts_with("--") => cmd_run(&argv),
        other => fail(&format!("unknown subcommand {other}")),
    }
}
