//! `md run`, `md check`, and `md video` for the Week 2 LJ fluid.

use std::env;
use std::path::{Path, PathBuf};

use md::ForceMethod;
use md::fluid::{RunConfig, run_fluid};
use md::observables::evaluate_saved_run;
use md::store;

fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("{message}");
    std::process::exit(2);
}

fn value<'a>(args: &'a [String], index: usize, flag: &str) -> &'a str {
    args.get(index + 1)
        .map(String::as_str)
        .unwrap_or_else(|| fail(format!("{flag} requires a value")))
}

fn parse_usize(args: &[String], index: usize, flag: &str) -> usize {
    value(args, index, flag)
        .parse()
        .unwrap_or_else(|_| fail(format!("{flag} must be a non-negative integer")))
}

fn parse_u64(args: &[String], index: usize, flag: &str) -> u64 {
    value(args, index, flag)
        .parse()
        .unwrap_or_else(|_| fail(format!("{flag} must be a non-negative integer")))
}

fn parse_f64(args: &[String], index: usize, flag: &str) -> f64 {
    value(args, index, flag)
        .parse()
        .unwrap_or_else(|_| fail(format!("{flag} must be a number")))
}

fn parse_run(args: &[String]) -> (RunConfig, PathBuf) {
    let mut config = RunConfig::default();
    let mut output = PathBuf::from("artifacts");
    let mut index = 0usize;
    while index < args.len() {
        let flag = args[index].as_str();
        match flag {
            "--n" | "--atoms" => config.n = parse_usize(args, index, flag),
            "--rho" => config.rho = parse_f64(args, index, flag),
            "--temperature" => config.temperature = parse_f64(args, index, flag),
            "--dt" => config.dt = parse_f64(args, index, flag),
            "--eq-steps" | "--eq" => config.eq_steps = parse_usize(args, index, flag),
            "--steps" => config.steps = parse_usize(args, index, flag),
            "--sample-every" | "--save-every" => {
                config.sample_every = parse_usize(args, index, flag)
            }
            "--seed" => config.seed = parse_u64(args, index, flag),
            "--cutoff" => config.cutoff = parse_f64(args, index, flag),
            "--force" => {
                config.force_method = ForceMethod::parse(value(args, index, flag))
                    .unwrap_or_else(|| fail("--force must be naive or cells"))
            }
            "--ramp-to" => config.ramp_to = Some(parse_f64(args, index, flag)),
            "--out" | "--dir" => output = PathBuf::from(value(args, index, flag)),
            "--help" | "-h" => print_help_and_exit(),
            _ => fail(format!("unknown run flag {flag}")),
        }
        index += 2;
    }
    if config.sample_every == 0 || config.steps % config.sample_every != 0 {
        fail("--sample-every must be positive and divide --steps exactly");
    }
    (config, output)
}

fn print_help_and_exit() -> ! {
    println!(
        "md run [--n 100] [--rho 0.8] [--temperature 0.5] [--dt 0.01] \\\n         [--eq-steps 2000] [--steps 10000] [--sample-every 50] [--seed 2026] \\\n         [--force cells|naive] [--ramp-to T] [--out artifacts]\n\
         md check [artifacts]\n\
         md video [artifacts] --out fluid.mp4"
    );
    std::process::exit(0)
}

fn cmd_run(args: &[String]) {
    let (config, output) = parse_run(args);
    let record = run_fluid(&config);
    store::write_output(&output, &record).unwrap_or_else(|error| fail(error));
    println!(
        "wrote {} frames for {} atoms with {} forces to {}",
        record.frames.len(),
        config.n,
        config.force_method.as_str(),
        output.display()
    );
}

fn cmd_check(dir: &Path) {
    let (meta, frames) = store::read_run(dir)
        .unwrap_or_else(|error| fail(format!("cannot read {}: {error}", dir.display())));
    let report = evaluate_saved_run(&meta, &frames).unwrap_or_else(|error| fail(error));
    println!(
        "secular drift  = {:.6e}   limit < 2e-3",
        report.secular_drift
    );
    println!(
        "T_speed        = {:.6}   |T_speed - {:.3}| limit < 0.05",
        report.t_speed, meta.temperature
    );
    println!("chi2/22        = {:.6}   limit < 2", report.chi2_per_22);
    println!(
        "stored E error = {:.3e}   limit < 1e-8",
        report.stored_energy_max_error
    );
    if report.passes(meta.temperature) {
        println!("PASS");
    } else {
        println!("FAIL");
        std::process::exit(1);
    }
}

fn cmd_video(args: &[String]) {
    let dir = args
        .first()
        .filter(|value| !value.starts_with("--"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("artifacts"));
    let output = args
        .iter()
        .position(|value| value == "--out")
        .and_then(|index| args.get(index + 1))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("fluid.mp4"));
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/render_video.py");
    let status = std::process::Command::new("python3")
        .arg(script)
        .arg(dir)
        .arg(output)
        .status()
        .unwrap_or_else(|error| fail(format!("cannot start video renderer: {error}")));
    std::process::exit(status.code().unwrap_or(1));
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        None => cmd_run(&[]),
        Some("run") => cmd_run(&args[1..]),
        Some("check") => cmd_check(
            args.get(1)
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("artifacts"))
                .as_path(),
        ),
        Some("video") => cmd_video(&args[1..]),
        Some("--help" | "-h") => print_help_and_exit(),
        Some(flag) if flag.starts_with("--") => cmd_run(&args),
        Some(other) => fail(format!("unknown subcommand {other}")),
    }
}
