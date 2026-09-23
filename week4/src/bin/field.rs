use std::env;
use std::io::{self, Write};
use week4::field::{random_vorticity_field, taylor_green};

fn value(args: &mut impl Iterator<Item = String>, name: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("missing value after {name}"))
}

fn usage() -> &'static str {
    "usage: field taylor-green --n N [--t T --nu NU]\n       field random --n N --seed SEED --k-min K --k-max K"
}

fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let case = args.next().ok_or_else(|| usage().to_owned())?;
    let mut n = None;
    let mut seed = None;
    let mut k_min = None;
    let mut k_max = None;
    let mut time = 0.0;
    let mut viscosity = None;
    while let Some(flag) = args.next() {
        let raw = value(&mut args, &flag)?;
        match flag.as_str() {
            "--n" => n = Some(raw.parse::<usize>().map_err(|e| e.to_string())?),
            "--seed" => seed = Some(raw.parse::<u64>().map_err(|e| e.to_string())?),
            "--k-min" => k_min = Some(raw.parse::<usize>().map_err(|e| e.to_string())?),
            "--k-max" => k_max = Some(raw.parse::<usize>().map_err(|e| e.to_string())?),
            "--t" => time = raw.parse::<f64>().map_err(|e| e.to_string())?,
            "--nu" => viscosity = Some(raw.parse::<f64>().map_err(|e| e.to_string())?),
            _ => return Err(format!("unknown option {flag}\n{}", usage())),
        }
    }
    let n = n.ok_or_else(|| format!("--n is required\n{}", usage()))?;
    if !n.is_power_of_two() || n < 8 {
        return Err("--n must be a power of two and at least 8".to_owned());
    }
    let field = match case.as_str() {
        "taylor-green" => {
            if time > 0.0 && viscosity.is_none() {
                return Err("--nu is required when --t is positive".to_owned());
            }
            if time < 0.0 {
                return Err("--t must be non-negative".to_owned());
            }
            taylor_green(n, time, viscosity)
        }
        "random" => random_vorticity_field(
            n,
            seed.ok_or_else(|| "--seed is required for random".to_owned())?,
            k_min.ok_or_else(|| "--k-min is required for random".to_owned())?,
            k_max.ok_or_else(|| "--k-max is required for random".to_owned())?,
        ),
        _ => return Err(format!("unknown case {case}\n{}", usage())),
    };
    let stdout = io::stdout();
    let mut handle = stdout.lock();
    serde_json::to_writer(&mut handle, &field).map_err(|e| e.to_string())?;
    handle.write_all(b"\n").map_err(|e| e.to_string())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("field: {error}");
        std::process::exit(2);
    }
}
