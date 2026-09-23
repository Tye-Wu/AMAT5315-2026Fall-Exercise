use std::env;
use week4::fluid::{FluidOptions, run_from_stdin};

fn value(args: &mut impl Iterator<Item = String>, name: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("missing value after {name}"))
}

fn run() -> Result<i32, String> {
    let mut args = env::args().skip(1);
    let mut method = None;
    let mut viscosity = None;
    let mut dt = None;
    let mut t_end = None;
    let mut every = None;
    let mut out = None;
    while let Some(flag) = args.next() {
        let raw = value(&mut args, &flag)?;
        match flag.as_str() {
            "--method" => method = Some(raw),
            "--nu" => viscosity = Some(raw.parse::<f64>().map_err(|e| e.to_string())?),
            "--dt" => dt = Some(raw.parse::<f64>().map_err(|e| e.to_string())?),
            "--t-end" => t_end = Some(raw.parse::<f64>().map_err(|e| e.to_string())?),
            "--every" => every = Some(raw.parse::<f64>().map_err(|e| e.to_string())?),
            "--out" => out = Some(raw),
            _ => return Err(format!("unknown option {flag}")),
        }
    }
    let options = FluidOptions {
        method: method.ok_or_else(|| "--method is required".to_owned())?,
        viscosity: viscosity.ok_or_else(|| "--nu is required".to_owned())?,
        dt: dt.ok_or_else(|| "--dt is required".to_owned())?,
        t_end: t_end.ok_or_else(|| "--t-end is required".to_owned())?,
        every: every.ok_or_else(|| "--every is required".to_owned())?,
        out: out.ok_or_else(|| "--out is required".to_owned())?,
    };
    run_from_stdin(&options)
}

fn main() {
    match run() {
        Ok(code) => std::process::exit(code),
        Err(error) => {
            eprintln!("fluid: {error}");
            std::process::exit(2);
        }
    }
}
