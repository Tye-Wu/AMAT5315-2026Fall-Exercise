use crate::complex::Complex;
use crate::fft::{fft2_real, ifft2_real, modes};
use crate::field::{VelocityField, vorticity_from_velocity};
use crate::integrator::{Euler, Integrator, Midpoint, Rk4};
use serde::Serialize;
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Write};
use std::path::Path;

pub fn apply_two_thirds_mask(spectrum: &mut [Complex], n: usize) {
    let cutoff = n / 3;
    let k = modes(n);
    for y in 0..n {
        for x in 0..n {
            if k[x].abs() as usize > cutoff || k[y].abs() as usize > cutoff {
                spectrum[y * n + x] = Complex::default();
            }
        }
    }
}

pub fn velocity_from_vorticity_spectrum(omega_hat: &[Complex], n: usize) -> (Vec<f64>, Vec<f64>) {
    let k = modes(n);
    let mut u_hat = vec![Complex::default(); n * n];
    let mut v_hat = vec![Complex::default(); n * n];
    for y in 0..n {
        for x in 0..n {
            let kx = k[x] as f64;
            let ky = k[y] as f64;
            let k_squared = kx * kx + ky * ky;
            if k_squared != 0.0 {
                let psi_hat = omega_hat[y * n + x].scale(1.0 / k_squared);
                u_hat[y * n + x] = psi_hat.times_i(ky);
                v_hat[y * n + x] = psi_hat.times_i(-kx);
            }
        }
    }
    (ifft2_real(&u_hat, n), ifft2_real(&v_hat, n))
}

pub fn velocity_from_vorticity(n: usize, omega: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut omega_hat = fft2_real(omega, n);
    apply_two_thirds_mask(&mut omega_hat, n);
    velocity_from_vorticity_spectrum(&omega_hat, n)
}

pub fn vorticity_rate(n: usize, omega: &[f64], viscosity: f64) -> Vec<f64> {
    let mut omega_hat = fft2_real(omega, n);
    apply_two_thirds_mask(&mut omega_hat, n);
    let (u, v) = velocity_from_vorticity_spectrum(&omega_hat, n);
    let k = modes(n);
    let mut dx_hat = omega_hat.clone();
    let mut dy_hat = omega_hat.clone();
    for y in 0..n {
        for x in 0..n {
            let index = y * n + x;
            let kx = if n % 2 == 0 && x == n / 2 {
                0.0
            } else {
                k[x] as f64
            };
            let ky = if n % 2 == 0 && y == n / 2 {
                0.0
            } else {
                k[y] as f64
            };
            dx_hat[index] = dx_hat[index].times_i(kx);
            dy_hat[index] = dy_hat[index].times_i(ky);
        }
    }
    let dx_omega = ifft2_real(&dx_hat, n);
    let dy_omega = ifft2_real(&dy_hat, n);
    let advection: Vec<f64> = u
        .iter()
        .zip(&v)
        .zip(dx_omega.iter().zip(&dy_omega))
        .map(|((u, v), (dx, dy))| u * dx + v * dy)
        .collect();
    let mut advection_hat = fft2_real(&advection, n);
    apply_two_thirds_mask(&mut advection_hat, n);
    for y in 0..n {
        for x in 0..n {
            let kx = k[x] as f64;
            let ky = k[y] as f64;
            let index = y * n + x;
            let laplacian = omega_hat[index].scale(-(kx * kx + ky * ky));
            advection_hat[index] = advection_hat[index].scale(-1.0) + laplacian.scale(viscosity);
        }
    }
    ifft2_real(&advection_hat, n)
}

pub fn energy_enstrophy(n: usize, omega: &[f64]) -> (f64, f64) {
    let mut omega_hat = fft2_real(omega, n);
    apply_two_thirds_mask(&mut omega_hat, n);
    let k = modes(n);
    let mut spectral_energy = 0.0;
    for y in 0..n {
        for x in 0..n {
            let kx = k[x] as f64;
            let ky = k[y] as f64;
            let k_squared = kx * kx + ky * ky;
            if k_squared > 0.0 {
                spectral_energy += omega_hat[y * n + x].norm_sqr() / k_squared;
            }
        }
    }
    let cells = (n * n) as f64;
    let energy = 0.5 * spectral_energy / (cells * cells);
    let enstrophy = 0.5 * omega.iter().map(|value| value * value).sum::<f64>() / cells;
    (energy, enstrophy)
}

#[derive(Serialize)]
struct RunMetadata<'a> {
    case: &'a str,
    n: usize,
    seed: Option<u64>,
    k_band: Option<[usize; 2]>,
    method: &'a str,
    nu: f64,
    dt: f64,
    t_end: f64,
    snapshot_every: f64,
}

#[derive(Serialize)]
struct Frame {
    t: f64,
    step: usize,
    u: Vec<f64>,
    v: Vec<f64>,
    omega: Vec<f64>,
}

pub struct FluidOptions {
    pub method: String,
    pub viscosity: f64,
    pub dt: f64,
    pub t_end: f64,
    pub every: f64,
    pub out: String,
}

pub fn run_from_stdin(options: &FluidOptions) -> Result<i32, String> {
    let stdin = io::stdin();
    let field: VelocityField = serde_json::from_reader(BufReader::new(stdin.lock()))
        .map_err(|error| format!("cannot read field JSON from stdin: {error}"))?;
    validate(&field, options)?;

    let out_path = Path::new(&options.out);
    fs::create_dir_all(out_path)
        .map_err(|error| format!("cannot create {}: {error}", out_path.display()))?;
    let run = RunMetadata {
        case: &field.case,
        n: field.n,
        seed: field.seed,
        k_band: field.k_band,
        method: &options.method,
        nu: options.viscosity,
        dt: options.dt,
        t_end: options.t_end,
        snapshot_every: options.every,
    };
    let run_file = File::create(out_path.join("run.json"))
        .map_err(|error| format!("cannot write run.json: {error}"))?;
    serde_json::to_writer_pretty(BufWriter::new(run_file), &run)
        .map_err(|error| format!("cannot serialize run.json: {error}"))?;

    let fields_file = File::create(out_path.join("fields.jsonl"))
        .map_err(|error| format!("cannot write fields.jsonl: {error}"))?;
    let mut fields = BufWriter::new(fields_file);
    let mut omega = vorticity_from_velocity(field.n, &field.u, &field.v);
    let mut omega_hat = fft2_real(&omega, field.n);
    apply_two_thirds_mask(&mut omega_hat, field.n);
    omega = ifft2_real(&omega_hat, field.n);

    let integrator: Box<dyn Integrator> = match options.method.as_str() {
        "euler" => Box::new(Euler),
        "rk2" => Box::new(Midpoint),
        "rk4" => Box::new(Rk4),
        _ => return Err("--method must be euler, rk2, or rk4".to_owned()),
    };
    let interval = (options.every / options.dt).round().max(1.0) as usize;
    let step_ratio = options.t_end / options.dt;
    let nearest_steps = step_ratio.round();
    let total_steps = if (step_ratio - nearest_steps).abs() < 1e-10 {
        nearest_steps as usize
    } else {
        step_ratio.ceil() as usize
    };
    let mut energy_history = BufWriter::new(io::stdout().lock());
    writeln!(energy_history, "t\tenergy\tenstrophy")
        .map_err(|error| format!("cannot write stdout: {error}"))?;
    let mut time = 0.0;

    for step_index in 0..=total_steps {
        let (energy, enstrophy) = energy_enstrophy(field.n, &omega);
        if !energy.is_finite() || !enstrophy.is_finite() || omega.iter().any(|x| !x.is_finite()) {
            writeln!(energy_history, "{time:.6}\tnon-finite\tnon-finite")
                .map_err(|error| format!("cannot write non-finite row: {error}"))?;
            energy_history.flush().ok();
            return Ok(1);
        }
        if step_index % interval == 0 {
            writeln!(energy_history, "{time:.6}\t{energy:.6}\t{enstrophy:.6}")
                .map_err(|error| format!("cannot write stdout: {error}"))?;
            write_frame(&mut fields, field.n, time, step_index, &omega)
                .map_err(|error| format!("cannot write snapshot: {error}"))?;
        }
        if step_index == total_steps {
            break;
        }
        let rate = |_: f64, state: &[f64]| vorticity_rate(field.n, state, options.viscosity);
        let step_size = options.dt.min(options.t_end - time);
        omega = integrator.step(&omega, time, step_size, &rate);
        time += step_size;
    }
    fields
        .flush()
        .map_err(|error| format!("cannot flush fields: {error}"))?;
    energy_history
        .flush()
        .map_err(|error| format!("cannot flush stdout: {error}"))?;
    Ok(0)
}

fn write_frame(
    output: &mut BufWriter<File>,
    n: usize,
    time: f64,
    step: usize,
    omega: &[f64],
) -> io::Result<()> {
    let (u, v) = velocity_from_vorticity(n, omega);
    let rounded = |values: &[f64]| {
        values
            .iter()
            .map(|value| (value * 1e6).round() / 1e6)
            .collect()
    };
    let frame = Frame {
        t: time,
        step,
        u: rounded(&u),
        v: rounded(&v),
        omega: rounded(omega),
    };
    serde_json::to_writer(&mut *output, &frame)?;
    output.write_all(b"\n")
}

fn validate(field: &VelocityField, options: &FluidOptions) -> Result<(), String> {
    if !field.n.is_power_of_two() || field.n < 8 {
        return Err("field n must be a power of two and at least 8".to_owned());
    }
    if field.u.len() != field.n * field.n || field.v.len() != field.n * field.n {
        return Err("input u and v arrays must each contain n*n values".to_owned());
    }
    if options.dt <= 0.0 || options.every <= 0.0 || options.t_end < 0.0 {
        return Err("dt and every must be positive; t-end must be non-negative".to_owned());
    }
    if !matches!(options.method.as_str(), "euler" | "rk2" | "rk4") {
        return Err("--method must be euler, rk2, or rk4".to_owned());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::taylor_green;

    #[test]
    fn taylor_green_has_zero_nonlinear_advection() {
        let n = 32;
        let field = taylor_green(n, 0.0, None);
        let omega = vorticity_from_velocity(n, &field.u, &field.v);
        let rate = vorticity_rate(n, &omega, 0.0);
        let max = rate.iter().map(|value| value.abs()).fold(0.0, f64::max);
        assert!(max < 2e-12, "Taylor-Green advection residual was {max:e}");
    }

    #[test]
    fn two_thirds_mask_removes_corner_modes() {
        let n = 64;
        let mut spectrum = vec![Complex::new(1.0, 0.0); n * n];
        apply_two_thirds_mask(&mut spectrum, n);
        let k = modes(n);
        for y in 0..n {
            for x in 0..n {
                if k[x].abs() as usize > n / 3 || k[y].abs() as usize > n / 3 {
                    assert_eq!(spectrum[y * n + x], Complex::default());
                }
            }
        }
    }
}
