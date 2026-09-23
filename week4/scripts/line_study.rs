use serde::Serialize;
use std::f64::consts::TAU;
use week4::integrator::{EqualWeightRk4, Euler, Integrator, Midpoint, Rk4, integrate};
use week4::line::{
    Derivative, advection_diffusion_rate, fourier_grid, gaussian_exact, gaussian_periodic,
};

#[derive(Serialize)]
struct GrowthMap {
    real: Vec<f64>,
    imag: Vec<f64>,
    measured_rk4: Vec<Vec<f64>>,
    mode_points: Vec<ModePoints>,
}

#[derive(Serialize)]
struct ModePoints {
    dt: f64,
    points: Vec<[f64; 2]>,
}

#[derive(Serialize)]
struct PulsePanel {
    dt: f64,
    times: Vec<f64>,
    values: Vec<Vec<f64>>,
}

#[derive(Serialize)]
struct ErrorSeries {
    method: String,
    steps: Vec<f64>,
    errors: Vec<f64>,
}

fn measured_growth(z_re: f64, z_im: f64) -> f64 {
    let rate = |_: f64, y: &[f64]| vec![z_re * y[0] - z_im * y[1], z_im * y[0] + z_re * y[1]];
    let result = Rk4.step(&[1.0, 0.0], 0.0, 1.0, &rate);
    result[0].hypot(result[1])
}

fn stability_study() -> GrowthMap {
    let real: Vec<f64> = (0..201).map(|i| -4.0 + i as f64 * 5.0 / 200.0).collect();
    let imag: Vec<f64> = (0..321).map(|i| -4.0 + i as f64 * 8.0 / 320.0).collect();
    let measured_rk4 = imag
        .iter()
        .map(|&im| real.iter().map(|&re| measured_growth(re, im)).collect())
        .collect();
    let n = 64isize;
    let modes = (-n / 2..n / 2)
        .map(|k| {
            let real_part = -0.05 * (k * k) as f64;
            let imaginary_part = if k == -n / 2 { 0.0 } else { -(k as f64) };
            [real_part, imaginary_part]
        })
        .collect::<Vec<_>>();
    let mode_points = [0.045, 0.056]
        .into_iter()
        .map(|dt| ModePoints {
            dt,
            points: modes.iter().map(|[re, im]| [re * dt, im * dt]).collect(),
        })
        .collect();
    GrowthMap {
        real,
        imag,
        measured_rk4,
        mode_points,
    }
}

fn pulse_frames(dt: f64) -> PulsePanel {
    let n = 64;
    let (x, _) = fourier_grid(n);
    let mut state = gaussian_periodic(n, 0.35, std::f64::consts::PI / 2.0);
    let integrator = Rk4;
    let rate =
        |_: f64, values: &[f64]| advection_diffusion_rate(values, 1.0, 0.05, Derivative::Fourier);
    let steps = (6.0 / dt).floor() as usize;
    let mut times = vec![0.0];
    let mut values = vec![state.clone()];
    for k in 1..=steps {
        state = integrator.step(&state, (k - 1) as f64 * dt, dt, &rate);
        if k % 3 == 0 || k == steps {
            times.push(k as f64 * dt);
            values.push(state.clone());
        }
    }
    let _ = x;
    let final_time = steps as f64 * dt;
    if final_time < 6.0 {
        let remaining = 6.0 - final_time;
        state = integrator.step(&state, final_time, remaining, &rate);
        times.push(6.0);
        values.push(state);
    }
    PulsePanel { dt, times, values }
}

fn error_series() -> Vec<ErrorSeries> {
    let n = 64;
    let (x, _) = fourier_grid(n);
    let steps = vec![0.02, 0.01, 0.005, 0.0025];
    let initial = gaussian_periodic(n, 0.35, std::f64::consts::PI / 2.0);
    let exact = gaussian_exact(n, 0.35, std::f64::consts::PI / 2.0, 1.0, 0.05, 1.0);
    let rate =
        |_: f64, values: &[f64]| advection_diffusion_rate(values, 1.0, 0.05, Derivative::Fourier);
    let methods: Vec<(&str, &dyn Integrator)> = vec![
        ("Euler", &Euler),
        ("midpoint", &Midpoint),
        ("RK4", &Rk4),
        ("equal-weight RK4", &EqualWeightRk4),
    ];
    let result = methods
        .into_iter()
        .map(|(name, method)| {
            let errors = steps
                .iter()
                .map(|&dt| {
                    let values = integrate(method, initial.clone(), 0.0, 1.0, dt, &rate);
                    values
                        .iter()
                        .zip(&exact)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0, f64::max)
                })
                .collect();
            ErrorSeries {
                method: name.to_owned(),
                steps: steps.clone(),
                errors,
            }
        })
        .collect();
    let _ = x;
    result
}

fn first_accuracy_profiles() -> (Vec<f64>, Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
    let n = 64;
    let (x, _) = fourier_grid(n);
    let center = std::f64::consts::PI / 2.0;
    let initial = gaussian_periodic(n, 0.25, center);
    let exact = gaussian_exact(n, 0.25, center, 1.0, 0.002, TAU);
    let cases = [
        (
            "RK4 Fourier",
            0.02,
            Derivative::Fourier,
            &Rk4 as &dyn Integrator,
        ),
        (
            "RK4 centered",
            0.02,
            Derivative::Centered,
            &Rk4 as &dyn Integrator,
        ),
        (
            "Euler Fourier",
            0.005,
            Derivative::Fourier,
            &Euler as &dyn Integrator,
        ),
    ];
    let profiles = cases
        .iter()
        .map(|(_, dt, derivative, integrator)| {
            let rate =
                |_: f64, values: &[f64]| advection_diffusion_rate(values, 1.0, 0.002, *derivative);
            integrate(*integrator, initial.clone(), 0.0, TAU, *dt, &rate)
        })
        .collect();
    (x, profiles, exact, initial)
}

#[derive(Serialize)]
struct LineData {
    stability: GrowthMap,
    stable_pulse: PulsePanel,
    unstable_pulse: PulsePanel,
    accuracy_x: Vec<f64>,
    accuracy_profiles: Vec<Vec<f64>>,
    accuracy_exact: Vec<f64>,
    accuracy_initial: Vec<f64>,
    error_series: Vec<ErrorSeries>,
}

fn main() {
    let (accuracy_x, accuracy_profiles, accuracy_exact, accuracy_initial) =
        first_accuracy_profiles();
    let result = LineData {
        stability: stability_study(),
        stable_pulse: pulse_frames(0.045),
        unstable_pulse: pulse_frames(0.056),
        accuracy_x,
        accuracy_profiles,
        accuracy_exact,
        accuracy_initial,
        error_series: error_series(),
    };
    serde_json::to_writer(std::io::stdout(), &result).expect("write JSON");
    println!();
}
