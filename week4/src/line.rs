use crate::fft::derivative_1d;

#[derive(Clone, Copy, Debug)]
pub enum Derivative {
    Fourier,
    Centered,
}

pub fn fourier_grid(n: usize) -> (Vec<f64>, f64) {
    let dx = std::f64::consts::TAU / n as f64;
    ((0..n).map(|j| j as f64 * dx).collect(), dx)
}

pub fn derivative(values: &[f64], dx: f64, order: usize, method: Derivative) -> Vec<f64> {
    match method {
        Derivative::Fourier => derivative_1d(values, order),
        Derivative::Centered => {
            let n = values.len();
            (0..n)
                .map(|j| {
                    let left = values[(j + n - 1) % n];
                    let center = values[j];
                    let right = values[(j + 1) % n];
                    if order == 1 {
                        (right - left) / (2.0 * dx)
                    } else {
                        (right - 2.0 * center + left) / (dx * dx)
                    }
                })
                .collect()
        }
    }
}

pub fn advection_diffusion_rate(
    values: &[f64],
    speed: f64,
    viscosity: f64,
    method: Derivative,
) -> Vec<f64> {
    let (_, dx) = fourier_grid(values.len());
    let first = derivative(values, dx, 1, method);
    let second = derivative(values, dx, 2, method);
    first
        .iter()
        .zip(second)
        .map(|(du, ddu)| -speed * du + viscosity * ddu)
        .collect()
}

pub fn gaussian_periodic(n: usize, sigma: f64, center: f64) -> Vec<f64> {
    let (x, _) = fourier_grid(n);
    x.iter()
        .map(|&point| {
            (-4..=4)
                .map(|image| {
                    let offset = point - center - image as f64 * std::f64::consts::TAU;
                    (-offset * offset / (2.0 * sigma * sigma)).exp()
                })
                .sum()
        })
        .collect()
}

pub fn gaussian_exact(
    n: usize,
    sigma: f64,
    center: f64,
    speed: f64,
    viscosity: f64,
    time: f64,
) -> Vec<f64> {
    let width = (sigma * sigma + 2.0 * viscosity * time).sqrt();
    let translated_center = center + speed * time;
    let amplitude = sigma / width;
    let (x, _) = fourier_grid(n);
    x.iter()
        .map(|&point| {
            amplitude
                * (-4..=4)
                    .map(|image| {
                        let offset =
                            point - translated_center - image as f64 * std::f64::consts::TAU;
                        (-offset * offset / (2.0 * width * width)).exp()
                    })
                    .sum::<f64>()
        })
        .collect()
}

pub fn single_wave(x: &[f64], wave_number: usize, phase: f64) -> Vec<f64> {
    x.iter()
        .map(|point| (wave_number as f64 * point + phase).cos())
        .collect()
}

pub fn single_wave_exact(
    x: &[f64],
    wave_number: usize,
    speed: f64,
    viscosity: f64,
    time: f64,
    phase: f64,
) -> Vec<f64> {
    let decay = (-viscosity * (wave_number * wave_number) as f64 * time).exp();
    x.iter()
        .map(|point| decay * (wave_number as f64 * (point - speed * time) + phase).cos())
        .collect()
}
