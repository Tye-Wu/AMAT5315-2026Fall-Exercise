use crate::complex::Complex;
use crate::fft::{fft2_real, ifft2_real, modes};
use crate::fluid::velocity_from_vorticity_spectrum;
use serde::{Deserialize, Serialize};
use std::f64::consts::TAU;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VelocityField {
    pub case: String,
    pub n: usize,
    pub seed: Option<u64>,
    pub k_band: Option<[usize; 2]>,
    pub u: Vec<f64>,
    pub v: Vec<f64>,
}

pub fn taylor_green(n: usize, time: f64, viscosity: Option<f64>) -> VelocityField {
    let decay = if time == 0.0 {
        1.0
    } else {
        (-2.0 * viscosity.expect("viscosity is required when t > 0") * time).exp()
    };
    let mut u = Vec::with_capacity(n * n);
    let mut v = Vec::with_capacity(n * n);
    for j in 0..n {
        let y = TAU * j as f64 / n as f64;
        for i in 0..n {
            let x = TAU * i as f64 / n as f64;
            u.push(x.cos() * y.sin() * decay);
            v.push(-x.sin() * y.cos() * decay);
        }
    }
    VelocityField {
        case: "taylor-green".to_owned(),
        n,
        seed: None,
        k_band: None,
        u,
        v,
    }
}

pub fn random_vorticity_field(n: usize, seed: u64, k_min: usize, k_max: usize) -> VelocityField {
    assert!(n.is_power_of_two() && n >= 16);
    assert!(k_min <= k_max && k_min > 0);
    assert!(
        k_max <= n / 3,
        "the band must fit under the two-thirds cutoff"
    );

    let wavenumbers = modes(n);
    let mut omega_hat = vec![Complex::default(); n * n];
    for ky_index in 0..n {
        for kx_index in 0..n {
            let kx = wavenumbers[kx_index];
            let ky = wavenumbers[ky_index];
            let radius_squared = (kx * kx + ky * ky) as usize;
            if radius_squared < k_min * k_min || radius_squared > k_max * k_max {
                continue;
            }
            let partner_x = (-kx).rem_euclid(n as isize) as usize;
            let partner_y = (-ky).rem_euclid(n as isize) as usize;
            let index = ky_index * n + kx_index;
            let partner = partner_y * n + partner_x;
            if index > partner {
                continue;
            }

            let phase_hash = phase_key(seed, kx, ky);
            let phase = TAU * (phase_hash as f64 / u64::MAX as f64);
            let value = if index == partner {
                Complex::new(
                    if phase < std::f64::consts::PI {
                        1.0
                    } else {
                        -1.0
                    },
                    0.0,
                )
            } else {
                Complex::new(phase.cos(), phase.sin())
            };
            omega_hat[index] = value;
            omega_hat[partner] = value.conjugate();
        }
    }

    let mut energy = 0.0;
    for ky_index in 0..n {
        for kx_index in 0..n {
            let kx = wavenumbers[kx_index] as f64;
            let ky = wavenumbers[ky_index] as f64;
            let k_squared = kx * kx + ky * ky;
            if k_squared != 0.0 {
                energy += omega_hat[ky_index * n + kx_index].norm_sqr() / k_squared;
            }
        }
    }
    energy *= 0.5 / (n * n * n * n) as f64;
    let scale = (0.5 / energy).sqrt();
    for value in &mut omega_hat {
        *value = value.scale(scale);
    }

    let (u, v) = velocity_from_vorticity_spectrum(&omega_hat, n);
    VelocityField {
        case: "random".to_owned(),
        n,
        seed: Some(seed),
        k_band: Some([k_min, k_max]),
        u,
        v,
    }
}

fn phase_key(seed: u64, kx: isize, ky: isize) -> u64 {
    let x = kx as i64 as u64;
    let y = ky as i64 as u64;
    splitmix64(seed ^ x.rotate_left(21) ^ y.rotate_left(43))
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = value;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

pub fn vorticity_from_velocity(n: usize, u: &[f64], v: &[f64]) -> Vec<f64> {
    let dx_v = crate::fft::derivative_2d(v, n, 1, 0);
    let dy_u = crate::fft::derivative_2d(u, n, 0, 1);
    dx_v.iter().zip(dy_u).map(|(dx, dy)| dx - dy).collect()
}

pub fn vorticity_spectrum_from_field(field: &VelocityField) -> Vec<Complex> {
    fft2_real(
        &vorticity_from_velocity(field.n, &field.u, &field.v),
        field.n,
    )
}

pub fn vorticity_real_from_spectrum(spectrum: &[Complex], n: usize) -> Vec<f64> {
    ifft2_real(spectrum, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_field_has_requested_energy_and_reproducible_seed() {
        let first = random_vorticity_field(128, 2026, 2, 6);
        let second = random_vorticity_field(128, 2026, 2, 6);
        assert_eq!(first.u, second.u);
        assert_eq!(first.v, second.v);
        let energy = first
            .u
            .iter()
            .zip(&first.v)
            .map(|(u, v)| 0.5 * (u * u + v * v))
            .sum::<f64>()
            / (first.n * first.n) as f64;
        assert!((energy - 0.5).abs() < 2e-14);
    }

    #[test]
    fn taylor_green_vorticity_has_the_declared_sign() {
        let n = 32;
        let field = taylor_green(n, 0.0, None);
        let omega = vorticity_from_velocity(n, &field.u, &field.v);
        let center = (n / 2) * n + n / 2;
        assert!((omega[center] + 2.0).abs() < 1e-12);
    }
}
