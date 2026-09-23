use crate::complex::Complex;
use std::f64::consts::TAU;

/// In-place radix-2 FFT. Forward transforms are unnormalised; inverse
/// transforms divide by the vector length.
pub fn fft1d(values: &mut [Complex], inverse: bool) {
    let n = values.len();
    assert!(n.is_power_of_two(), "FFT length must be a power of two");

    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            values.swap(i, j);
        }
    }

    let mut width = 2;
    while width <= n {
        let angle = (if inverse { TAU } else { -TAU }) / width as f64;
        let root = Complex::new(angle.cos(), angle.sin());
        for block in (0..n).step_by(width) {
            let mut twiddle = Complex::new(1.0, 0.0);
            for offset in 0..(width / 2) {
                let even = values[block + offset];
                let odd = values[block + offset + width / 2] * twiddle;
                values[block + offset] = even + odd;
                values[block + offset + width / 2] = even - odd;
                twiddle = twiddle * root;
            }
        }
        width <<= 1;
    }

    if inverse {
        let scale = 1.0 / n as f64;
        for value in values {
            *value = value.scale(scale);
        }
    }
}

pub fn fft2_real(values: &[f64], n: usize) -> Vec<Complex> {
    assert_eq!(values.len(), n * n);
    let mut spectrum: Vec<Complex> = values
        .iter()
        .map(|&value| Complex::new(value, 0.0))
        .collect();
    transform2d(&mut spectrum, n, false);
    spectrum
}

pub fn ifft2_real(spectrum: &[Complex], n: usize) -> Vec<f64> {
    assert_eq!(spectrum.len(), n * n);
    let mut data = spectrum.to_vec();
    transform2d(&mut data, n, true);
    data.into_iter().map(|value| value.re).collect()
}

fn transform2d(data: &mut [Complex], n: usize, inverse: bool) {
    assert!(
        n.is_power_of_two(),
        "FFT side length must be a power of two"
    );
    let mut line = vec![Complex::default(); n];

    for row in 0..n {
        let start = row * n;
        fft1d(&mut data[start..start + n], inverse);
    }
    for column in 0..n {
        for row in 0..n {
            line[row] = data[row * n + column];
        }
        fft1d(&mut line, inverse);
        for row in 0..n {
            data[row * n + column] = line[row];
        }
    }
}

/// FFT-ordered integer wavenumbers, with the Nyquist mode represented as -n/2.
pub fn modes(n: usize) -> Vec<isize> {
    (0..n)
        .map(|index| {
            if index < n.div_ceil(2) {
                index as isize
            } else {
                index as isize - n as isize
            }
        })
        .collect()
}

pub fn derivative_1d(values: &[f64], order: usize) -> Vec<f64> {
    assert!(order == 1 || order == 2);
    let n = values.len();
    let wave_numbers = modes(n);
    let mut spectrum: Vec<Complex> = values
        .iter()
        .map(|&value| Complex::new(value, 0.0))
        .collect();
    fft1d(&mut spectrum, false);

    for (index, value) in spectrum.iter_mut().enumerate() {
        let k = wave_numbers[index] as f64;
        if order == 1 {
            let multiplier = if n % 2 == 0 && index == n / 2 { 0.0 } else { k };
            *value = value.times_i(multiplier);
        } else {
            *value = value.scale(-k * k);
        }
    }

    fft1d(&mut spectrum, true);
    spectrum.into_iter().map(|value| value.re).collect()
}

/// Spectral derivative in x/y or a second/mixed derivative on an n-by-n grid.
pub fn derivative_2d(values: &[f64], n: usize, x_order: usize, y_order: usize) -> Vec<f64> {
    assert_eq!(values.len(), n * n);
    assert!(x_order + y_order <= 2);
    let k = modes(n);
    let mut spectrum = fft2_real(values, n);
    for y in 0..n {
        for x in 0..n {
            let index = y * n + x;
            let kx = k[x] as f64;
            let ky = k[y] as f64;
            match (x_order, y_order) {
                (1, 0) => {
                    let multiplier = if n % 2 == 0 && x == n / 2 { 0.0 } else { kx };
                    spectrum[index] = spectrum[index].times_i(multiplier);
                }
                (0, 1) => {
                    let multiplier = if n % 2 == 0 && y == n / 2 { 0.0 } else { ky };
                    spectrum[index] = spectrum[index].times_i(multiplier);
                }
                (2, 0) => spectrum[index] = spectrum[index].scale(-kx * kx),
                (0, 2) => spectrum[index] = spectrum[index].scale(-ky * ky),
                (1, 1) => spectrum[index] = spectrum[index].scale(-kx * ky),
                (0, 0) => {}
                _ => unreachable!(),
            }
        }
    }
    ifft2_real(&spectrum, n)
}

pub fn laplacian_2d(values: &[f64], n: usize) -> Vec<f64> {
    let k = modes(n);
    let mut spectrum = fft2_real(values, n);
    for y in 0..n {
        for x in 0..n {
            let kx = k[x] as f64;
            let ky = k[y] as f64;
            spectrum[y * n + x] = spectrum[y * n + x].scale(-(kx * kx + ky * ky));
        }
    }
    ifft2_real(&spectrum, n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn two_dimensional_fft_round_trip() {
        let n = 32;
        let values: Vec<f64> = (0..n * n)
            .map(|index| (index as f64 * 0.17).sin())
            .collect();
        let recovered = ifft2_real(&fft2_real(&values, n), n);
        let error = values
            .iter()
            .zip(recovered)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(error < 2e-14, "round-trip error was {error:e}");
    }

    #[test]
    fn represented_wave_has_exact_fourier_derivatives() {
        let n = 32;
        let dx = 2.0 * PI / n as f64;
        let mut g = Vec::with_capacity(n * n);
        for y in 0..n {
            for x in 0..n {
                g.push((3.0 * x as f64 * dx).sin() * (2.0 * y as f64 * dx).cos());
            }
        }
        let exact_dx = derivative_2d(&g, n, 1, 0);
        let max = exact_dx
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let x = (index % n) as f64 * dx;
                let y = (index / n) as f64 * dx;
                (value - 3.0 * (3.0 * x).cos() * (2.0 * y).cos()).abs()
            })
            .fold(0.0, f64::max);
        assert!(max < 2e-13, "Fourier derivative error was {max:e}");
    }
}
