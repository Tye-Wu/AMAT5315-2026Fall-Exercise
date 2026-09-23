use std::f64::consts::TAU;
use week4::fft::derivative_2d;

fn field(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let dx = TAU / n as f64;
    let xs: Vec<f64> = (0..n).map(|i| i as f64 * dx).collect();
    let ys = xs.clone();
    let mut g = vec![0.0; n * n];
    for (j, &y) in ys.iter().enumerate() {
        for (i, &x) in xs.iter().enumerate() {
            g[j * n + i] = (3.0 * x).sin() * (2.0 * y).cos();
        }
    }
    (xs, ys, g)
}

fn centered(values: &[f64], n: usize, dx: f64, axis: usize, order: usize) -> Vec<f64> {
    (0..n * n)
        .map(|index| {
            let x = index % n;
            let y = index / n;
            let (minus, plus) = if axis == 0 {
                (y * n + (x + n - 1) % n, y * n + (x + 1) % n)
            } else {
                (((y + n - 1) % n) * n + x, ((y + 1) % n) * n + x)
            };
            if order == 1 {
                (values[plus] - values[minus]) / (2.0 * dx)
            } else {
                (values[plus] - 2.0 * values[index] + values[minus]) / (dx * dx)
            }
        })
        .collect()
}

fn max_error(values: &[f64], exact: &[f64]) -> f64 {
    values
        .iter()
        .zip(exact)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}

fn compare(n: usize, fourier: bool) -> [f64; 4] {
    let (xs, ys, g) = field(n);
    let dx = TAU / n as f64;
    let mut exact_dx = Vec::with_capacity(n * n);
    let mut exact_dxx = Vec::with_capacity(n * n);
    let mut exact_dxdy = Vec::with_capacity(n * n);
    let mut exact_lap = Vec::with_capacity(n * n);
    for &y in &ys {
        for &x in &xs {
            let value = (3.0 * x).sin() * (2.0 * y).cos();
            exact_dx.push(3.0 * (3.0 * x).cos() * (2.0 * y).cos());
            exact_dxx.push(-9.0 * value);
            exact_dxdy.push(-6.0 * (3.0 * x).cos() * (2.0 * y).sin());
            exact_lap.push(-13.0 * value);
        }
    }
    let calculated = if fourier {
        let dxg = derivative_2d(&g, n, 1, 0);
        let dxxg = derivative_2d(&g, n, 2, 0);
        let dxdyg = derivative_2d(&g, n, 1, 1);
        let lap = derivative_2d(&g, n, 2, 0)
            .iter()
            .zip(derivative_2d(&g, n, 0, 2))
            .map(|(x, y)| x + y)
            .collect::<Vec<_>>();
        [
            max_error(&dxg, &exact_dx),
            max_error(&dxxg, &exact_dxx),
            max_error(&dxdyg, &exact_dxdy),
            max_error(&lap, &exact_lap),
        ]
    } else {
        let dxg = centered(&g, n, dx, 0, 1);
        let dxxg = centered(&g, n, dx, 0, 2);
        let dy_dxg = centered(&dxg, n, dx, 1, 1);
        let dyyg = centered(&g, n, dx, 1, 2);
        let lap: Vec<f64> = dxxg.iter().zip(&dyyg).map(|(x, y)| x + y).collect();
        [
            max_error(&dxg, &exact_dx),
            max_error(&dxxg, &exact_dxx),
            max_error(&dy_dxg, &exact_dxdy),
            max_error(&lap, &exact_lap),
        ]
    };
    calculated
}

fn main() {
    let fd32 = compare(32, false);
    let fd64 = compare(64, false);
    let spectral = compare(32, true);
    println!(
        "{:<12} {:>18} {:>18} {:>14} {:>18}",
        "Derivative", "FD, N=32", "FD, N=64", "ratio", "Fourier, N=32"
    );
    for (index, name) in ["dx g", "dxx g", "dxdy g", "Laplacian g"]
        .iter()
        .enumerate()
    {
        println!(
            "{:<12} {:>18.12e} {:>18.12e} {:>14.9} {:>18.12e}",
            name,
            fd32[index],
            fd64[index],
            fd32[index] / fd64[index],
            spectral[index],
        );
    }
}
