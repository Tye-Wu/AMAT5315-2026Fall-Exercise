pub trait Integrator {
    fn name(&self) -> &'static str;
    fn step(
        &self,
        state: &[f64],
        time: f64,
        step: f64,
        rate: &dyn Fn(f64, &[f64]) -> Vec<f64>,
    ) -> Vec<f64>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Euler;

#[derive(Clone, Copy, Debug, Default)]
pub struct Midpoint;

#[derive(Clone, Copy, Debug, Default)]
pub struct Rk4;

#[derive(Clone, Copy, Debug, Default)]
pub struct EqualWeightRk4;

fn shifted(state: &[f64], rate: &[f64], factor: f64) -> Vec<f64> {
    state
        .iter()
        .zip(rate)
        .map(|(value, derivative)| value + factor * derivative)
        .collect()
}

impl Integrator for Euler {
    fn name(&self) -> &'static str {
        "euler"
    }

    fn step(
        &self,
        state: &[f64],
        time: f64,
        step: f64,
        rate: &dyn Fn(f64, &[f64]) -> Vec<f64>,
    ) -> Vec<f64> {
        shifted(state, &rate(time, state), step)
    }
}

impl Integrator for Midpoint {
    fn name(&self) -> &'static str {
        "rk2"
    }

    fn step(
        &self,
        state: &[f64],
        time: f64,
        step: f64,
        rate: &dyn Fn(f64, &[f64]) -> Vec<f64>,
    ) -> Vec<f64> {
        let k1 = rate(time, state);
        let trial = shifted(state, &k1, 0.5 * step);
        let k2 = rate(time + 0.5 * step, &trial);
        shifted(state, &k2, step)
    }
}

impl Integrator for Rk4 {
    fn name(&self) -> &'static str {
        "rk4"
    }

    fn step(
        &self,
        state: &[f64],
        time: f64,
        step: f64,
        rate: &dyn Fn(f64, &[f64]) -> Vec<f64>,
    ) -> Vec<f64> {
        let k1 = rate(time, state);
        let k2 = rate(time + 0.5 * step, &shifted(state, &k1, 0.5 * step));
        let k3 = rate(time + 0.5 * step, &shifted(state, &k2, 0.5 * step));
        let k4 = rate(time + step, &shifted(state, &k3, step));
        state
            .iter()
            .zip(k1.iter().zip(k2.iter().zip(k3.iter().zip(k4.iter()))))
            .map(|(value, (a, (b, (c, d))))| value + (step / 6.0) * (a + 2.0 * b + 2.0 * c + d))
            .collect()
    }
}

impl Integrator for EqualWeightRk4 {
    fn name(&self) -> &'static str {
        "equal-weight RK4"
    }

    fn step(
        &self,
        state: &[f64],
        time: f64,
        step: f64,
        rate: &dyn Fn(f64, &[f64]) -> Vec<f64>,
    ) -> Vec<f64> {
        let k1 = rate(time, state);
        let k2 = rate(time + 0.5 * step, &shifted(state, &k1, 0.5 * step));
        let k3 = rate(time + 0.5 * step, &shifted(state, &k2, 0.5 * step));
        let k4 = rate(time + step, &shifted(state, &k3, step));
        state
            .iter()
            .zip(k1.iter().zip(k2.iter().zip(k3.iter().zip(k4.iter()))))
            .map(|(value, (a, (b, (c, d))))| value + (step / 4.0) * (a + b + c + d))
            .collect()
    }
}

pub fn integrate(
    integrator: &dyn Integrator,
    mut state: Vec<f64>,
    initial_time: f64,
    final_time: f64,
    step: f64,
    rate: &dyn Fn(f64, &[f64]) -> Vec<f64>,
) -> Vec<f64> {
    assert!(step > 0.0);
    let mut time = initial_time;
    while time < final_time {
        let h = step.min(final_time - time);
        state = integrator.step(&state, time, h, rate);
        time += h;
    }
    state
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_step_uses_each_methods_runge_kutta_weights() {
        let rate = |_: f64, y: &[f64]| vec![y[0]];
        let y0 = [1.0];
        let h = 0.1;
        assert!((Euler.step(&y0, 0.0, h, &rate)[0] - 1.1).abs() < 1e-15);
        assert!((Midpoint.step(&y0, 0.0, h, &rate)[0] - 1.105).abs() < 1e-15);
        let rk4 = Rk4.step(&y0, 0.0, h, &rate)[0];
        assert!((rk4 - (1.0 + h + h * h / 2.0 + h.powi(3) / 6.0 + h.powi(4) / 24.0)).abs() < 1e-15);
    }

    #[test]
    fn methods_reach_their_expected_orders_on_a_fourier_wave() {
        use crate::line::{
            Derivative, advection_diffusion_rate, fourier_grid, single_wave, single_wave_exact,
        };

        let n = 128;
        let (x, _dx) = fourier_grid(n);
        let initial = single_wave(&x, 2, 0.0);
        let c = 1.0;
        let nu = 0.01;
        let final_time = 0.7;
        let hs = [0.04, 0.02, 0.01];
        let expected = single_wave_exact(&x, 2, c, nu, final_time, 0.0);
        let rate = |_: f64, y: &[f64]| advection_diffusion_rate(y, c, nu, Derivative::Fourier);

        let observed_slope = |integrator: &dyn Integrator| {
            let errors: Vec<f64> = hs
                .iter()
                .map(|&h| {
                    let actual = integrate(integrator, initial.clone(), 0.0, final_time, h, &rate);
                    actual
                        .iter()
                        .zip(&expected)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0, f64::max)
                })
                .collect();
            (errors[0] / errors[2]).ln() / (hs[0] / hs[2]).ln()
        };

        assert!((observed_slope(&Euler) - 1.0).abs() < 0.2);
        assert!((observed_slope(&Midpoint) - 2.0).abs() < 0.2);
        assert!((observed_slope(&Rk4) - 4.0).abs() < 0.3);
    }
}
