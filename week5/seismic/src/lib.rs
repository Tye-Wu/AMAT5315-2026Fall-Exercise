use serde::{Deserialize, Serialize};
pub mod checkpoint;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Experiment {
    pub name: String,
    pub nx: usize,
    pub nz: usize,
    pub steps: usize,
    pub dx: f64,
    pub dt: f64,
    pub length_unit_m: f64,
    pub time_unit_s: f64,
    pub source_frequency: f64,
    pub source_peak_time: f64,
    pub source_amplitude: f64,
    pub sponge_width: usize,
    pub sponge_strength: f64,
    pub background: Vec<Vec<f64>>,
    pub perturbation: Vec<Vec<f64>>,
    pub shots: Vec<[f64; 2]>,
    pub receivers: Vec<[usize; 2]>,
}

impl Experiment {
    pub fn validate(&self) -> Result<(), String> {
        if self.nx < 3 || self.nz < 3 || self.steps == 0 {
            return Err("grid and step counts must be positive".into());
        }
        for (label, grid) in [
            ("background", &self.background),
            ("perturbation", &self.perturbation),
        ] {
            if grid.len() != self.nz || grid.iter().any(|r| r.len() != self.nx) {
                return Err(format!("{label} must have shape nz × nx"));
            }
        }
        if self.shots.is_empty() || self.receivers.is_empty() {
            return Err("at least one shot and receiver required".into());
        }
        if self
            .receivers
            .iter()
            .any(|&[x, z]| x >= self.nx || z >= self.nz)
        {
            return Err("receiver outside grid".into());
        }
        if self
            .shots
            .iter()
            .any(|&[x, z]| x < 0.0 || z < 0.0 || x >= self.nx as f64 || z >= self.nz as f64)
        {
            return Err("shot outside grid".into());
        }
        if self
            .background
            .iter()
            .flatten()
            .any(|v| !v.is_finite() || *v <= 0.0)
        {
            return Err("background speeds must be finite and positive".into());
        }
        Ok(())
    }

    pub fn velocities(&self, perturbed: bool) -> Vec<f64> {
        self.background
            .iter()
            .flatten()
            .zip(self.perturbation.iter().flatten())
            .map(|(&v, &dv)| if perturbed { v + dv } else { v })
            .collect()
    }

    pub fn sponge(&self) -> Vec<f64> {
        let mut s = vec![0.0; self.nx * self.nz];
        for z in 0..self.nz {
            for x in 0..self.nx {
                let d = x.min(z).min(self.nx - 1 - x).min(self.nz - 1 - z);
                if d < self.sponge_width {
                    let y = 1.0 - d as f64 / self.sponge_width as f64;
                    s[z * self.nx + x] = self.sponge_strength * y * y;
                }
            }
        }
        s
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct ForwardResult {
    pub traces: Vec<Vec<Vec<f64>>>,
    pub wavefield: Vec<Vec<f32>>,
    pub echo: Vec<Vec<f32>>,
    pub frame_steps: Vec<usize>,
    pub trace_l2: f64,
    pub shot_maxima: Vec<f64>,
    pub max_trace_step: Vec<usize>,
}

pub fn source_wavelet(exp: &Experiment, t: f64) -> f64 {
    let theta = std::f64::consts::PI * exp.source_frequency * (t - exp.source_peak_time);
    exp.source_amplitude * (1.0 - 2.0 * theta * theta) * (-theta * theta).exp()
}

pub fn source_grid(exp: &Experiment, t: f64, source: [f64; 2]) -> Vec<f64> {
    let amp = source_wavelet(exp, t);
    (0..exp.nz)
        .flat_map(|z| {
            (0..exp.nx).map(move |x| {
                let dx = x as f64 - source[0];
                let dz = z as f64 - source[1];
                amp * (-0.5 * (dx * dx + dz * dz)).exp()
            })
        })
        .collect()
}

pub fn step(
    exp: &Experiment,
    c: &[f64],
    sigma: &[f64],
    prev: &[f64],
    curr: &[f64],
    t: f64,
    source: [f64; 2],
) -> Vec<f64> {
    let mut next = vec![0.0; exp.nx * exp.nz];
    let dx2 = exp.dx * exp.dx;
    let q = source_wavelet(exp, t);
    for z in 1..exp.nz - 1 {
        for x in 1..exp.nx - 1 {
            let i = z * exp.nx + x;
            let lap = (curr[i - 1] + curr[i + 1] + curr[i - exp.nx] + curr[i + exp.nx]
                - 4.0 * curr[i])
                / dx2;
            let dxs = x as f64 - source[0];
            let dzs = z as f64 - source[1];
            let source_term = q * (-0.5 * (dxs * dxs + dzs * dzs)).exp();
            let sdt = sigma[i] * exp.dt;
            next[i] = (2.0 * curr[i] - (1.0 - sdt) * prev[i]
                + exp.dt * exp.dt * (c[i] * c[i] * lap + source_term))
                / (1.0 + sdt);
        }
    }
    next
}

pub fn forward_shot(
    exp: &Experiment,
    velocity: &[f64],
    sigma: &[f64],
    source: [f64; 2],
    every: usize,
) -> (Vec<Vec<f64>>, Vec<Vec<f32>>) {
    let mut prev = vec![0.0; exp.nx * exp.nz];
    let mut curr = vec![0.0; exp.nx * exp.nz];
    let mut traces = vec![vec![0.0; exp.receivers.len()]; exp.steps];
    let mut frames = vec![curr.iter().map(|&v| v as f32).collect()];
    for n in 0..exp.steps {
        let next = step(
            exp,
            velocity,
            sigma,
            &prev,
            &curr,
            n as f64 * exp.dt,
            source,
        );
        for (ri, &[x, z]) in exp.receivers.iter().enumerate() {
            traces[n][ri] = next[z * exp.nx + x];
        }
        prev = curr;
        curr = next;
        if (n + 1) % every == 0 {
            frames.push(curr.iter().map(|&v| v as f32).collect());
        }
    }
    (traces, frames)
}

pub fn run_forward(
    exp: &Experiment,
    perturbed: bool,
    every: usize,
) -> Result<ForwardResult, String> {
    exp.validate()?;
    if every == 0 {
        return Err("recording interval must be positive".into());
    }
    let velocity = exp.velocities(perturbed);
    let base_velocity = exp.velocities(false);
    let sigma = exp.sponge();
    let mut traces = Vec::with_capacity(exp.shots.len());
    let mut shot_maxima = Vec::with_capacity(exp.shots.len());
    let mut max_trace_step = Vec::with_capacity(exp.shots.len());
    let mut first_wavefield = Vec::new();
    let mut first_echo = Vec::new();
    let mut frame_steps = vec![0];
    frame_steps.extend((1..=exp.steps).filter(|n| n % every == 0));
    let mut sum2 = 0.0;
    for (shot_i, &shot) in exp.shots.iter().enumerate() {
        let (gather, frames) = forward_shot(exp, &velocity, &sigma, shot, every);
        for frame in &gather {
            for &v in frame {
                sum2 += v * v;
            }
        }
        let (mut peak, mut peak_step) = (0.0_f64, 0);
        for (step, frame) in gather.iter().enumerate() {
            for value in frame {
                if value.abs() > peak {
                    peak = value.abs();
                    peak_step = step;
                }
            }
        }
        shot_maxima.push(peak);
        max_trace_step.push(peak_step);
        traces.push(gather);
        if shot_i == 0 {
            first_wavefield = frames.clone();
            let (pert_frames, base_frames) = if perturbed {
                let (_, b) = forward_shot(exp, &base_velocity, &sigma, shot, every);
                (frames.clone(), b)
            } else {
                let (_, p) = forward_shot(exp, &exp.velocities(true), &sigma, shot, every);
                (p, frames.clone())
            };
            first_echo = pert_frames
                .iter()
                .zip(base_frames.iter())
                .map(|(p, b)| p.iter().zip(b).map(|(&a, &v)| a - v).collect())
                .collect();
        }
    }
    Ok(ForwardResult {
        traces,
        wavefield: first_wavefield,
        echo: first_echo,
        frame_steps,
        trace_l2: sum2.sqrt(),
        shot_maxima,
        max_trace_step,
    })
}

#[cfg(feature = "enzyme")]
pub mod enzyme {
    use super::Experiment;
    unsafe extern "C" {
        fn enzyme_step_primal(
            nx: usize,
            nz: usize,
            dt: f64,
            dx: f64,
            c: *const f64,
            sigma: *const f64,
            prev: *const f64,
            curr: *const f64,
            q: *const f64,
            next: *mut f64,
        );
        fn enzyme_step_jvp(
            nx: usize,
            nz: usize,
            dt: f64,
            dx: f64,
            c: *const f64,
            dc: *const f64,
            sigma: *const f64,
            prev: *const f64,
            dprev: *const f64,
            curr: *const f64,
            dcurr: *const f64,
            q: *const f64,
            next: *mut f64,
            dnext: *mut f64,
        );
        fn enzyme_step_vjp(
            nx: usize,
            nz: usize,
            dt: f64,
            dx: f64,
            c: *const f64,
            dc: *mut f64,
            sigma: *const f64,
            prev: *const f64,
            dprev: *mut f64,
            curr: *const f64,
            dcurr: *mut f64,
            q: *const f64,
            next: *mut f64,
            dnext: *mut f64,
        );
    }

    fn check(n: usize, arrays: &[&[f64]]) -> Result<(), String> {
        if arrays.iter().any(|a| a.len() != n) {
            return Err("Enzyme kernel input length mismatch".into());
        }
        Ok(())
    }

    pub fn primal(
        e: &Experiment,
        c: &[f64],
        sigma: &[f64],
        prev: &[f64],
        curr: &[f64],
        q: &[f64],
    ) -> Result<Vec<f64>, String> {
        let n = e.nx * e.nz;
        check(n, &[c, sigma, prev, curr, q])?;
        let mut next = vec![0.0; n];
        unsafe {
            enzyme_step_primal(
                e.nx,
                e.nz,
                e.dt,
                e.dx,
                c.as_ptr(),
                sigma.as_ptr(),
                prev.as_ptr(),
                curr.as_ptr(),
                q.as_ptr(),
                next.as_mut_ptr(),
            );
        }
        Ok(next)
    }

    pub fn jvp(
        e: &Experiment,
        c: &[f64],
        dc: &[f64],
        sigma: &[f64],
        prev: &[f64],
        dprev: &[f64],
        curr: &[f64],
        dcurr: &[f64],
        q: &[f64],
    ) -> Result<(Vec<f64>, Vec<f64>), String> {
        let n = e.nx * e.nz;
        check(n, &[c, dc, sigma, prev, dprev, curr, dcurr, q])?;
        let mut next = vec![0.0; n];
        let mut dnext = vec![0.0; n];
        unsafe {
            enzyme_step_jvp(
                e.nx,
                e.nz,
                e.dt,
                e.dx,
                c.as_ptr(),
                dc.as_ptr(),
                sigma.as_ptr(),
                prev.as_ptr(),
                dprev.as_ptr(),
                curr.as_ptr(),
                dcurr.as_ptr(),
                q.as_ptr(),
                next.as_mut_ptr(),
                dnext.as_mut_ptr(),
            );
        }
        Ok((next, dnext))
    }

    pub fn vjp(
        e: &Experiment,
        c: &[f64],
        sigma: &[f64],
        prev: &[f64],
        curr: &[f64],
        q: &[f64],
        seed: &[f64],
    ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), String> {
        let n = e.nx * e.nz;
        check(n, &[c, sigma, prev, curr, q, seed])?;
        let mut dc = vec![0.0; n];
        let mut dprev = vec![0.0; n];
        let mut dcurr = vec![0.0; n];
        let mut next = vec![0.0; n];
        let mut dnext = seed.to_vec();
        unsafe {
            enzyme_step_vjp(
                e.nx,
                e.nz,
                e.dt,
                e.dx,
                c.as_ptr(),
                dc.as_mut_ptr(),
                sigma.as_ptr(),
                prev.as_ptr(),
                dprev.as_mut_ptr(),
                curr.as_ptr(),
                dcurr.as_mut_ptr(),
                q.as_ptr(),
                next.as_mut_ptr(),
                dnext.as_mut_ptr(),
            );
        }
        Ok((dc, dprev, dcurr))
    }
}

#[cfg(feature = "enzyme")]
#[derive(Clone, Debug, Serialize)]
pub struct BornResult {
    pub traces: Vec<Vec<Vec<f64>>>,
    pub l2: f64,
}

#[cfg(feature = "enzyme")]
#[derive(Clone, Debug, Serialize)]
pub struct AdjointResult {
    pub image: Vec<Vec<f64>>,
    pub transpose_left: f64,
    pub transpose_right: f64,
    pub transpose_relative_error: f64,
    pub wavefield: Vec<Vec<f32>>,
    pub frame_steps: Vec<usize>,
}

#[cfg(feature = "enzyme")]
#[derive(Clone, Debug, Serialize)]
pub struct TreeverseResult {
    pub image: Vec<Vec<f64>>,
    pub transpose_left: f64,
    pub transpose_right: f64,
    pub transpose_relative_error: f64,
    pub wavefield: Vec<Vec<f32>>,
    pub frame_steps: Vec<usize>,
    pub actions: Vec<Vec<checkpoint::ActionRecord>>,
    pub reverse_calls: usize,
    pub scheduler_forward_calls: usize,
    pub peak_saved_states: usize,
    pub peak_saved_bytes: usize,
}

#[cfg(feature = "enzyme")]
pub fn run_born_enzyme(exp: &Experiment) -> Result<BornResult, String> {
    exp.validate()?;
    let n = exp.nx * exp.nz;
    let c = exp.velocities(false);
    let dm = exp
        .perturbation
        .iter()
        .flatten()
        .copied()
        .collect::<Vec<_>>();
    let sigma = exp.sponge();
    let mut all = Vec::with_capacity(exp.shots.len());
    let mut sum2 = 0.0;
    for &shot in &exp.shots {
        let (mut prev, mut curr, mut dprev, mut dcurr) =
            (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        let mut trace = vec![vec![0.0; exp.receivers.len()]; exp.steps];
        for k in 0..exp.steps {
            let q = source_grid(exp, k as f64 * exp.dt, shot);
            let (next, dnext) =
                enzyme::jvp(exp, &c, &dm, &sigma, &prev, &dprev, &curr, &dcurr, &q)?;
            for (ri, &[x, z]) in exp.receivers.iter().enumerate() {
                trace[k][ri] = dnext[z * exp.nx + x];
                sum2 += trace[k][ri] * trace[k][ri];
            }
            prev = curr;
            curr = next;
            dprev = dcurr;
            dcurr = dnext;
        }
        all.push(trace);
    }
    Ok(BornResult {
        traces: all,
        l2: sum2.sqrt(),
    })
}

#[cfg(feature = "enzyme")]
pub fn run_adjoint_enzyme(
    exp: &Experiment,
    data: &[f64],
    every: usize,
) -> Result<AdjointResult, String> {
    exp.validate()?;
    if every == 0 {
        return Err("recording interval must be positive".into());
    }
    let ns = exp.shots.len();
    let nt = exp.steps;
    let nr = exp.receivers.len();
    let n = exp.nx * exp.nz;
    if data.len() != ns * nt * nr {
        return Err(format!(
            "data length mismatch: expected {}, got {}",
            ns * nt * nr,
            data.len()
        ));
    }
    let c = exp.velocities(false);
    let sigma = exp.sponge();
    let mut image = vec![0.0; n];
    let mut left = 0.0;
    let mut first_frames = Vec::new();
    let mut frame_steps = Vec::new();
    for (si, &shot) in exp.shots.iter().enumerate() {
        let (mut prev, mut curr) = (vec![0.0; n], vec![0.0; n]);
        let mut history = Vec::with_capacity(nt);
        for k in 0..nt {
            history.push((prev.clone(), curr.clone()));
            let q = source_grid(exp, k as f64 * exp.dt, shot);
            let next = enzyme::primal(exp, &c, &sigma, &prev, &curr, &q)?;
            prev = curr;
            curr = next;
        }
        let (mut lprev, mut lcurr) = (vec![0.0; n], vec![0.0; n]);
        let mut frames = Vec::new();
        let mut shot_frame_steps = Vec::new();
        for k in (0..nt).rev() {
            let output_first = std::mem::take(&mut lprev);
            let mut lnext = std::mem::take(&mut lcurr);
            for (ri, &[x, z]) in exp.receivers.iter().enumerate() {
                let w = data[(si * nt + k) * nr + ri];
                left += w * w;
                lnext[z * exp.nx + x] += w;
            }
            let (p, curr_state) = &history[k];
            let q = source_grid(exp, k as f64 * exp.dt, shot);
            let (dc, dp, dcurr) = enzyme::vjp(exp, &c, &sigma, p, curr_state, &q, &lnext)?;
            for i in 0..n {
                image[i] += dc[i];
            }
            lprev = dp;
            lcurr = (0..n).map(|i| output_first[i] + dcurr[i]).collect();
            // The output-state first component u^n carries directly to the input
            // state's second component; the VJP contributes another path to u^n.
            if k % every == 0 {
                frames.push(lcurr.iter().map(|&v| v as f32).collect());
                shot_frame_steps.push(k);
            }
        }
        if si == 0 {
            first_frames = frames;
            frame_steps = shot_frame_steps;
        }
    }
    let perturb = exp.perturbation.iter().flatten().collect::<Vec<_>>();
    let right = perturb
        .iter()
        .zip(&image)
        .map(|(&m, &g)| m * g)
        .sum::<f64>();
    let err = (left - right).abs() / left.abs().max(right.abs()).max(f64::MIN_POSITIVE);
    let image = image.chunks(exp.nx).map(|r| r.to_vec()).collect();
    Ok(AdjointResult {
        image,
        transpose_left: left,
        transpose_right: right,
        transpose_relative_error: err,
        wavefield: first_frames,
        frame_steps,
    })
}

#[cfg(feature = "enzyme")]
pub fn run_adjoint_treeverse(
    exp: &Experiment,
    data: &[f64],
    extra_slots: usize,
    every: usize,
) -> Result<TreeverseResult, String> {
    exp.validate()?;
    if every == 0 {
        return Err("recording interval must be positive".into());
    }
    let (actions, expected_peak) = checkpoint::treeverse_schedule(exp.steps, extra_slots)?;
    let ns = exp.shots.len();
    let nt = exp.steps;
    let nr = exp.receivers.len();
    let n = exp.nx * exp.nz;
    if data.len() != ns * nt * nr {
        return Err(format!(
            "data length mismatch: expected {}, got {}",
            ns * nt * nr,
            data.len()
        ));
    }
    let c = exp.velocities(false);
    let sigma = exp.sponge();
    let mut image = vec![0.0; n];
    let mut left = 0.0;
    let mut all_actions = Vec::with_capacity(ns);
    let mut first_frames = Vec::new();
    let mut frame_steps = Vec::new();
    let mut total_calls = 0;
    let mut total_grads = 0;
    let mut peak_states = 1;

    for (si, &shot) in exp.shots.iter().enumerate() {
        let zero = vec![0.0; n];
        let mut saved = std::collections::BTreeMap::from([(0_usize, (zero.clone(), zero.clone()))]);
        let mut working = None::<(usize, Vec<f64>, Vec<f64>)>;
        let (mut lprev, mut lcurr) = (vec![0.0; n], vec![0.0; n]);
        let mut shot_frames = Vec::new();
        let mut shot_frame_steps = Vec::new();
        let mut calls = 0;
        let mut grads = 0;

        for action in &actions {
            match action.action.as_str() {
                "restore" => {
                    let (prev, curr) = saved
                        .get(&action.n)
                        .ok_or_else(|| format!("invalid restore s_{}", action.n))?
                        .clone();
                    working = Some((action.n, prev, curr));
                }
                "call" => {
                    let (step_n, prev, curr) = working
                        .take()
                        .ok_or("Treeverse call without working state")?;
                    if step_n != action.n {
                        return Err(format!(
                            "call expected s_{}, working state is s_{step_n}",
                            action.n
                        ));
                    }
                    let q = source_grid(exp, action.n as f64 * exp.dt, shot);
                    let next = enzyme::primal(exp, &c, &sigma, &prev, &curr, &q)?;
                    working = Some((step_n + 1, curr, next));
                    calls += 1;
                }
                "store" => {
                    let (step_n, prev, curr) = working
                        .as_ref()
                        .ok_or("Treeverse store without working state")?;
                    if *step_n != action.n {
                        return Err(format!(
                            "store expected s_{}, working state is s_{step_n}",
                            action.n
                        ));
                    }
                    if saved
                        .insert(action.n, (prev.clone(), curr.clone()))
                        .is_some()
                    {
                        return Err(format!("duplicate store s_{}", action.n));
                    }
                }
                "grad" => {
                    let (p, curr_state) = saved
                        .get(&action.n)
                        .ok_or_else(|| format!("missing VJP input s_{}", action.n))?;
                    let output_first = std::mem::take(&mut lprev);
                    let mut lnext = std::mem::take(&mut lcurr);
                    for (ri, &[x, z]) in exp.receivers.iter().enumerate() {
                        let w = data[(si * nt + action.n) * nr + ri];
                        left += w * w;
                        lnext[z * exp.nx + x] += w;
                    }
                    let q = source_grid(exp, action.n as f64 * exp.dt, shot);
                    let (dc, dp, dcurr) = enzyme::vjp(exp, &c, &sigma, p, curr_state, &q, &lnext)?;
                    for i in 0..n {
                        image[i] += dc[i];
                    }
                    lprev = dp;
                    lcurr = (0..n).map(|i| output_first[i] + dcurr[i]).collect();
                    if action.n % every == 0 {
                        shot_frames.push(lcurr.iter().map(|&v| v as f32).collect());
                        shot_frame_steps.push(action.n);
                    }
                    grads += 1;
                }
                "fetch" => {
                    if action.n == 0 || saved.remove(&action.n).is_none() {
                        return Err(format!("invalid fetch s_{}", action.n));
                    }
                }
                other => return Err(format!("unknown Treeverse action {other}")),
            }
            if saved.len() != action.saved_states {
                return Err(format!(
                    "saved-state audit mismatch after {} s_{}: expected {}, got {}",
                    action.action,
                    action.n,
                    action.saved_states,
                    saved.len()
                ));
            }
            peak_states = peak_states.max(saved.len());
        }
        if grads != nt {
            return Err(format!("shot {si} reversed {grads} steps, expected {nt}"));
        }
        if calls != actions.iter().filter(|a| a.action == "call").count() {
            return Err(format!(
                "shot {si} replay count differs from planned schedule"
            ));
        }
        if saved.len() != 1 || !saved.contains_key(&0) {
            return Err(format!("shot {si} did not finish with only s_0 saved"));
        }
        if si == 0 {
            first_frames = shot_frames;
            frame_steps = shot_frame_steps;
        }
        total_calls += calls;
        total_grads += grads;
        all_actions.push(actions.clone());
    }
    if peak_states != expected_peak {
        return Err(format!(
            "peak state count {peak_states} differs from schedule prediction {expected_peak}"
        ));
    }
    let perturb = exp.perturbation.iter().flatten().collect::<Vec<_>>();
    let right = perturb
        .iter()
        .zip(&image)
        .map(|(&m, &g)| m * g)
        .sum::<f64>();
    let err = (left - right).abs() / left.abs().max(right.abs()).max(f64::MIN_POSITIVE);
    let image = image.chunks(exp.nx).map(|r| r.to_vec()).collect();
    Ok(TreeverseResult {
        image,
        transpose_left: left,
        transpose_right: right,
        transpose_relative_error: err,
        wavefield: first_frames,
        frame_steps,
        actions: all_actions,
        reverse_calls: total_grads,
        scheduler_forward_calls: total_calls,
        peak_saved_states: peak_states,
        peak_saved_bytes: peak_states * 2 * exp.nx * exp.nz * 8,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> Experiment {
        Experiment {
            name: "test".into(),
            nx: 5,
            nz: 5,
            steps: 2,
            dx: 1.0,
            dt: 0.1,
            length_unit_m: 100.0,
            time_unit_s: 0.1,
            source_frequency: 0.08,
            source_peak_time: 0.0,
            source_amplitude: 1.0,
            sponge_width: 2,
            sponge_strength: 0.4,
            background: vec![vec![1.8; 5]; 5],
            perturbation: vec![vec![0.0; 5]; 5],
            shots: vec![[2.0, 2.0]],
            receivers: vec![[2, 2]],
        }
    }

    #[test]
    fn ricker_has_unit_peak_and_source_is_gaussian_footprint() {
        let e = tiny();
        assert!((source_wavelet(&e, 0.0) - 1.0).abs() < 1e-15);
        let u = step(
            &e,
            &e.velocities(false),
            &e.sponge(),
            &vec![0.0; 25],
            &vec![0.0; 25],
            0.0,
            [2.0, 2.0],
        );
        assert!(u[2 * 5 + 2] > u[2 * 5 + 1]);
        assert!(u[0].abs() == 0.0 && u[4].abs() == 0.0 && u[20].abs() == 0.0);
    }

    #[test]
    fn trace_shape_and_recorded_steps_are_exact() {
        let e = tiny();
        let r = run_forward(&e, false, 1).unwrap();
        assert_eq!(r.traces[0].len(), 2);
        assert_eq!(r.wavefield.len(), 3);
        assert_eq!(r.frame_steps, vec![0, 1, 2]);
    }
}
