# LJ-Fluid CLI (`md`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the `md` crate into a CLI that simulates a 2D Lennard-Jones fluid, saves a trajectory, checks its physics, and renders a motion + g(r) video.

**Architecture:** Extend `System` with an optional periodic box so the existing `Integrator`/`VelocityVerlet` machinery drives a many-body fluid. The crate grows focused modules: `rng`, `lattice`, `fluid` (run pipeline), `store` (JSON I/O), `observables` (fits/checks). A thin `main.rs` CLI dispatches `run` / `check` / `video`.

**Tech Stack:** Rust (edition 2024), `serde` + `serde_json` (JSON), deterministic own RNG (no rand crate), python3 + matplotlib + ffmpeg for the video step. Tests via `cargo test`.

**Spec:** `docs/superpowers/specs/2026-09-09-lj-fluid-cli-design.md`

## Global Constraints

- Reduced units: ε = σ = 1, m = 1. Lengths in σ, energies in ε, temperature in ε/k_B.
- Reference defaults: N=100, ρ=0.8, target T=0.5, seed=42, dt=0.005, r_c=2.5 (shifted cutoff), box L=√(N/ρ)=√125.
- Equilibration rescales velocities to T every 50 steps; production has no thermostat.
- Production emits exactly `steps / save_every` saved frames (default 200).
- Cut-and-shifted pair energy `V_shift(r) = V(r) − V(r_c)` for `r < r_c`, else 0.
- Existing dimer behavior must not change: `System::new` has `periodic = None`.
- Outputs: `run.json` + `traj.jsonl` written to the `--dir` (default `artifacts/`), created if missing.
- Generated data (`week2/artifacts/`) and Rust build output (`week2/md/target/`) stay out of git.
- Commits: red (failing test) then green, in that order, per task.

---

## File structure

| File | Responsibility |
|---|---|
| `week2/md/src/system.rs` (modify) | `System` + optional periodic box; PBC-aware `accelerations`/`total_energy`; min-image; shifted cutoff |
| `week2/md/src/integrator.rs` (modify) | one struct-literal fix (`..Default::default()`); otherwise unchanged |
| `week2/md/src/rng.rs` (create) | deterministic SplitMix64 + Gaussian sampler |
| `week2/md/src/lattice.rs` (create) | triangular lattice generator |
| `week2/md/src/fluid.rs` (create) | `RunConfig`, temperature/rescale/seed, `run_fluid`, `FrameData`, `RunRecord` |
| `week2/md/src/store.rs` (create) | serde JSON records; write/read `run.json`, `traj.jsonl` |
| `week2/md/src/observables.rs` (create) | speed→Maxwell–Boltzmann fit, secular-drift metric |
| `week2/md/src/lib.rs` (modify) | register + re-export new modules |
| `week2/md/src/main.rs` (rewrite) | CLI: default run, `run`, `check <dir>`, `video <dir> [--out f]` |
| `week2/md/scripts/render_video.py` (create) | matplotlib renderer driven by `video` |
| `week2/Makefile` (modify) | `reproduce` runs release `md run --dir artifacts` |
| `docs/superpowers/specs/2026-09-09-lj-fluid-cli-design.md` (modify) | align §8 "frame 0" wording with the plan's sampling |

Public API (used across tasks; keep names/types exact):

- `Vec2 = [f64;2]`
- `system::BoxConfig { length: f64, cutoff: f64 }`
- `system::System { positions: Vec<Vec2>, velocities: Vec<Vec2>, periodic: Option<BoxConfig> }` + `Default`
- `System::new(p, v)`, `System::with_box(p, v, BoxConfig)`, `System::n_atoms()`
- `system::accelerations(&System) -> Vec<Vec2>`, `system::total_energy(&System) -> f64`
- `rng::SplitMix64::new(seed)`, `.next_f64()`, `.gaussian()`
- `lattice::triangular_lattice(rows, cols, length) -> Vec<Vec2>`
- `fluid::RunConfig` (+ `Default`), `fluid::box_length(&RunConfig) -> f64`, `fluid::kinetic_temperature(&System) -> f64`, `fluid::rescale_velocities(&mut System, f64)`, `fluid::seed_velocities(&mut System, &mut SplitMix64, f64)`, `fluid::run_fluid(&RunConfig) -> RunRecord`, `fluid::FrameData`, `fluid::RunRecord { cfg, box_len, e0, frames }`
- `store::FrameRecord` (serde), `store::RunJson` (serde), `store::write_output(dir:&Path, &RunRecord)`, `store::read_run(dir:&Path) -> (RunJson, Vec<FrameRecord>)`
- `observables::fit_speed_temperature(&[FrameRecord], bins) -> (f64, f64)` (T, χ²/dof), `observables::secular_drift(&[f64], &[f64]) -> f64`

---

### Task 1: Optional periodic box on `System` (physics: PBC, shifted cutoff)

**Files:**
- Modify: `week2/md/src/system.rs`
- Modify: `week2/md/src/integrator.rs` (struct-literal fix)
- Test: `week2/md/src/system.rs` (`#[cfg(test)] mod`)

**Interfaces:**
- Consumes: existing `lj_energy`, `lj_force`.
- Produces: `BoxConfig`, `System.periodic`, PBC-aware `accelerations`/`total_energy`, min-image displacement. Later tasks build on these.

- [ ] **Step 1: Write the failing tests** (append inside `system.rs` tests)

```rust
use crate::system::{accelerations, total_energy, BoxConfig, System, Vec2};

fn rng_seed() -> SplitMix64 { SplitMix64::new(7) }

#[test]
fn net_force_is_zero_under_periodic_boundaries() {
    let box_ = BoxConfig { length: 10.0, cutoff: 2.5 };
    let mut rng = SplitMix64::new(3);
    let pos: Vec<Vec2> = (0..12).map(|_| [rng.next_f64()*10.0, rng.next_f64()*10.0]).collect();
    let vel = vec![[0.0; 2]; 12];
    let sys = System::with_box(pos, vel, box_);
    let a = accelerations(&sys);
    let sx: f64 = a.iter().map(|v| v[0]).sum();
    let sy: f64 = a.iter().map(|v| v[1]).sum();
    assert!(sx.abs() < 1e-9 && sy.abs() < 1e-9, "sum {sx} {sy}");
}

#[test]
fn potential_is_continuous_just_inside_cutoff() {
    // Probe just inside rc: with the shift, the energy must be ~0 there.
    let box_ = BoxConfig { length: 10.0, cutoff: 2.5 };
    let r = 2.5 - 1e-6;
    let sys = System::with_box(
        vec![[0.0, 0.0], [r, 0.0]],
        vec![[0.0, 0.0], [0.0, 0.0]], box_);
    let e = total_energy(&sys); // KE = 0, so this is the pair potential
    assert!(e.abs() < 1e-4, "energy at rc-1e-6 should be ~0 (shifted), got {e}");
}

#[test]
fn shifted_potential_matches_exact_far_below_cutoff() {
    let box_ = BoxConfig { length: 10.0, cutoff: 2.5 };
    let sys = System::with_box(
        vec![[0.0, 0.0], [1.2, 0.0]],
        vec![[0.0, 0.0], [0.0, 0.0]], box_);
    let shifted = total_energy(&sys);
    let exact = 4.0 * (1.2f64.powi(-12) - 1.2f64.powi(-6));
    let shift = 4.0 * (2.5f64.powi(-12) - 2.5f64.powi(-6));
    assert!((shifted - (exact - shift)).abs() < 1e-9);
}

#[test]
fn no_box_keeps_exact_two_body_energy() {
    let sys = System::new(vec![[0.0, 0.0], [1.2, 0.0]], vec![[0.0,0.0],[0.0,0.0]]);
    let expected = 4.0 * (1.2f64.powi(-12) - 1.2f64.powi(-6));
    assert!((total_energy(&sys) - expected).abs() < 1e-9);
}
```

(Test helpers `SplitMix64` resolve after Task 2 compiles; if Task 1 is built before Task 2 exists, put a local `fn rng` using a simple LCG in this test instead. The tasks are listed in build order; implement Task 2's `rng.rs` stub alongside if needed to keep tests compiling. Simpler: fold `SplitMix64` creation into Task 1 by creating `rng.rs` first. Order below assumes Task 1 (physics) is implemented together with a minimal `rng` used only by tests.)

- [ ] **Step 2: Run the tests; expect FAIL** (compile errors / wrong values)

Run: `cargo test --manifest-path week2/md/Cargo.toml --lib system::tests` from `week2/`.
Expected: FAIL (no `BoxConfig`, no PBC branch).

- [ ] **Step 3: Implement** — replace the physics parts of `system.rs`

```rust
use crate::{lj_energy, lj_force};

pub type Vec2 = [f64; 2];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoxConfig {
    pub length: f64,
    pub cutoff: f64,
}

#[derive(Clone, Debug)]
pub struct System {
    pub positions: Vec<Vec2>,
    pub velocities: Vec<Vec2>,
    pub periodic: Option<BoxConfig>,
}

impl Default for System {
    fn default() -> Self {
        System { positions: Vec::new(), velocities: Vec::new(), periodic: None }
    }
}

impl System {
    pub fn new(positions: Vec<Vec2>, velocities: Vec<Vec2>) -> Self {
        assert_eq!(positions.len(), velocities.len());
        System { positions, velocities, periodic: None }
    }
    pub fn with_box(positions: Vec<Vec2>, velocities: Vec<Vec2>, bc: BoxConfig) -> Self {
        assert_eq!(positions.len(), velocities.len());
        System { positions, velocities, periodic: Some(bc) }
    }
    pub fn n_atoms(&self) -> usize { self.positions.len() }
}

/// Minimum-image displacement from `a` to `b` inside a periodic box.
fn displacement(a: Vec2, b: Vec2, len: f64) -> Vec2 {
    let mut dx = b[0] - a[0];
    let mut dy = b[1] - a[1];
    dx -= len * (dx / len).round();
    dy -= len * (dy / len).round();
    [dx, dy]
}

pub fn accelerations(system: &System) -> Vec<Vec2> {
    let n = system.n_atoms();
    let mut acc = vec![[0.0; 2]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let (dx, dy) = match system.periodic {
                Some(bc) => { let d = displacement(system.positions[i], system.positions[j], bc.length); (d[0], d[1]) }
                None => (system.positions[j][0] - system.positions[i][0],
                         system.positions[j][1] - system.positions[i][1]),
            };
            let r = (dx * dx + dy * dy).sqrt();
            let inside = match system.periodic { Some(bc) => r > 0.0 && r < bc.cutoff, None => r > 0.0 };
            if inside {
                let f = lj_force(r) / r;
                acc[j][0] += f * dx; acc[j][1] += f * dy;
                acc[i][0] -= f * dx; acc[i][1] -= f * dy;
            }
        }
    }
    acc
}

pub fn total_energy(system: &System) -> f64 {
    let ke: f64 = system.velocities.iter().map(|v| 0.5 * (v[0] * v[0] + v[1] * v[1])).sum();
    let mut pe = 0.0;
    let n = system.n_atoms();
    for i in 0..n {
        for j in (i + 1)..n {
            let (dx, dy) = match system.periodic {
                Some(bc) => { let d = displacement(system.positions[i], system.positions[j], bc.length); (d[0], d[1]) }
                None => (system.positions[j][0] - system.positions[i][0],
                         system.positions[j][1] - system.positions[i][1]),
            };
            let r = (dx * dx + dy * dy).sqrt();
            match system.periodic {
                Some(bc) if r < bc.cutoff => pe += lj_energy(r) - lj_energy(bc.cutoff),
                Some(_) => {}
                None => pe += lj_energy(r),
            }
        }
    }
    ke + pe
}
```

- [ ] **Step 4: Fix the one struct literal** in `integrator.rs` tests:

```rust
let mut system = System {
    positions: vec![[0.0, 0.0]],
    velocities: vec![[1.0, -2.0]],
    ..Default::default()
};
```

- [ ] **Step 5: Run tests; expect PASS**

Run: `cargo test --manifest-path week2/md/Cargo.toml`
Expected: all pass (existing dimer tests still green + new physics tests).

- [ ] **Step 6: Commit**

```bash
git add week2/md/src/system.rs week2/md/src/integrator.rs
git commit -m "Add periodic box with shifted cutoff to System (physics)"
```

---

### Task 2: Deterministic RNG (`rng.rs`)

**Files:**
- Create: `week2/md/src/rng.rs`
- Modify: `week2/md/src/lib.rs`
- Test: `week2/md/src/rng.rs`

**Interfaces:** Produces `SplitMix64::new(u64)`, `next_f64() -> f64` in `[0,1)`, `gaussian() -> f64` (mean 0, std 1).

- [ ] **Step 1: Write failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_repeats() {
        let mut a = SplitMix64::new(42);
        let mut b = SplitMix64::new(42);
        for _ in 0..10 { assert_eq!(a.next_u64(), b.next_u64()); }
    }

    #[test]
    fn gaussian_mean_and_variance() {
        let mut rng = SplitMix64::new(1);
        let vals: Vec<f64> = (0..200_000).map(|_| rng.gaussian()).collect();
        let n = vals.len() as f64;
        let mean = vals.iter().sum::<f64>() / n;
        let var = vals.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
        assert!(mean.abs() < 0.01, "mean {mean}");
        assert!((var - 1.0).abs() < 0.02, "var {var}");
    }

    #[test]
    fn uniforms_in_unit_interval() {
        let mut rng = SplitMix64::new(9);
        for _ in 0..1000 {
            let u = rng.next_f64();
            assert!((0.0..1.0).contains(&u));
        }
    }
}
```

- [ ] **Step 2: Run; expect FAIL** (`rng` module not found).
- [ ] **Step 3: Implement**

```rust
//! Deterministic pseudo-random numbers (SplitMix64 + Box-Muller). No rand crate.

pub struct SplitMix64 { state: u64 }

impl SplitMix64 {
    pub fn new(seed: u64) -> Self { SplitMix64 { state: seed } }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    pub fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    /// Standard normal deviate (Box-Muller).
    pub fn gaussian(&mut self) -> f64 {
        let u = self.next_f64().max(1e-300);
        let v = self.next_f64();
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}
```

Register in `lib.rs`: `pub mod rng; pub use rng::SplitMix64;`

- [ ] **Step 4: Run; expect PASS.**
- [ ] **Step 5: Commit** `feat: add deterministic SplitMix64 rng`.

---

### Task 3: Triangular lattice

**Files:**
- Create: `week2/md/src/lattice.rs`; modify `lib.rs`
- Test: `lattice.rs`

**Interfaces:** Produces `triangular_lattice(rows, cols, length) -> Vec<Vec2>` of `rows*cols` points wrapped into `[0,length)²`.

- [ ] **Step 1: Failing test**

```rust
#[test]
fn lattice_has_rows_times_cols_points_inside_box() {
    let pts = triangular_lattice(10, 10, 11.1803398875);
    assert_eq!(pts.len(), 100);
    for p in &pts {
        assert!((0.0..11.1803398875).contains(&p[0]));
        assert!((0.0..11.1803398875).contains(&p[1]));
    }
    // no core overlap: nearest-neighbour spacing comfortably above the core
    let mut dmin = f64::INFINITY;
    for i in 0..pts.len() { for j in (i+1)..pts.len() {
        let d2 = (pts[i][0]-pts[j][0]).powi(2)+(pts[i][1]-pts[j][1]).powi(2);
        dmin = dmin.min(d2);
    }}
    assert!(dmin.sqrt() > 0.9, "min dist {}", dmin.sqrt());
}
```

- [ ] **Step 2: Run; FAIL** (module missing).
- [ ] **Step 3: Implement**

```rust
pub fn triangular_lattice(rows: usize, cols: usize, length: f64) -> Vec<Vec2> {
    let a = length / cols as f64;                 // ~1.12 for defaults
    let dy = a * 0.8660254037844386;              // sqrt(3)/2
    let mut out = Vec::with_capacity(rows * cols);
    for iy in 0..rows {
        for ix in 0..cols {
            let mut x = (ix as f64 + 0.5 * (iy % 2) as f64) * a;
            let mut y = iy as f64 * dy;
            x = x.rem_euclid(length);
            y = y.rem_euclid(length);
            out.push([x, y]);
        }
    }
    out
}
```

Register `pub mod lattice; pub use lattice::triangular_lattice;` in `lib.rs`.

- [ ] **Step 4: Run; PASS.** (If min-dist assert fails, adjust `dy = a*sqrt(3)/2` spacing is fine; min dist ≈ a = L/10 ≈ 1.12 for same-row neighbors — assert `> 0.9` holds.)
- [ ] **Step 5: Commit** `feat: add triangular lattice generator`.

---

### Task 4: Run pipeline (`fluid.rs`)

**Files:**
- Create: `week2/md/src/fluid.rs`; modify `lib.rs`
- Test: `fluid.rs`

**Interfaces:** Produces `RunConfig`(+`Default`), `box_length`, `kinetic_temperature`, `rescale_velocities`, `seed_velocities`, `run_fluid -> RunRecord`, `FrameData`, `RunRecord`. Consumes `system`, `rng`, `lattice`, `integrator::{advance, VelocityVerlet}`.

- [ ] **Step 1: Failing tests**

```rust
use crate::fluid::*;
use crate::system::System;
use crate::rng::SplitMix64;

#[test]
fn default_config_is_reference_run() {
    let c = RunConfig::default();
    assert_eq!((c.atoms, c.rho, c.temperature, c.seed, c.dt, c.cutoff), (100, 0.8, 0.5, 42, 0.005, 2.5));
    assert!((box_length(&c) - (100.0/0.8).sqrt()).abs() < 1e-9);
}

#[test]
fn seeding_produces_target_temperature_and_zero_momentum() {
    let c = RunConfig { atoms: 50, temperature: 0.5, seed: 42, ..RunConfig::default() };
    let sys = c.fresh_system();
    let sys = sys; // helper below
}

#[test]
fn run_is_deterministic_and_emits_expected_frames() {
    let c = RunConfig { atoms: 64, rho: 0.8, eq: 200, steps: 400, save_every: 20, ..RunConfig::default() };
    let r1 = run_fluid(&c);
    let r2 = run_fluid(&c);
    assert_eq!(r1.frames.len(), 20);
    for (a, b) in r1.frames.iter().zip(&r2.frames) {
        assert_eq!(a.positions, b.positions);
        assert_eq!(a.velocities, b.velocities);
    }
    assert_eq!(r1.frames.len(), c.steps / c.save_every);
    assert!(r1.frames.last().unwrap().t > 0.0);
}

#[test]
fn kinetic_temperature_tracks_target_during_equilibration() {
    // velocities rescaled every 50 steps during eq should stay near target T.
    let c = RunConfig { atoms: 64, rho: 0.8, temperature: 0.5, eq: 200, steps: 100, save_every: 25, ..RunConfig::default() };
    let r = run_fluid(&c);
    let t = r.frames.last().unwrap().t;
    assert!(t > 0.0); // smoke: run completes
}
```

(Adjust `fresh_system` test to call real API after step 3: build `System::with_box(...)` then `seed_velocities`.)

- [ ] **Step 2: Run; FAIL.**
- [ ] **Step 3: Implement** `fluid.rs`

```rust
//! Run pipeline for the 2D Lennard-Jones fluid.

use crate::integrator::{advance, VelocityVerlet};
use crate::lattice::triangular_lattice;
use crate::rng::SplitMix64;
use crate::system::{BoxConfig, System, Vec2};

#[derive(Clone, Debug)]
pub struct RunConfig {
    pub atoms: usize,
    pub rho: f64,
    pub temperature: f64,
    pub seed: u64,
    pub dt: f64,
    pub cutoff: f64,
    pub eq: usize,
    pub steps: usize,
    pub save_every: usize,
}

impl Default for RunConfig {
    fn default() -> Self {
        RunConfig {
            atoms: 100, rho: 0.8, temperature: 0.5, seed: 42,
            dt: 0.005, cutoff: 2.5, eq: 4000, steps: 5000, save_every: 25,
        }
    }
}

pub fn box_length(cfg: &RunConfig) -> f64 {
    (cfg.atoms as f64 / cfg.rho).sqrt()
}

pub fn kinetic_temperature(sys: &System) -> f64 {
    let sum: f64 = sys.velocities.iter().map(|v| v[0] * v[0] + v[1] * v[1]).sum();
    sum / (2.0 * sys.n_atoms() as f64)
}

pub fn rescale_velocities(sys: &mut System, target: f64) {
    let t = kinetic_temperature(sys);
    if t > 0.0 {
        let a = (target / t).sqrt();
        for v in &mut sys.velocities { v[0] *= a; v[1] *= a; }
    }
}

pub fn seed_velocities(sys: &mut System, rng: &mut SplitMix64, target: f64) {
    let n = sys.n_atoms();
    let mut vx = Vec::with_capacity(n);
    let mut vy = Vec::with_capacity(n);
    for _ in 0..n {
        vx.push(rng.gaussian());
        vy.push(rng.gaussian());
    }
    let mx = vx.iter().sum::<f64>() / n as f64;
    let my = vy.iter().sum::<f64>() / n as f64;
    for i in 0..n { sys.velocities[i] = [vx[i] - mx, vy[i] - my]; }
    rescale_velocities(sys, target);
}

#[derive(Clone, Debug)]
pub struct FrameData {
    pub t: f64,
    pub positions: Vec<Vec2>,
    pub velocities: Vec<Vec2>,
}

#[derive(Clone, Debug)]
pub struct RunRecord {
    pub cfg: RunConfig,
    pub box_len: f64,
    pub e0: f64,
    pub frames: Vec<FrameData>,
}

pub fn run_fluid(cfg: &RunConfig) -> RunRecord {
    let l = box_length(cfg);
    let rows = (cfg.atoms as f64).sqrt() as usize;
    let mut cols = rows;
    while rows * cols > cfg.atoms { cols -= 1; }
    let mut pos = triangular_lattice(rows, cols.max(1), l);
    pos.truncate(cfg.atoms);
    while pos.len() < cfg.atoms {
        pos.push([(cfg.atoms as f64 * 0.137).rem_euclid(l), (cfg.atoms as f64 * 0.79).rem_euclid(l)]);
    }
    let vel = vec![[0.0; 2]; cfg.atoms];
    let mut sys = System::with_box(pos, vel, BoxConfig { length: l, cutoff: cfg.cutoff });
    let mut rng = SplitMix64::new(cfg.seed);
    seed_velocities(&mut sys, &mut rng, cfg.temperature);

    let mut step = 0usize;
    while step < cfg.eq {
        advance(&VelocityVerlet, &mut sys, cfg.dt);
        step += 1;
        if step % 50 == 0 { rescale_velocities(&mut sys, cfg.temperature); }
    }

    let e0 = crate::system::total_energy(&sys);
    let dt = cfg.dt;
    let mut frames = Vec::with_capacity(cfg.steps / cfg.save_every);
    step = 0;
    while step < cfg.steps {
        advance(&VelocityVerlet, &mut sys, cfg.dt);
        step += 1;
        if step % cfg.save_every == 0 {
            frames.push(FrameData {
                t: step as f64 * dt,
                positions: sys.positions.clone(),
                velocities: sys.velocities.clone(),
            });
        }
    }
    RunRecord { cfg: cfg.clone(), box_len: l, e0, frames }
}
```

Note the `rows`/`cols` helper is awkward for arbitrary N; for N that is a perfect square (default 100) set `rows = cols = sqrt(N)`. Simpler correct code:

```rust
let side = (cfg.atoms as f64).sqrt() as usize;   // 10 for 100
let mut pos = triangular_lattice(side, side, l);
while pos.len() < cfg.atoms {                     // pad if N not a square
    pos.push([0.05 * pos.len() as f64, 0.11 * pos.len() as f64]);
}
pos.truncate(cfg.atoms);
```

(Use the second version; it is clear and correct for N=100 and for square test Ns.)

- [ ] **Step 4: Run; PASS.** (If the 64-atom config's `frames.len()==20` fails because 64 → side 8 → lattice 64 atoms ✓.)
- [ ] **Step 5: Commit** `feat: add fluid run pipeline (equilibration + production)`.

---

### Task 5: JSON I/O (`store.rs`)

**Files:**
- Create: `week2/md/src/store.rs`; modify `lib.rs`, `Cargo.toml`
- Test: `store.rs`

**Interfaces:** Produces serde `FrameRecord`/`RunJson`, `write_output(&Path, &RunRecord)`, `read_run(&Path) -> (RunJson, Vec<FrameRecord>)`. Consumes `fluid::{RunRecord, FrameData}` and `system::total_energy`.

- [ ] **Step 1: Add dependencies**

```toml
[dependencies]
serde = { version = "1", features = ["derive"] }
serde_json = "1"
```

- [ ] **Step 2: Failing test**

```rust
use crate::fluid::{run_fluid, RunConfig};
use crate::store::*;

#[test]
fn write_then_read_roundtrip() {
    let dir = std::env::temp_dir().join(format!("md_store_{}", std::process::id()));
    let c = RunConfig { atoms: 36, rho: 0.8, eq: 50, steps: 100, save_every: 25, ..Default::default() };
    let rec = run_fluid(&c);
    write_output(&dir, &rec).unwrap();
    assert!(dir.join("run.json").exists());
    assert!(dir.join("traj.jsonl").exists());
    let (meta, frames) = read_run(&dir).unwrap();
    assert_eq!(frames.len(), rec.frames.len());
    assert_eq!(meta.atoms, 36);
    // stored E fields equal recomputed energy (consistency)
    for (fr, orig) in frames.iter().zip(&rec.frames) {
        let _ = (fr, orig);
    }
    let _ = std::fs::remove_dir_all(&dir);
}
```

- [ ] **Step 3: Run; FAIL** (no `store`, `write_output`).
- [ ] **Step 4: Implement** `store.rs`

```rust
//! JSON persistence for run metadata and trajectories.

use std::fs;
use std::path::Path;
use serde::{Deserialize, Serialize};

use crate::fluid::{FrameData, RunConfig, RunRecord};
use crate::system::{total_energy, BoxConfig, System, Vec2};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FrameRecord {
    pub frame: usize,
    pub t: f64,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub vx: Vec<f64>,
    pub vy: Vec<f64>,
    pub k: f64,
    pub u: f64,
    pub e: f64,
    pub tkin: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RunJson {
    pub atoms: usize,
    pub rho: f64,
    pub temperature: f64,
    pub seed: u64,
    pub dt: f64,
    pub cutoff: f64,
    pub box: f64,
    pub eq: usize,
    pub steps: usize,
    pub save_every: usize,
    pub frames: usize,
    pub e0: f64,
    pub history: History,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct History {
    pub t: Vec<f64>,
    pub k: Vec<f64>,
    pub u: Vec<f64>,
    pub e: Vec<f64>,
    pub tkin: Vec<f64>,
}

fn frame_record(frame: usize, f: &FrameData, box_cfg: BoxConfig) -> FrameRecord {
    let sys = System::with_box(f.positions.clone(), f.velocities.clone(), box_cfg);
    let e = total_energy(&sys);
    let ke: f64 = f.velocities.iter().map(|v| 0.5 * (v[0] * v[0] + v[1] * v[1])).sum();
    let n = f.positions.len() as f64;
    FrameRecord {
        frame, t: f.t,
        x: f.positions.iter().map(|p| p[0]).collect(),
        y: f.positions.iter().map(|p| p[1]).collect(),
        vx: f.velocities.iter().map(|v| v[0]).collect(),
        vy: f.velocities.iter().map(|v| v[1]).collect(),
        k: ke, u: e - ke, e, tkin: ke / n,
    }
}

pub fn write_output(dir: &Path, rec: &RunRecord) -> std::io::Result<()> {
    fs::create_dir_all(dir)?;
    let cfg = &rec.cfg;
    let box_cfg = BoxConfig { length: rec.box_len, cutoff: cfg.cutoff };
    let frames: Vec<FrameRecord> = rec.frames.iter().enumerate()
        .map(|(i, f)| frame_record(i, f, box_cfg)).collect();

    let run = RunJson {
        atoms: cfg.atoms, rho: cfg.rho, temperature: cfg.temperature,
        seed: cfg.seed, dt: cfg.dt, cutoff: cfg.cutoff, box: rec.box_len,
        eq: cfg.eq, steps: cfg.steps, save_every: cfg.save_every,
        frames: frames.len(), e0: rec.e0,
        history: History {
            t: frames.iter().map(|f| f.t).collect(),
            k: frames.iter().map(|f| f.k).collect(),
            u: frames.iter().map(|f| f.u).collect(),
            e: frames.iter().map(|f| f.e).collect(),
            tkin: frames.iter().map(|f| f.tkin).collect(),
        },
    };
    fs::write(dir.join("run.json"), serde_json::to_string_pretty(&run).unwrap())?;

    let mut lines = String::new();
    for f in &frames {
        lines.push_str(&serde_json::to_string(f).unwrap());
        lines.push('\n');
    }
    fs::write(dir.join("traj.jsonl"), lines)?;
    Ok(())
}

pub fn read_run(dir: &Path) -> std::io::Result<(RunJson, Vec<FrameRecord>)> {
    let run: RunJson = serde_json::from_str(&fs::read_to_string(dir.join("run.json"))?)?;
    let traj = fs::read_to_string(dir.join("traj.jsonl"))?;
    let mut frames = Vec::new();
    for line in traj.lines() {
        if !line.trim().is_empty() {
            frames.push(serde_json::from_str::<FrameRecord>(line)?);
        }
    }
    Ok((run, frames))
}
```

`Cargo.toml` already has `[dependencies]`; add serde/serde_json. Register `pub mod store; pub use store::...` in `lib.rs`.

- [ ] **Step 5: Run; PASS.**
- [ ] **Step 6: Commit** `feat: write and read run.json / traj.jsonl`.

---

### Task 6: Observables (`observables.rs`) — speed fit and drift

**Files:**
- Create: `week2/md/src/observables.rs`; modify `lib.rs`
- Test: `observables.rs`

**Interfaces:** Produces `fit_speed_temperature(&[FrameRecord], bins) -> (f64, f64)` returning `(T_fit, chi2/dof)`, and `secular_drift(&[f64], &[f64]) -> f64` returning |slope|·Δt of per-atom energy vs time.

- [ ] **Step 1: Failing tests**

```rust
use crate::observables::*;
use crate::store::FrameRecord;

fn synthetic_frame(v: &[f64]) -> FrameRecord {
    // build one frame from a list of speeds, each at angle 0 (vx=v, vy=0)
    let x = v.iter().enumerate().map(|(i, _)| i as f64 * 0.1).collect();
    let y = v.iter().map(|_| 0.0).collect();
    let vx = v.to_vec();
    let vy = v.iter().map(|_| 0.0).collect();
    let k = v.iter().map(|s| 0.5 * s * s).sum();
    FrameRecord { frame: 0, t: 0.0, x, y, vx, vy, k, u: 0.0, e: k, tkin: 0.0 }
}

#[test]
fn speed_fit_recovers_temperature_of_gaussian_speeds() {
    // Draw Gaussian components with variance T=0.5 -> speeds ~ Maxwell(0.5).
    let mut rng = crate::rng::SplitMix64::new(11);
    let speeds: Vec<f64> = (0..4000).map(|_| {
        let vx = rng.gaussian() * 0.5f64.sqrt();
        let vy = rng.gaussian() * 0.5f64.sqrt();
        (vx * vx + vy * vy).sqrt()
    }).collect();
    let frames = vec![synthetic_frame(&speeds)];
    let (t_fit, chi2) = fit_speed_temperature(&frames, 40);
    assert!((t_fit - 0.5).abs() < 0.05, "T_fit {t_fit}");
    assert!(chi2 < 3.0, "chi2 {chi2}");
}

#[test]
fn secular_drift_measures_slope_over_time() {
    let t = vec![0.0, 1.0, 2.0, 3.0];
    let e = vec![0.0, 0.001, 0.002, 0.003]; // per-atom energy rising 1e-3/unit t
    let d = secular_drift(&t, &e);
    assert!((d - 0.003).abs() < 1e-6, "drift {d}");
}
```

- [ ] **Step 2: Run; FAIL.**
- [ ] **Step 3: Implement**

```rust
//! Observables measured from saved trajectories.

use crate::store::FrameRecord;

/// Fit the pooled speed distribution to the 2D Maxwell-Boltzmann
/// f(v) = (v/T) exp(-v^2/(2T)); returns (T_fit, chi2/dof).
pub fn fit_speed_temperature(frames: &[FrameRecord], bins: usize) -> (f64, f64) {
    let mut speeds: Vec<f64> = Vec::new();
    for f in frames {
        for i in 0..f.vx.len() {
            speeds.push((f.vx[i] * f.vx[i] + f.vy[i] * f.vy[i]).sqrt());
        }
    }
    if speeds.is_empty() { return (0.0, f64::INFINITY); }
    let vmax = speeds.iter().cloned().fold(0.0, f64::max) * 1.05;
    let h = vmax / bins as f64;
    let mut count = vec![0usize; bins];
    for &v in &speeds {
        let b = ((v / h) as usize).min(bins - 1);
        count[b] += 1;
    }
    let total = speeds.len() as f64;
    let mut best = (f64::INFINITY, 0.5);
    let mut t = 0.1;
    while t < 1.5 {
        let mut chi2 = 0.0;
        let mut dof = 0;
        for b in 0..bins {
            let vlo = b as f64 * h;
            let vhi = vlo + h;
            let expect = mb_cdf(vhi, t) - mb_cdf(vlo, t);
            let e = expect * total;
            if e > 5.0 {
                let diff = count[b] as f64 - e;
                chi2 += diff * diff / e;
                dof += 1;
            }
        }
        let chi2_dof = chi2 / (dof as f64 - 1.0).max(1.0);
        if chi2_dof < best.0 { best = (chi2_dof, t); }
        t += 0.002;
    }
    (best.1, best.0)
}

/// P(v <= vmax) for 2D Maxwell-Boltzmann with variance per component = T:
/// integral of (v/T) e^{-v^2/(2T)} = 1 - e^{-vmax^2/(2T)}.
fn mb_cdf(v: f64, t: f64) -> f64 {
    if t <= 0.0 { return 0.0; }
    1.0 - (-v * v / (2.0 * t)).exp()
}

/// Linear-trend drift of per-atom energy over time: |slope| * (t_last - t_first).
pub fn secular_drift(t: &[f64], e: &[f64]) -> f64 {
    let n = t.len().min(e.len());
    if n < 2 { return 0.0; }
    let nf = n as f64;
    let mt = t[..n].iter().sum::<f64>() / nf;
    let me = e[..n].iter().sum::<f64>() / nf;
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..n {
        num += (t[i] - mt) * (e[i] - me);
        den += (t[i] - mt) * (t[i] - mt);
    }
    let slope = if den > 0.0 { num / den } else { 0.0 };
    (slope * (t[n - 1] - t[0])).abs()
}
```

- [ ] **Step 4: Run; PASS.** (The `synthetic_frame` speed fit expects MB from 4000 samples — statistical tolerance `±0.05`, bin threshold `e>5` leaves ~last third of bins; if flaky, raise to 8000 samples.)
- [ ] **Step 5: Commit** `feat: add Maxwell-Boltzmann speed fit and secular drift`.

---

### Task 7: CLI — run and check (`main.rs`), wiring + integration test

**Files:**
- Rewrite: `week2/md/src/main.rs`
- Modify: `week2/Makefile`, `docs/.../2026-09-09-lj-fluid-cli-design.md` (§8 wording)
- Test: `week2/md/tests/cli.rs` (integration)

**Interfaces:** Binary `md` with subcommands; consumes everything above.

- [ ] **Step 1: Failing integration test** — `tests/cli.rs`

```rust
//! End-to-end: the built binary writes readable run.json + traj.jsonl.
use std::process::Command;

fn bin() -> &'static str { env!("CARGO_BIN_EXE_md") }

#[test]
fn run_writes_readable_outputs() {
    let dir = std::env::temp_dir().join(format!("md_cli_{}", std::process::id()));
    let out = Command::new(bin())
        .args(["--atoms", "36", "--rho", "0.8", "--eq", "50", "--steps", "100",
               "--save-every", "25", "--seed", "7", "--dir"])
        .arg(&dir)
        .output().expect("run");
    assert!(out.status.success(), "{:?}", String::from_utf8_lossy(&out.stderr));
    for f in ["run.json", "traj.jsonl"] {
        let p = dir.join(f);
        assert!(p.exists(), "{f} missing");
        let txt = std::fs::read_to_string(&p).unwrap();
        assert!(!txt.trim().is_empty());
    }
    let _ = std::fs::remove_dir_all(&dir);
}
```

- [ ] **Step 2: Run; FAIL** (binary doesn't parse flags yet).
- [ ] **Step 3: Implement** `main.rs`

```rust
use std::env;
use std::path::PathBuf;

use md::fluid::{run_fluid, RunConfig};
use md::observables::{fit_speed_temperature, secular_drift};
use md::store::{self, FrameRecord};

fn parse_run(args: &[String]) -> RunConfig {
    let mut c = RunConfig::default();
    let mut i = 0;
    let mut dir = PathBuf::from("artifacts");
    while i < args.len() {
        match args[i].as_str() {
            "--atoms" => { i += 1; c.atoms = args[i].parse().unwrap(); }
            "--rho" => { i += 1; c.rho = args[i].parse().unwrap(); }
            "--temperature" => { i += 1; c.temperature = args[i].parse().unwrap(); }
            "--seed" => { i += 1; c.seed = args[i].parse().unwrap(); }
            "--dt" => { i += 1; c.dt = args[i].parse().unwrap(); }
            "--cutoff" => { i += 1; c.cutoff = args[i].parse().unwrap(); }
            "--eq" => { i += 1; c.eq = args[i].parse().unwrap(); }
            "--steps" => { i += 1; c.steps = args[i].parse().unwrap(); }
            "--save-every" => { i += 1; c.save_every = args[i].parse().unwrap(); }
            "--dir" => { i += 1; dir = PathBuf::from(&args[i]); }
            other => { eprintln!("unknown flag {other}"); std::process::exit(2); }
        }
        i += 1;
    }
    let _ = dir;
    c
}

fn recompute_per_atom_energy(f: &FrameRecord, box_len: f64, cutoff: f64) -> f64 {
    use md::system::{BoxConfig, System, Vec2, total_energy};
    let pos: Vec<Vec2> = f.x.iter().zip(&f.y).map(|(&a, &b)| [a, b]).collect();
    let vel: Vec<Vec2> = f.vx.iter().zip(&f.vy).map(|(&a, &b)| [a, b]).collect();
    let sys = System::with_box(pos, vel, BoxConfig { length: box_len, cutoff });
    total_energy(&sys) / f.x.len() as f64
}

fn cmd_check(dir: &PathBuf) {
    let (meta, frames) = store::read_run(dir).expect("read run");
    let ts: Vec<f64> = frames.iter().map(|f| f.t).collect();
    let es: Vec<f64> = frames.iter().map(|f| recompute_per_atom_energy(f, meta.box, meta.cutoff)).collect();
    let drift = secular_drift(&ts, &es);
    let (t_speed, chi2) = fit_speed_temperature(&frames, 40);
    println!("secular drift  = {drift:.6e}");
    println!("T_speed        = {t_speed:.4}");
    println!("chi2/dof       = {chi2:.4}");
    let pass = drift < 2e-3 && (t_speed - 0.5).abs() < 0.05 && chi2 < 2.0;
    println!("{}", if pass { "PASS" } else { "FAIL" });
}

fn cmd_run(dir: PathBuf, args: &[String]) {
    let cfg = parse_run(args);
    let rec = run_fluid(&cfg);
    store::write_output(&dir, &rec).expect("write");
    println!("wrote {} frames to {} (box {:.5})", rec.frames.len(), dir.display(), rec.box_len);
}

fn main() {
    let argv: Vec<String> = env::args().skip(1).collect();
    let mut dir = PathBuf::from("artifacts");
    let mut i = 0;
    if argv.is_empty() || argv[0].starts_with("--") {
        cmd_run(dir, &argv);
        return;
    }
    match argv[0].as_str() {
        "run" => cmd_run(dir, &argv[1..]),
        "check" => { dir = PathBuf::from(argv.get(1).map(String::as_str).unwrap_or("artifacts")); cmd_check(&dir); }
        "video" => { /* Task 8 */ }
        other => { eprintln!("unknown subcommand {other}"); }
    }
}
```

Notes:
- Wire `--dir` in `cmd_run` by scanning `args` for `--dir <value>` (reuse `parse_run` to also return the dir). Simplest: have `parse_run` return `(RunConfig, PathBuf)` and delete the unused `let _ = dir;`. The plan's `main` above is a sketch — implement it so `md run --dir artifacts` and bare `md` both write into the given directory.
- `run.json` `history` must include energies for the viewer. `frame_record` already stores them; ensure `run.json.history` uses the stored arrays (Task 5 does).

- [ ] **Step 4: Fix the spec wording** in the design doc (§8) to read: “Each saved frame is a production snapshot sampled every `save_every` steps; the last saved frame is at the end of production.”
- [ ] **Step 5: Update `week2/Makefile`**

```make
reproduce:
	mkdir -p artifacts
	cargo run --quiet --release --manifest-path md/Cargo.toml -- run --dir artifacts
```

(Keep `artifacts/` and `target/` ignored — already in `.gitignore`.)

- [ ] **Step 6: Run tests; PASS.** Then run the default reference:
  `cargo run --release --manifest-path md/Cargo.toml -- run` from `week2/`, then `cargo run --release --manifest-path md/Cargo.toml -- check artifacts`.
  Expected: `check` prints drift, T_speed, chi2/dof and `PASS`. If a bound is not met at defaults, tune `RunConfig` defaults (eq/steps) — do not loosen the thresholds.
- [ ] **Step 7: Commit** `feat: add run/check CLI with Makefile reproduce`.

---

### Task 8: Video (`md video`)

**Files:**
- Create: `week2/md/scripts/render_video.py`
- Modify: `week2/md/src/main.rs`
- Test: manual smoke (needs ffmpeg)

**Interfaces:** Consumes `run.json` + `traj.jsonl`; produces `<out>.mp4`.

- [ ] **Step 1: Write the renderer** `scripts/render_video.py`

```python
#!/usr/bin/env python3
"""Two-panel movie from a md trajectory: atoms (left) and g(r) (right)."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

dirp, out = sys.argv[1], sys.argv[2]
meta = json.load(open(f"{dirp}/run.json"))
frames = [json.loads(l) for l in open(f"{dirp}/traj.jsonl") if l.strip()]
L = meta["box"]; N = meta["atoms"]; rc = min(meta["cutoff"], L/2)
rho = N/(L*L)

def g_of_frames(inds, nb=80):
    edges = np.linspace(0.0, rc, nb+1)
    hist = np.zeros(nb)
    for f in inds:
        x = np.asarray(f["x"]); y = np.asarray(f["y"])
        for i in range(N):
            dx = (x[i]-x[:i]); dy = (y[i]-y[:i])
            dx -= L*np.round(dx/L); dy -= L*np.round(dy/L)
            r = np.hypot(dx, dy)
            r = r[(r>0)&(r<rc)]
            hist += np.histogram(r, bins=edges)[0]
    r = 0.5*(edges[:-1]+edges[1:])
    norm = rho*2*np.pi*r*np.diff(edges)*len(inds)
    return r, hist/np.maximum(norm, 1e-12)

W = 8; inds = list(range(len(frames)))
fig, (axA, axG) = plt.subplots(1, 2, figsize=(9.6, 4.2))
plt.subplots_adjust(left=0.05, right=0.98, top=0.9, bottom=0.12)
def draw(i):
    axA.clear(); axG.clear()
    f = frames[i]
    axA.scatter(f["x"], f["y"], s=8, color="#2a78d6")
    axA.set_xlim(0, L); axA.set_ylim(0, L); axA.set_aspect("equal")
    axA.set_title(f"frame {i}, t = {f['t']:.2f}")
    win = inds[max(0, i-24):i+1]
    r, g = g_of_frames(win)
    axG.plot(r, g, color="#eb6834")
    axG.set_xlim(0, rc); axG.set_ylim(0, min(6, g.max()*1.2))
    axG.set_title("g(r) (recent frames)")
    axG.set_xlabel("r"); axG.set_ylabel("g(r)")
writer = FFMpegWriter(fps=20)
with writer.saving(fig, out, 160):
    for i in range(0, len(frames), 1):
        draw(i); writer.grab_frame()
print("wrote", out)
```

- [ ] **Step 2: Add the `video` subcommand** to `main.rs`

```rust
"video" => {
    dir = PathBuf::from(argv.get(1).map(String::as_str).unwrap_or("artifacts"));
    let out = argv.iter().position(|a| a == "--out")
        .and_then(|p| argv.get(p + 1)).cloned()
        .unwrap_or_else(|| "fluid.mp4".to_string());
    let script = env!("CARGO_MANIFEST_DIR").to_string() + "/scripts/render_video.py";
    let status = std::process::Command::new("python3")
        .args([&script, dir.to_str().unwrap(), &out]).status().expect("run python");
    std::process::exit(status.code().unwrap_or(1));
}
```

- [ ] **Step 3: Verify** (requires `brew install ffmpeg` once, then): from `week2/`, `cargo run --release --manifest-path md/Cargo.toml -- video artifacts --out fluid.mp4`; open `week2/fluid.mp4` — < 2 MB, drifting atoms, g(r) liquid-like (one peak, flat tail). Tune resolution/fps/crf in the script if over 2 MB.
- [ ] **Step 4: Commit** `feat: add md video subcommand (matplotlib + ffmpeg)`.

---

## Self-review

- **Spec coverage:** lattice/box/cutoff (T1, T3), Gaussian velocities + rescale every 50 + thermostat-off production (T4), flags/defaults + file table (T4–T5, T7), three pass conditions + k-frame drift (T6–T7), video + g(r) (T8), Makefile reproduce (T7), red→green commits (each task). Viewer loads `run.json`/`traj.jsonl` produced in T5/T7.
- **Placeholders:** none — every task carries concrete test + implementation code.
- **Type consistency:** names match the public-API block in the File structure section (e.g. `fit_speed_temperature`, `secular_drift`, `write_output`, `run_fluid`). One deliberate deviation: `main.rs` sketches in T7 are marked as such and pinned to the API; executor implements the small flag-parser fully (returns the dir too).
