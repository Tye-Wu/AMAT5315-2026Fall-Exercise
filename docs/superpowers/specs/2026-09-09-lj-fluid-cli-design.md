# Design: `md` — Lennard-Jones fluid CLI

Date: 2026-09-09
Scope: `week2/md/` (the `md` crate)
Status: approved design, pre-plan

## 1. Goal

Turn the `md` crate into a small CLI tool for molecular dynamics of a
two-dimensional Lennard-Jones fluid, in reduced units (ε = σ = 1, m = 1):

1. **run** — simulate the fluid and save the trajectory;
2. **check** — verify the physics of a saved trajectory;
3. **video** — render the motion together with the evolving radial
   distribution function g(r).

Plus a `make reproduce` rule (already in `week2/Makefile`, to be wired to the
real CLI) so the whole thing is reproducible in release mode.

## 2. CLI contract (subcommands, flags, defaults)

Binary target: `md` (the crate's sole binary).

```
md                         # default: same as `run`
md run [flags]             # simulate and write run.json + traj.jsonl
md check <dir>             # read a saved run, print 3 measured values + PASS/FAIL
md video <dir> [--out f]   # render atoms + g(r) movie to f (default fluid.mp4)
```

Default flags (the reference run):

| flag | default | meaning |
|---|---|---|
| `--atoms` | 100 | N, from the 10×10 triangular lattice |
| `--rho` | 0.8 | number density (area density) |
| `--temperature` | 0.5 | target T (units of ε/k_B) |
| `--seed` | 42 | RNG seed (reference value) |
| `--dt` | 0.005 | time step |
| `--cutoff` | 2.5 | shifted cutoff r_c |
| `--eq` | 4000 | equilibration steps (rescale every 50) |
| `--steps` | 5000 | production steps (thermostat off) |
| `--save-every` | 25 | record a frame every 25 production steps → 200 frames |
| `--dir` | `artifacts` | output directory (created if missing) |

Box side L = √(N/ρ). For defaults: N=100, ρ=0.8 → L = √125 ≈ 11.18034.
All runs are deterministic for a given seed (own RNG, no external crate).

## 3. Physics

Reduced units: energy in ε, length in σ, mass 1, so force = acceleration and
temperature T is in units of ε/k_B.

**Pair interaction** (existing `lj_energy`, `lj_force`):
V(r) = 4 (r⁻¹² − r⁻⁶), F(r) = −V'(r) = 4 (12 r⁻¹³ − 6 r⁻⁷), positive outward.

**Periodic box, minimum image.** Two atoms interact through their closest
periodic image: displacement d = wrap(p_j − p_i) into (−L/2, L/2].

**Cut-and-shifted cutoff at r_c = 2.5.** Inside the cutoff the pair energy is
V_shift(r) = V(r) − V(r_c); at r ≥ r_c it is 0. This makes the potential
continuous at r_c (value 0) — the point the cutoff continuity test probes just
inside r_c, not only at r_c. Force inside the cutoff is −dV_shift/dr = F(r)
(the constant shift contributes nothing); at/outside it is 0. A small force
discontinuity of size |F(r_c)| remains; it is negligible at r_c = 2.5 and is
not part of any acceptance check.

**Total energy**: E = KE + U_shift, KE = ½ Σ_i |v_i|² (m = 1),
U_shift = Σ_{i<j} V_shift(r_ij). Kinetic temperature (d = 2):
T_kin = (1/(d N)) Σ |v_i|² = ⟨v²⟩/2.

## 4. Data model — extending `System`

`System` gains an optional periodic box while keeping the existing
dimer/lesson behaviour (box = `None`) intact:

```rust
pub struct System {
    pub positions:  Vec<Vec2>,
    pub velocities: Vec<Vec2>,
    pub periodic:   Option<BoxConfig>,   // new; None preserves old behaviour
}
pub struct BoxConfig { pub length: f64, pub cutoff: f64 }
```

`System::new(positions, velocities)` sets `periodic: None` (dimer path,
existing tests unchanged except one struct literal gains `..Default::default()`).
`System::with_box(positions, velocities, length, cutoff)` enables periodicity.

`accelerations` and `total_energy` branch on `periodic`:
- `None`: current all-pairs, no cutoff (unchanged);
- `Some`: minimum-image pairs within r_c, shifted potential as in §3.

Because the change lives inside these two functions, the shared
`Integrator` trait and `Euler`/`VelocityVerlet`/`advance` machinery (§5) work
for the fluid unchanged. Thermostat/rescale logic stays in the run loop.

## 5. Integrators (reuse)

Keep `Integrator { fn step(&self, &mut System, dt) }`, `advance`, `Euler`,
`VelocityVerlet` exactly as implemented for the dimer. The fluid run uses
`VelocityVerlet` only.

## 6. Initialization

- **Lattice**: 10×10 triangular lattice → N = 100. Positions generated from
  basis vectors a₁ = a(1, 0), a₂ = a(½, √3/2) over ix, iy ∈ 0..10, spacing a
  chosen so the cluster is wrapped into the box of side L at ρ = 0.8.
  Coordinates are wrapped into [0, L).
- **Velocities**: deterministic RNG (SplitMix64 for uniform u64; Box–Muller
  for Gaussian) seeded with `seed`. Draw N×2 Gaussian deviates, subtract the
  centre-of-mass velocity, then rescale so the kinetic temperature equals T
  (per-component variance T).

## 7. Run pipeline

```
build lattice System (periodic = Some(L, r_c))
seed velocities at T
for eq in 0..=eq:                      # equilibration
    advance(&VelocityVerlet, sys, dt)
    if eq steps done % 50 == 0: rescale velocities to T
# production, thermostat off
frame f:  for each saved frame: snapshot {t, positions, velocities, K, U, E, T}
produce steps in blocks of `save-every`; after each block record one frame
write run.json and traj.jsonl (see §8)
```

Rescale = multiply every velocity by α so ⟨v²⟩/2 → T after the 50-step block.

## 8. Output files

### `traj.jsonl` — one JSON object per saved frame
```
{"frame":0,"t":0.0,"x":[...100],"y":[...100],"vx":[...100],"vy":[...100],
 "K":...,"U":...,"E":...,"T":...}
```
Frame 0 is the state at the start of production; 200 frames total by default.
`K, U, E` are the stored energy fields for this snapshot.

### `run.json`
```json
{
  "atoms":100,"rho":0.8,"temperature":0.5,"seed":42,"dt":0.005,
  "cutoff":2.5,"box":11.1803398875,"eq":4000,"steps":5000,
  "save_every":25,"frames":200,
  "E0": -1.23e2,
  "history": {"t":[...],"K":[...],"U":[...],"E":[...],"T":[...]}
}
```
`history` holds one entry per saved frame (the full run). `E0` is the total
energy of the initial state.

## 9. `md check <dir>`

1. Read `run.json` and `traj.jsonl`.
2. **Consistency**: for each saved frame recompute E from the stored
   positions/velocities (using the same shifted potential) and compare with the
   stored `E` field; report the max deviation.
3. **Three measured values** (from saved production frames):
   - **secular drift** ΔE_drift: ordinary least-squares slope m of per-atom
     energy e_f = E_f/N versus t over the saved frames; report m·(t_last − t_first).
     PASS if < 2e-3.
   - **T_speed**: fit the pooled speed distribution of all atoms over the saved
     frames to the 2D Maxwell–Boltzmann f(v) = (v/T) e^(−v²/(2T)); report fitted T.
     PASS if |T_speed − 0.5| < 0.05.
   - **χ²/dof**: goodness-of-fit of that speed fit (binned histogram,
     expected counts from the fitted f(v)). PASS if < 2.
4. Print the three values and `PASS` (or `FAIL` listing the failed bound).

## 10. `md video <dir> --out <file>`

Render `fluid.mp4` (< 2 MB) with two panels: atoms moving in the periodic box
(left) and an evolving g(r) (right).

- **g(r)**: average over the most recent W saved frames (rolling window), bin
  histogram of minimum-image distances (r < L/2), normalised so g(r) → 1;
  liquid shape = one sharp first peak, flat tail.
- Encoding path: the crate writes the frames; a small committed Python renderer
  (matplotlib, as with `week2/plot_*.py`) draws both panels per frame and the
  `video` subcommand shells out to it, which encodes with ffmpeg.
- Size control: ~200 frames, modest resolution (e.g. 640×360), crf/quality set
  so the output is well under 2 MB.
- Runtime requirement: python3 + matplotlib + ffmpeg on PATH. (No ffmpeg is
  installed on the dev machine; installing via Homebrew is part of verification.)

## 11. `make reproduce`

Already present in `week2/Makefile`. After the CLI exists, `make reproduce`
(from `week2/`) runs the crate in release with default flags writing into
`week2/artifacts/`:

```
mkdir -p artifacts
cargo run --quiet --release --manifest-path md/Cargo.toml -- run --dir artifacts
```

Generated data (`week2/artifacts/`) and build files (`md/target/`) stay out of
git (`.gitignore`).

## 12. Testing plan (red → green)

Unit tests:
- net force over all atoms ≈ 0 for a random PBC configuration (tolerance);
- potential continuity at the cutoff — compare V_shift just inside r_c
  (r = r_c − 10⁻⁶) with the value at r_c (0);
- minimum-image distance returns the nearest image;
- shifted potential equals the exact potential for r well below r_c;
- lattice has N = 100 atoms at ρ = 0.8 in the box.

Integration test (runs the built binary):
- `md run` with tiny defaults into a temp dir produces readable `run.json` and
  `traj.jsonl` (parse + field checks).

Commit sequence: red (tests written, failing / not yet implemented), then green
(implementation), committed in that order, per feature slice.

## 13. Dependencies and system changes

- Cargo deps added: `serde` (derive), `serde_json`. No RNG or CLI-parsing
  crate (own deterministic RNG; small hand-rolled flag parser).
- `System` gains an optional `periodic` field (one existing struct literal is
  updated with `..Default::default()`).
- Verification of `video` needs `ffmpeg` (installed locally via Homebrew).

## 14. Acceptance

`cargo test --manifest-path week2/md/Cargo.toml --release` — all pass.
`make reproduce` (week2) — default run writes artifacts.
`md check artifacts` — prints the three measured values satisfying
drift < 2e-3, |T_speed − 0.5| < 0.05, χ²/dof < 2, then `PASS`.
`md video artifacts --out fluid.mp4` — < 2 MB, drifting atoms, liquid g(r).
`traj.jsonl`/`run.json` load in the supplied viewer (200 saved frames).
