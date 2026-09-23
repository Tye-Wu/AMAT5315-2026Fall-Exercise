# Week 4: Continuum fluid dynamics

This crate implements the periodic advection-diffusion line and two-dimensional
incompressible vorticity equation. The spatial grid is periodic, derivatives
are Fourier pseudospectral, nonlinear products use the two-thirds mask, and
forward Euler, explicit midpoint, and classical RK4 share one `Integrator`
trait. `field` writes an initial or exact velocity field as JSON; `fluid` reads
that JSON from standard input and writes energy, enstrophy, run metadata, and
six-decimal snapshots.

## Regenerate from a clean clone

Run these commands from the exercise repository root. The Rust dependencies
are locked in `Cargo.lock`. The plotting scripts use NumPy and Matplotlib; the
course Python environment already has both, or install them with
`python3 -m pip install numpy matplotlib`.

```sh
cd week4
cargo install --path . --quiet
cd scripts
python3 line_studies.py
python3 derivative_compare.py
python3 taylor_green.py
python3 part3_stability.py
python3 sensitivity.py
python3 order_refinement.py
```

The plot and comparison scripts are run from `week4/scripts/`. They resolve
the crate and output paths from their own location. `cargo install` places
`field` and `fluid` in `~/.cargo/bin`; make sure that directory is on `PATH`.

## Command pipelines and values

The Taylor-Green verification pipeline is:

```sh
field taylor-green --n 64 | fluid --method rk4 --nu 0.1 --dt 0.01 \
  --t-end 1 --every 0.1 --out artifacts/taylor-green \
  > artifacts/taylor-green.tsv
```

The seeded random-flow pipeline is:

```sh
field random --n 128 --seed 2026 --k-min 2 --k-max 6 \
  | fluid --method rk4 --nu 0.004 --dt 0.01 --t-end 10 --every 0.1 \
      --out artifacts/random > artifacts/random.tsv
```

The scripts run these commands with the scan, sensitivity, and refinement
values from the learning sheet. Random phases use a deterministic SplitMix64
hash keyed by the seed and integer wavevector, so they do not depend on grid
size or iteration order. This valid phase realization gives `Umax = 2.5575`
and the RK4 advective estimate `dt <= 0.01863`; the measured RK4 bracket is
`0.030` to `0.032`.

## Scripts and generated files

Run the scripts above in order. Every run, comparison, and plot helper is kept
under `scripts/`; all simulation data is written below the ignored
`artifacts/` directory.

| Script | Files it writes |
| --- | --- |
| `scripts/line_studies.py`, `scripts/line_study.rs` | `evidence/line-stability.png`, `evidence/line-accuracy.png`; intermediate `artifacts/line-study/line-study.json` |
| `scripts/derivative_compare.py`, `scripts/derivative_compare.rs` | Prints the full-precision finite-difference and Fourier derivative error table; no file |
| `scripts/common.py` | Shared CLI and output-path helpers for the other scripts; no direct output |
| `scripts/taylor_green.py` | `artifacts/taylor-green.tsv`, `artifacts/taylor-green/`, `evidence/taylor-green.png` |
| `scripts/part3_stability.py` | `artifacts/random.tsv`, `artifacts/random/`, `artifacts/unstable/`, `artifacts/scan/`, `artifacts/scan-summary.json`, `evidence/random.png` |
| `scripts/sensitivity.py` | `artifacts/sensitivity/`, `artifacts/sensitivity-summary.json`, `evidence/sensitivity.png`, `evidence/blowup.png` |
| `scripts/order_refinement.py` | `artifacts/order/`, `artifacts/convergence/`, `evidence/order.png`, `evidence/convergence.json`, `evidence/convergence.png` |

The committed evidence files and the command that produces each one are:

| File in `evidence/` | Producing command |
| --- | --- |
| `line-stability.png` | `(cd week4/scripts && python3 line_studies.py)` |
| `line-accuracy.png` | `(cd week4/scripts && python3 line_studies.py)` |
| `taylor-green.png` | `(cd week4/scripts && python3 taylor_green.py)` |
| `random.png` | `(cd week4/scripts && python3 part3_stability.py)` |
| `sensitivity.png` | `(cd week4/scripts && python3 sensitivity.py)` |
| `blowup.png` | `(cd week4/scripts && python3 sensitivity.py)` |
| `order.png` | `(cd week4/scripts && python3 order_refinement.py)` |
| `convergence.json` | `(cd week4/scripts && python3 order_refinement.py)` |
| `convergence.png` | `(cd week4/scripts && python3 order_refinement.py)` |

## Verification results

- On the line, measured fitted error slopes are `1.0327` (Euler), `2.0055`
  (midpoint), `4.0040` (RK4), and `2.0025` (equal-weight four-stage rule).
  The measured RK4 stability boundary is near `dt = 0.0494`.
- For `g = sin(3x) cos(2y)`, the N=32 Fourier maximum errors are between
  `1.4e-14` and `1.7e-13`; centered-difference errors fall by factors
  `3.90` to `3.97` when N doubles.
- Taylor-Green gives `E(1) = 0.167580`, `Z(1) = 0.335160`, and relative
  velocity error `7.0386e-7`.
- The random flow starts with `E = 0.500000`, `Z = 6.634685`, and reaches
  `E = 0.298190`, `Z = 0.985143` at t=10. The 0.032 run fails while the
  0.030 run reaches t=10; the 0.01 Euler run fails before t=2.
- Both sensitivity runs reproduce the requested initial vorticity ripple
  after six-decimal field storage (maximum absolute discrepancy below
  `1.0e-6`). The Taylor-Green perturbation decays to `0.1291` of its initial
  velocity difference by t=20; the seeded random-flow perturbation grows by
  `39.67x`.
- The Taylor-Green RK4 refinement slope is `4.1044`. The random-flow
  self-convergence slope is `4.0422`. With this seeded phase realization,
  Richardson predicts `6.8703e-6` at dt=0.0125, above the `5e-6` threshold,
  and `2.8141e-6` at dt=0.01. The selected candidate is therefore dt=0.01;
  its measured relative error is `2.7229e-6`.

The `evidence/` directory and source files are committed. `artifacts/` contains
the run folders needed to regenerate the plots and stays untracked.
