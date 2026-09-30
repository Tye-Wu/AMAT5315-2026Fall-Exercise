# Week 2: agentic coding with Rust

This directory contains a two-dimensional Lennard-Jones molecular-dynamics program in reduced units (`sigma = epsilon = mass = kB = 1`). The same Rust crate grows from a tested pair force and two-atom integrators into a periodic fluid, an independently checked trajectory format, two interchangeable pair-search paths, and a heating experiment.

## Build and test

```bash
cargo run --manifest-path md/Cargo.toml
cargo test --manifest-path md/Cargo.toml --release
```

`System` in `md/src/system.rs` owns the position and velocity arrays. Read-only physics routines borrow it as `&System`; integrators receive `&mut System` because they update both arrays. `Integrator::step` is the shared contract used by `advance` for `FreeFlight`, `Euler`, and `VelocityVerlet`. `#[test]` is a Rust attribute macro that registers a function with the Cargo test harness.

## Force and dimer checks

The pair model is

```text
U(r) = 4 (r^-12 - r^-6)
F(r) = -dU/dr = (24/r) (2 r^-12 - r^-6).
```

The force test compares the independently implemented analytic force with a central numerical derivative of the energy at separations on both sides of `r0 = 2^(1/6)`, using `h = 1e-5` and tolerance `1e-6 max(1, |F|)`. `field.png` adds a visual sign check: arrows point outward inside `r0`, inward outside it, and the negative-energy ring surrounds the repulsive core.

The dimer test releases two atoms from rest at separation 1.2 and passes either `Euler` or `VelocityVerlet` through the same driver. At `dt = 0.01`, the Verlet relative energy error remains below `1e-3` for both 500 and 5000 steps, while Euler's final relative error exceeds 0.5 after 500 steps. Recreate both figures with:

```bash
python3 plot_field.py
python3 plot_dimer.py
```

## Fluid contract and independent check

The default command is exactly the contract run:

```bash
cargo run --manifest-path md/Cargo.toml --release -- run \
  --n 100 --rho 0.8 --temperature 0.5 --dt 0.01 \
  --eq-steps 2000 --steps 10000 --sample-every 50 --seed 2026 \
  --out artifacts

# equivalent because all values above are defaults
cargo run --manifest-path md/Cargo.toml --release -- run --out artifacts

cargo run --manifest-path md/Cargo.toml --release -- check artifacts
```

`md check` rebuilds a naive-force `System` from every saved `pos` and `vel`, recomputes shifted potential and kinetic energies, and only then compares them with `E_pot` and `E_kin`. It applies the specified first/last-window drift, `T_speed = <v^2>/2`, and 24 equal-probability Maxwell-Boltzmann bins. The current reproducible run reports:

```text
secular drift  = 4.030239e-5   limit < 2e-3
T_speed        = 0.525166      |T_speed - 0.500| limit < 0.05
chi2/22        = 1.033164      limit < 2
stored E error = 1.913e-15     limit < 1e-8
PASS
```

Use `make reproduce` from `week2/` to regenerate `artifacts/` and repeat the check.

## Timing

The contract run was timed three times per program on macOS arm64. Each entry reports the median wall time and the full min-max range. `week2-sim.py` is the supplied NumPy baseline.

| Program | Median (s) | Range: min-max (s) |
| --- | ---: | ---: |
| NumPy `week2-sim.py` | 4.076 | 4.073-4.093 |
| Rust debug | 4.333 | 4.243-4.369 |
| Rust release | 0.288 | 0.287-0.292 |

The release median is 6.6% of the debug median, comfortably below the required one third. NumPy and debug are similar here; release Rust is faster. Reproduce all timing and scaling measurements with:

```bash
cargo build --manifest-path md/Cargo.toml
cargo build --manifest-path md/Cargo.toml --release
python3 benchmark.py
```

Raw values and platform metadata are in `benchmark-results.json`.

## Profile

Profiles used the release binary at `N = 400`, `eq_steps = 200`, and `steps = 1000`. The PNGs summarize inclusive samples from the saved samply profiles; the `.json.gz` files are the raw profiles and can be opened with `samply load`.

| Version | Force share (%) | Elapsed time (s) |
| --- | ---: | ---: |
| Naive | 97.4 | 0.154 |
| Cell list | 95.9 | 0.145 |

```bash
dsymutil --flat --out md/target/release/md.dwarf md/target/release/md
samply record --save-only --output profile-naive.json.gz \
  md/target/release/md run --force naive --n 400 --eq-steps 200 \
  --steps 1000 --out /tmp/md-prof-naive
samply record --save-only --output profile-cells.json.gz \
  md/target/release/md run --force cells --n 400 --eq-steps 200 \
  --steps 1000 --out /tmp/md-prof-cells
python3 profile_summary.py
```

The naive profile establishes the target: pair interactions take more than 90% of samples. The cell-list elapsed time is lower. Its force share remains high because share is a fraction of a now shorter run, not an absolute duration.

## Benchmark

The following release measurements use 100 equilibration plus 500 production steps, three runs per cell. Times are median (min-max), in seconds.

| N | Naive (s) | Cells (s) | Speedup: naive/cells |
| ---: | ---: | ---: | ---: |
| 100 | 0.01055 (0.01041-0.01113) | 0.01835 (0.01803-0.01849) | 0.58x |
| 400 | 0.07630 (0.07590-0.07653) | 0.06938 (0.06878-0.07005) | 1.10x |
| 1600 | 0.93265 (0.91980-0.93327) | 0.28000 (0.27867-0.28825) | 3.33x |

Regenerate the plotted data with `python3 benchmark.py`, then render `scaling.png` with `python3 plot_scaling.py`. At small N, constructing cells costs more than the pair distances it saves. As N grows at fixed density and cutoff, the naive path examines all `N(N-1)/2` pairs while the cell path searches only the current cell and eight wrapped neighbours; the speedup therefore rises with N and exceeds 2 at N = 1600.

## Heating and melting

`--ramp-to` rescales velocities every 50 production steps toward a target that rises linearly from `--temperature` at production step 0 to the final target at the last step. `run.json` records `ramp_to`. Recreate the cold, hot, and 400-atom heating evidence with:

```bash
cargo run --manifest-path md/Cargo.toml --release -- run \
  --temperature 0.2 --out /tmp/week2-cold
cargo run --manifest-path md/Cargo.toml --release -- video \
  /tmp/week2-cold --out cold.mp4

cargo run --manifest-path md/Cargo.toml --release -- run \
  --temperature 1.0 --out /tmp/week2-hot
cargo run --manifest-path md/Cargo.toml --release -- video \
  /tmp/week2-hot --out hot.mp4

cargo run --manifest-path md/Cargo.toml --release -- run \
  --n 400 --temperature 0.2 --ramp-to 1.2 \
  --steps 20000 --sample-every 100 --out ../docs
python3 analyze_structure.py
```

The two constant-temperature trajectories separate structure from animation alone. The mean speed temperatures are 0.189 (cold) and 0.954 (hot). Their long-range RDF contrasts, defined as `RMS[g(r)-1]` for `r > 2`, are 0.560 and 0.186: the cold run preserves several neighbour-shell peaks, while the hot run approaches the liquid limit `g(r) -> 1` beyond the first few shells.

The heating run stores 400 atoms and 200 frames. The first and last saved speed temperatures are 0.204 and 1.197. Over the first and last ten frames, the same long-range contrast falls from 0.348 to 0.097. This is the quantitative melting evidence plotted in `melting.png`; `cold.mp4` and `hot.mp4` provide the corresponding particle-plus-RDF visual check.

## Pages

The public viewer is served from `docs/` and contains the 400-atom heating trajectory above. Its final URL is recorded here after the publication gate.

## Recording

`RECORDING_SCRIPT.md` is a ready-to-use, under-two-minute walkthrough. The final recording must use the student's own voice and screen capture after the fresh-clone and public Pages gates pass.
