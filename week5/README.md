# Week 5 — Automatic differentiation, acoustic waves, and checkpointed adjoints

This week connects three ideas that are easy to study separately but powerful together: automatic differentiation (AD), wave-equation modeling, and memory-aware reverse-mode differentiation. The same discrete acoustic update is used throughout, so the derivative checks, the Born data, and the adjoint image all refer to one consistent numerical model.

## Learning goals

1. Distinguish forward-mode AD (`Jv`) from reverse-mode AD (`Jᵀy`) and check both against independent numerical evidence.
2. Treat a time-stepping PDE as a differentiable program: a local derivative of one update can be composed into a derivative of the full simulation.
3. Use the dot-product identity `⟨Jv,y⟩ = ⟨v,Jᵀy⟩` to verify that the implemented Born operator and adjoint are transposes.
4. Understand why reverse-mode differentiation of a long simulation needs saved intermediate states, and how checkpoint scheduling trades recomputation for memory.
5. Apply the validated method first to a small reflector, then to a larger Marmousi velocity model without storing its complete time history.

## The numerical experiment

The program advances a two-dimensional pressure field using the five-point discrete Laplacian. At each step it applies the acoustic update, adds a Ricker source with a unit-Gaussian spatial footprint, and damps the boundary with a sponge. The outermost grid values remain zero. Receivers record the field after each update. The model and exact experiment settings are in the ignored input files `inputs/reflector.json` and `inputs/marmousi.json`; provenance and licensing for the Marmousi source are recorded in `MARMOUSI-LICENSE` and the input metadata.

The update is a recurrence: the next field depends on the current and previous fields. That matters for reverse mode: to differentiate an earlier step, one needs the primal state at that step. Keeping every state is simple but expensive; recomputing selected intervals is the checkpointing solution studied in Part 4.

## What each part does, why, and how it is verified

### Part 1 — Lennard-Jones differentiation

The Lennard-Jones pair potential provides a small, known function on which to understand AD before introducing a PDE. The implementation evaluates a scalar energy and its derivative over 601 distances from 0.95 to 2.50. Both forward and reverse AD use 64-bit arithmetic.

The independent checks serve different purposes: the analytic derivative checks the mathematical formula; centered finite differences check the implemented function without relying on AD; and the printed computation graph makes the reverse-mode accumulation visible. The timings across cluster sizes illustrate that a reverse pass can compute a scalar function's full gradient at a cost that does not grow like one fresh forward evaluation per parameter.

Verified values at `r=1.3` include `U = -0.6570169144600471`, `dU/dr = 2.239979929791143`, and the reverse-mode intermediate adjoint `a = -2.3425903117359734`. The maximum AD-versus-analytic error is about `1.42e-14`; the maximum finite-difference error is about `4.95e-9`. For the `P=3072` scaling case, the measured forward/reverse time ratio is about `1854.8` on this host (timings are hardware- and load-dependent; correctness does not depend on that ratio).

Evidence: `artifacts/ad/derivatives.json`, `modes.png`, `graph.png`, `grad-graph.png`, `scaling.json`, and `scaling.png`.

### Part 2 — Acoustic forward model and reflector

This part implements the discrete wave solver and compares a background-only simulation with one containing a thin velocity perturbation. The difference isolates the reflected/scattered response while keeping source, grid, and time stepping identical. The three-shot, 14-receiver experiment has 41×41 cells and 240 time steps.

The forward trace norm checks the overall data scale; per-shot maxima, trace indices and receiver coordinates catch source/update/recording off-by-one errors; and saving selected pressure frames shows that the wave actually propagates and the perturbation generates an echo. The waveform and input plots make the model geometry, Ricker pulse, shot locations, receivers, and gathers inspectable. Receivers sample after each update: trace index 83 is the sample following update 84.

Verified: trace L2 norm `11.57476950361405`; shot maxima `[0.6080951434, 0.5927139735, 0.6080951434]`, each at trace index 83, with its receiver index and `[x,z]` coordinate reported in the terminal and `result.json`. At step 150, the direct-field maximum is `0.25524342` and the echo maximum is `0.00622470` (about 2.44% of the direct maximum). The trace tensor has shape `(3, 240, 14)`.

Evidence: `artifacts/inputs.png` (two panels: combined model/survey geometry and the Ricker pulse), `artifacts/forward/gathers.png`, the standalone `artifacts/forward/wavefield.png` and `echo.png` snapshots, and their combined `wavefield-echo-step150.png`; machine-readable values and frame metadata are in `artifacts/forward/result.json` and `recording.json`.

### Part 3 — Enzyme JVP/VJP, Born data, and adjoint image

The local time-step kernel is differentiated by Enzyme. A tangent input to the velocity model produces the Born response `Jv` (JVP). The receiver data seed is propagated backward to produce `Jᵀy` (VJP); correlating the adjoint field with the local derivative gives the image. This is why the Born and adjoint paths are not implemented as unrelated formulas: they are two directions through the derivative of the same discrete program.

First, the toolchain smoke test differentiates `x³` at `x=2` and must print primal/JVP/VJP values `8/12/12`. On the reflector, a dot-product test checks `⟨Jv,y⟩` against `⟨v,Jᵀy⟩`; this is a strong test of indexing, receiver injection, time ordering, and derivative consistency. The image depth profile checks whether the recovered energy lands at the known reflector, while a saved reverse wavefield makes the propagation direction tangible.

Verified on the reflector: both dot products are `0.0348478902152`, relative mismatch `1.59e-15`; image peak row 21 is exactly the known 2.1 km reflector depth. Full history uses 241 states (`6,481,936` bytes for the two pressure fields per state). The Enzyme cube smoke test returned `8, 12, 12`.

Evidence: `artifacts/born/`, `artifacts/adjoint/image.png`, and `artifacts/adjoint/wavefield.png`.

### Part 4 — Treeverse checkpointing

Reverse mode needs the primal state at each reverse step. Full-history storage is a useful correctness reference, but its storage grows with simulation duration and grid area. Treeverse schedules `restore`, `call` (recompute), `store`, `grad`, and `fetch` actions so only a bounded number of states are resident. The initial state is reserved, so a budget of `b` *extra* slots permits at most `b+1` saved states.

Four extra-slot budgets are tested. The audit independently interprets every action log: each forward call starts at the expected state; every restore refers to a saved state; stores are unique; the budget is respected; gradients occur exactly once in descending order; and only the initial state remains at the end. The resulting images are compared against the full-history image, not just against a second checkpoint run.

| Extra slots | Peak saved states | Replay calls per shot | Image relative L2 error vs. full history |
|---:|---:|---:|---:|
| 1 | 2 | 28,680 | 0 |
| 3 | 4 | 1,695 | 0 |
| 5 | 6 | 990 | 0 |
| 10 | 11 | 642 | 0 |

All budgets have zero invalid restores, duplicate stores, bad calls/fetches, and budget overruns. The work counts match the course reference. This exposes the central tradeoff: more checkpoints cost more memory but reduce recomputation.

Evidence: `artifacts/checkpoint-actions.png` (shot 0, five extra slots), `checkpoint-work.png` (replay steps and saved bytes, including full history), `checkpoint-audit.json`, and the per-shot `actions-*.json` logs in each `artifacts/checkpoint-{1,3,5,10}/` directory.

### Part 5 — Marmousi scale-up

The larger 805×269 Marmousi model uses 1,200 steps, nine shots, and 91 receivers. It is run with Enzyme JVP to generate Born data, then Enzyme VJP under Treeverse with five extra checkpoint slots. Full-history adjoint storage is deliberately not used for this case. The image norm checks agreement with the course reference; the transposition dot-product check validates the large run's forward/adjoint consistency; and the saved-state count and bytes document the memory bound.

Verified: image L2 norm `6.7037740604e-4` (relative difference about `5.91e-9` from the reference `6.7037741e-4`); transpose relative error `1.73e-14`; 10,800 reverse steps and 70,956 replay steps over nine shots; peak six saved states, `20,788,320` bytes. Born data shape is `(9, 1200, 91)`. The four-panel figure displays the background, perturbation, Born gather for the shot at 10 km, and raw adjoint image; the strongest image energy is in the shallow layers. Even an accurate derivative need not reproduce the perturbation pixel-for-pixel: the RTM image is `JᵀJm`, not `m`. The normal operator `JᵀJ` blurs and reshapes the model according to finite source bandwidth, limited illumination, and receiver coverage, while deeper structure is weakly illuminated. No depth-dependent display gain is applied.

Evidence: `artifacts/marmousi.png`, `artifacts/marmousi-validation.json`, `artifacts/marmousi-born/result.json`, and `artifacts/marmousi-image/result.json`. The larger raw arrays and source input are intentionally not tracked; regenerate them using the commands below.

## Reproducing the computations

Run from the repository root. Plotting scripts use the `uv` project declared in `week5/pyproject.toml`; the Enzyme kernel requires the pinned Rust nightly and Enzyme component in `week5/seismic/rust-toolchain.toml`.

```bash
# Part 1
uv run --project week5 python week5/scripts/ad.py
uv run --project week5 python week5/scripts/cluster_scaling.py

# Reflector forward run (portable stable Rust build, with Enzyme disabled)
cargo +stable run --release --no-default-features --manifest-path week5/seismic/Cargo.toml -- \
  --experiment week5/inputs/reflector.json --out week5/artifacts/forward --mode forward --every 3
uv run --project week5 python week5/scripts/forward_figures.py

# Enzyme verification, Born modeling, and full-history reference (small model only)
cargo +nightly-2026-09-05 test --release --features enzyme --manifest-path week5/seismic/Cargo.toml
cargo +nightly-2026-09-05 run --release --features enzyme --manifest-path week5/seismic/Cargo.toml -- --enzyme-smoke
cargo +nightly-2026-09-05 run --release --features enzyme --manifest-path week5/seismic/Cargo.toml -- \
  --experiment week5/inputs/reflector.json --out week5/artifacts/born --mode born
cargo +nightly-2026-09-05 run --release --features enzyme --manifest-path week5/seismic/Cargo.toml -- \
  --experiment week5/inputs/reflector.json --out week5/artifacts/adjoint --mode adjoint \
  --data week5/artifacts/born/born_data.npy --storage full --every 3
uv run --project week5 python week5/scripts/adjoint_figures.py

# Checkpointed reflector budgets and their audits/figures
for b in 1 3 5 10; do
  cargo +nightly-2026-09-05 run --release --features enzyme --manifest-path week5/seismic/Cargo.toml -- \
    --experiment week5/inputs/reflector.json --out week5/artifacts/checkpoint-$b --mode adjoint \
    --data week5/artifacts/born/born_data.npy --storage treeverse --checkpoints $b --every 3
done
uv run --project week5 python week5/scripts/audit_checkpoints.py

# Marmousi — Treeverse only; do not replace --storage treeverse with full history
cargo +nightly-2026-09-05 run --release --features enzyme --manifest-path week5/seismic/Cargo.toml -- \
  --experiment week5/inputs/marmousi.json --out week5/artifacts/marmousi-born --mode born
cargo +nightly-2026-09-05 run --release --features enzyme --manifest-path week5/seismic/Cargo.toml -- \
  --experiment week5/inputs/marmousi.json --out week5/artifacts/marmousi-image --mode adjoint \
  --data week5/artifacts/marmousi-born/born_data.npy --storage treeverse --checkpoints 5 --every 20
uv run --project week5 python week5/scripts/marmousi_figures.py
```

## Evidence inventory and generating commands

The commands above are the full invocations; this inventory maps each named output to its generator. All listed JSON and PNG evidence is committed. `.npy` arrays shown below are generated locally and remain Git-ignored, as required.

| Evidence files | Generating command |
|---|---|
| `artifacts/ad/derivatives.json`, `modes.png`, `graph.png`, `grad-graph.png` | `uv run --project week5 python week5/scripts/ad.py` |
| `artifacts/ad/scaling.json`, `scaling.png` | `uv run --project week5 python week5/scripts/cluster_scaling.py` |
| `artifacts/inputs.png`; `artifacts/forward/gathers.png`, `wavefield.png`, `echo.png`, `wavefield-echo-step150.png`; `forward/result.json`, `run.json`, `recording.json`, `traces.npy`, `wavefield.npy`, `echo.npy` | Stable forward command plus `uv run --project week5 python week5/scripts/forward_figures.py` |
| `artifacts/born/result.json`, `run.json`, `born_data.npy` | Enzyme `--mode born` command (small reflector) |
| `artifacts/adjoint/result.json`, `run.json`, `recording.json`, `image.npy`, `wavefield.npy`; `image.png`, `wavefield.png` | Enzyme `--mode adjoint --storage full` command plus `uv run --project week5 python week5/scripts/adjoint_figures.py` |
| `artifacts/checkpoint-{1,3,5,10}/result.json`, `run.json`, `recording.json`, `image.npy`, `wavefield.npy`, `actions-{0,1,2}.json` | Four-command Treeverse loop in the checkpoint section |
| `artifacts/checkpoint-audit.json`, `checkpoint-actions.png`, `checkpoint-work.png` | `uv run --project week5 python week5/scripts/audit_checkpoints.py` |
| `artifacts/marmousi-born/result.json`, `run.json`, `born_data.npy` | Enzyme Marmousi `--mode born` command |
| `artifacts/marmousi-image/result.json`, `run.json`, `recording.json`, `image.npy`, `wavefield.npy`, `actions-{0..8}.json`; `artifacts/marmousi-validation.json`, `marmousi.png` | Enzyme Marmousi Treeverse command plus `uv run --project week5 python week5/scripts/marmousi_figures.py` |
| Per-run `result.json` quantities (trace/Born norms, maxima, transpose check, replay counts, saved states and bytes) | Written by the corresponding Rust CLI command listed above |

## Reproducibility and scope notes

- Before running from a fresh clone, copy the exact course-provided experiment JSONs into the ignored `week5/inputs/` directory. The inputs used for this run had SHA-256 `22b84948e28e8e13e958f47b2d4bf9a70a3d0c8ee6ffcaab3cdeb8003b9902a6` (`reflector.json`) and `67beabdee96d8e3f6433c2f897b3b24b0c744e7808784148bb06310665d888b0` (`marmousi.json`). Verify them with `shasum -a 256 week5/inputs/*.json` before rerunning. JSON files, plots, action logs, and validation summaries are the reviewable evidence. Generated `.npy` arrays, the Python environment, Rust build output, and experiment input JSONs are ignored by Git. Marmousi input provenance and its license are retained.
- Enzyme was built and executed successfully on the available native Apple Silicon host with the pinned nightly. Linux access details were not supplied in this task, so a separate remote Linux run is not claimed.
- Performance ratios vary by machine and load. Scientific verification is based on numerical values, finite-difference/analytic checks, transpose consistency, exact schedule audits, and image comparisons.
- The Week 5 exercise work is complete in this repository. The course handout asks students to push the repository; the push status and commit are recorded in `REVIEW.md` after remote verification. This is not a course-platform submission.
