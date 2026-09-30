# Week 2 Parts 3–5 verification plan

## Part 3 — exact fluid and checker

- Implement the rectangular triangular lattice and deterministic Gaussian velocities.
- Implement periodic minimum-image LJ forces and the shifted potential.
- Run velocity-Verlet with thermostat only during equilibration.
- Save the exact JSON/JSONL contract.
- Verify unit tests, CLI schema, a full default run, independently recomputed energy, energy drift, kinetic temperature, and the 24-bin Maxwell speed-shape statistic.

Acceptance command:

```bash
cargo test --manifest-path week2/md/Cargo.toml --release
make -C week2 reproduce
cargo run --manifest-path week2/md/Cargo.toml --release -- check week2/artifacts
```

## Part 4 — profile and cell lists

- Time the supplied NumPy baseline and Rust debug/release builds three times.
- Record a release profile at `N=400` and confirm force work exceeds 90% of samples.
- Add cell lists behind `--force naive|cells` without changing results.
- Test naive/cells equality at difficult periodic/cutoff configurations.
- Benchmark `N=100,400,1600`, three runs each, and plot the crossover.

Acceptance: release time below one third of debug time, rising cell-list speedup with a value above `2x` by `N=1600`, and saved raw timing/profile evidence.

## Part 5 — heating, public viewer, and release proof

- Generate cold `T=0.2` and hot `T=1.0` 100-atom controls plus videos below 2 MB.
- Generate 400 atoms, 200 frames, and a linear `0.2 -> 1.2` heating run in `docs/`.
- Recompute RDFs and long-range contrast from the saved trajectories; produce `melting.png`.
- Inspect first/last video frames and the local HTML viewer.
- Publish `docs/` through GitHub Pages and verify it without authentication.
- Review code, commit the review, then obtain a fresh published copy and repeat the test/reproduce/check commands.
- Leave only the student's identity-dependent voice recording as a manual step; use `week2/RECORDING_SCRIPT.md`.
