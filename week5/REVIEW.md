# Week 5 evidence review

## Acceptance review

- **JAX differentiation:** analytic scalar values and derivatives match; maximum AD discrepancy `1.42e-14`; finite-difference discrepancy `4.95e-9`.
- **Acoustic forward run:** trace norm `11.57476950361405`; shot maxima, trace index 83, and receiver coordinates match; step-150 scattered/direct maximum ratio `0.0243873`. The input plot is two panels and depth increases downward; step-150 direct/echo PNGs are separate.
- **Enzyme derivatives:** pinned-nightly `x³` smoke test gives `(8,12,12)`; the actual timestep JVP/VJP produces reflector dot products `0.0348478902152` on both sides, with relative mismatch `1.59e-15`.
- **Checkpoint schedules:** budgets 1/3/5/10 use peaks 2/4/6/11 states and replay 28,680/1,695/990/642 steps per shot. All logged gradient indices descend from 239 to 0 exactly once; invalid restores, duplicate stores, bad calls/fetches, and budget overruns are zero. Checkpoint images have zero relative L2 difference from the full-history result at saved precision. The work plot has a log replay axis and a linear saved-bytes axis with the full-history baseline.
- **Marmousi:** nine shots, 1,200 steps, Treeverse with five extra slots only; peak six states / `20,788,320` bytes. The image norm is `6.7037740604e-4` (relative difference `5.91e-9` from the reference) and the transpose check is `1.73e-14`. Outputs are in the course-named `marmousi-born/`, `marmousi-image/`, with the four-panel figure `marmousi.png`.

## Automated checks

Commands run from repository root:

```text
cargo +stable test --no-default-features --manifest-path week5/seismic/Cargo.toml
CARGO_PROFILE_RELEASE_STRIP=none cargo +nightly-2026-09-05 test --release --features enzyme --manifest-path week5/seismic/Cargo.toml
CARGO_PROFILE_RELEASE_STRIP=none cargo +nightly-2026-09-05 run --release --features enzyme --manifest-path week5/seismic/Cargo.toml -- --enzyme-smoke
uv run --project week5 python week5/scripts/audit_checkpoints.py
uv run --project week5 python week5/scripts/forward_figures.py
uv run --project week5 python week5/scripts/adjoint_figures.py
uv run --project week5 python week5/scripts/marmousi_figures.py
git diff --check
```

Rust tests passed (4/4 in both stable and Enzyme-enabled release builds), the Enzyme smoke test passed, all figure/metric generation scripts completed, and whitespace validation passed. Plots were visually checked. All generated image/JSON evidence is below 5 MB per file. The course-provided JSON inputs and all generated `.npy` arrays are ignored; the two input hashes and commands needed to stage them are documented in `README.md`. `MARMOUSI-LICENSE` is tracked.

## Platform, repository, and remaining owner actions

The Enzyme kernel was compiled and executed on native `aarch64-apple-darwin` using `rustc 1.100.0-nightly (0ed41eb41 2026-09-04)` plus the pinned `nightly-2026-09-05` toolchain. Although Linux is available, no SSH alias or remote work directory was supplied, so a separate Linux run is not claimed. The course handout asks students to push the exercise repository. This work was pushed to `origin/main`; remote `refs/heads/main` was verified at commit `ca55a2f0f8445868305aaa153249740973d36ac4`. It has not been submitted to the course.

Week 6 Alibaba account/payment setup remains a student-owned action to take only after the instructor explains expected costs. No GPU was rented.
