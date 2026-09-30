# Week 2 code review

Reviewed publication commit: `6820d65ef832304ee92d4f562910a0977d0efd83`

## Outcome

No blocking correctness, reproducibility, or scientific-validation findings remain. Release tests pass, the exact contract trajectory passes the independent checker, naive and cell-list results agree in adversarial boundary/cutoff tests, all figures and videos were inspected, and the public viewer loads without authentication.

## Findings and disposition

1. **Fixed — CLI help displayed two stray `+` characters.** The continuation lines in `print_help_and_exit` contained patch markers. They were removed before publication; `md --help` now prints the exact defaults and commands cleanly. Fix: `6820d65ef832304ee92d4f562910a0977d0efd83`.
2. **Fixed — old planning documents could be mistaken for the final contract.** Their early square-box/default values are intentionally preserved as history, but both now begin with a prominent archived-draft warning and link to the exact Part 3–5 design and verification plan. Fix: `6820d65ef832304ee92d4f562910a0977d0efd83`.
3. **Accepted, non-functional — strict Clippy style lints.** `cargo clippy -- -D warnings` flags explicit range indexing in the two-component integrator and one collapsible cutoff condition. These are style-only: the loops are bounded by validated equal-length arrays, all release tests pass, and the explicit two-coordinate form mirrors the course equations. They are recorded rather than mechanically rewritten at the final scientific-validation gate.

## Evidence reviewed

- `cargo fmt --check`
- `cargo test --manifest-path week2/md/Cargo.toml --release`: 25 library tests and 3 CLI integration tests passed
- `make -C week2 reproduce`
- `md check week2/artifacts`: drift `4.030239e-5`, `T_speed=0.525166`, `chi2/22=1.033164`, stored-energy error `1.913e-15`, `PASS`
- Three-run timing, Samply profile, three-size naive/cell benchmark, and raw result files
- Original-resolution figure inspection and first/last-frame movie inspection
- Public GitHub Pages first and last frames
