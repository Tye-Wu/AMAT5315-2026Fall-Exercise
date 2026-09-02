# AGENTS.md

Instructions and context for agents working in this repository.
`CLAUDE.md` references this file via `@AGENTS.md`, so it is part of the
context Claude Code loads.

## Repository purpose

This repo holds Yongtai Wu's weekly exercises for the course
**AMAT5315 – Modern Scientific Computing (2026 Fall)**. Each week's work
lives in its own folder, named `week1/`, `week2/`, and so on.

## Per-week layout

A week's folder follows the pattern set by `week1/`:

- `SPEC.md` — the exercise spec handed in for the week. Treat it as
  course material: read it, do not edit it.
- A self-contained solution module (e.g. `pi.py`). Where randomness is
  used, the solution is deterministic under a fixed seed.
- A test file (e.g. `test_pi.py`) that checks the spec, run with pytest.

## Working conventions

- Before touching code in a week, read that week's `SPEC.md`.
- Keep each week self-contained; do not reach across `week*/` folders.
- The spec's tests must pass before a week is considered done.
- Run a week's tests from the repo root with `pytest weekN/`.
- Python 3.11 is the baseline (see `setup.txt`); pytest is the test runner.

## Memory probe

Memory probe: W1-MEMORY-5315
