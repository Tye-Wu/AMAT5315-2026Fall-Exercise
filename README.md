# AMAT5315 – Modern Scientific Computing: Weekly Exercises

Weekly exercise work for **AMAT5315 – Modern Scientific Computing (2026 Fall)**,
completed by Yongtai Wu.

Each week's work lives in its own self-contained folder — `week1/`, `week2/`,
and so on — containing that week's exercise spec, the solution module, and its
test. A small AI coding agent writes most of the code; git records the order in
which the work was done. The agent-facing memory file `AGENTS.md` (imported by
Claude Code through `CLAUDE.md`) describes this layout, and a `tutor` project
skill turns course lesson sheets into guided, one-step-at-a-time study sessions.

**Week 1** implements a Monte Carlo estimator of π: it throws random darts at a
unit square, counts how many land within distance 1 of the origin, and returns
four times that fraction. See `week1/SPEC.md` for the definition of "correct."

## Requirements

- Python 3.10 or newer (`setup.txt` records the exact versions used)
- `pytest` — the test runner
- `pypdf` — reads the PDF lesson sheets (used by the `tutor` skill)

## Install pytest

```bash
python3 -m pip install pytest pypdf
```

## Run the test

```bash
python3 -m pytest week1/
```

Expected: `1 passed`.

<!-- Screenshot placeholder: drop the cropped "pytest green" image here, e.g.
![pytest passing](assets/pytest-green.png)
with short alt text such as "Terminal: `python3 -m pytest week1/`, result 1 passed."
Replace this comment once the image file exists. -->
