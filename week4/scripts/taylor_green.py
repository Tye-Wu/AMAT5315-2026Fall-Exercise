#!/usr/bin/env python3
"""Run and compare the Taylor-Green solution, then draw the requested fields."""

import json
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


WEEK4 = Path(__file__).resolve().parent.parent
ARTIFACTS = WEEK4 / "artifacts"
EVIDENCE = WEEK4 / "evidence"
FIELD = shutil.which("field")
FLUID = shutil.which("fluid")
if FIELD is None or FLUID is None:
    raise SystemExit("Install the two CLIs first with: cargo install --path . --quiet")


def main():
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    run_dir = ARTIFACTS / "taylor-green"
    run_dir.mkdir(parents=True, exist_ok=True)
    input_field = subprocess.run(
        [FIELD, "taylor-green", "--n", "64"],
        cwd=WEEK4,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    run = subprocess.run(
        [
            FLUID, "--method", "rk4", "--nu", "0.1", "--dt", "0.01",
            "--t-end", "1", "--every", "0.1", "--out",
            "artifacts/taylor-green",
        ],
        cwd=WEEK4,
        input=input_field,
        capture_output=True,
        text=True,
    )
    if run.returncode:
        raise RuntimeError(run.stderr or run.stdout)
    (ARTIFACTS / "taylor-green.tsv").write_text(run.stdout)
    lines = run.stdout.splitlines()
    print("\n".join(lines[:2]))
    print(lines[-1])

    exact = subprocess.run(
        [
            FIELD, "taylor-green", "--n", "64", "--nu", "0.1", "--t", "1",
        ],
        cwd=WEEK4,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    exact_path = run_dir / "exact-t1.json"
    exact_path.write_text(exact)
    exact_data = json.loads(exact)
    frames = [json.loads(line) for line in (run_dir / "fields.jsonl").read_text().splitlines()]
    final = frames[-1]
    actual_vector = np.asarray(final["u"] + final["v"], dtype=float)
    exact_vector = np.asarray(exact_data["u"] + exact_data["v"], dtype=float)
    relative_error = np.linalg.norm(actual_vector - exact_vector) / np.linalg.norm(exact_vector)
    print(f"relative velocity error at t=1: {relative_error:.9e}")

    frame0, frame1 = frames[0], final
    n = exact_data["n"]
    grid = np.arange(n) * 2 * np.pi / n
    xx, yy = np.meshgrid(grid, grid)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.7), constrained_layout=True)
    image = None
    for ax, frame in zip(axes, (frame0, frame1)):
        omega = np.asarray(frame["omega"]).reshape(n, n)
        image = ax.imshow(
            omega,
            origin="lower",
            extent=(0, 2 * np.pi, 0, 2 * np.pi),
            cmap="RdBu_r",
            vmin=-2,
            vmax=2,
            interpolation="nearest",
        )
        u = np.asarray(frame["u"]).reshape(n, n)
        v = np.asarray(frame["v"]).reshape(n, n)
        stride = 4
        ax.quiver(
            xx[::stride, ::stride], yy[::stride, ::stride],
            u[::stride, ::stride], v[::stride, ::stride],
            color="black", scale=13, width=0.0022,
        )
        ax.set(
            xlim=(0, 2 * np.pi), ylim=(0, 2 * np.pi),
            xlabel="x", ylabel="y", title=f"t = {frame['t']:.1f}",
            aspect="equal",
        )
    fig.colorbar(image, ax=axes, label="vorticity ω", shrink=0.86)
    fig.suptitle("Taylor-Green decay with a shared vorticity scale")
    fig.savefig(EVIDENCE / "taylor-green.png", dpi=180)
    plt.close(fig)
    print(f"wrote {EVIDENCE / 'taylor-green.png'}")


if __name__ == "__main__":
    main()
