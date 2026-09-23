#!/usr/bin/env python3
"""Run the Part 3 random-flow, diffusive-limit, and advective-limit studies."""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import ARTIFACTS, EVIDENCE, make_field, read_tsv, run_fluid, save_json


def stable(result):
    return result.returncode == 0


def plot_random(field):
    frames = [
        json.loads(line)
        for line in (ARTIFACTS / "random" / "fields.jsonl").read_text().splitlines()
    ]
    requested = (0, 2, 5, 10)
    chosen = [min(frames, key=lambda frame: abs(frame["t"] - t)) for t in requested]
    n = field["n"]
    vlim = max(abs(value) for value in chosen[0]["omega"])
    grid = np.arange(n) * 2 * np.pi / n
    xx, yy = np.meshgrid(grid, grid)
    fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.8), constrained_layout=True)
    image = None
    for ax, frame, target in zip(axes, chosen, requested):
        omega = np.asarray(frame["omega"]).reshape(n, n)
        image = ax.imshow(
            omega,
            origin="lower",
            extent=(0, 2 * np.pi, 0, 2 * np.pi),
            cmap="RdBu_r",
            vmin=-vlim,
            vmax=vlim,
            interpolation="nearest",
        )
        stride = 8
        u = np.asarray(frame["u"]).reshape(n, n)
        v = np.asarray(frame["v"]).reshape(n, n)
        ax.quiver(
            xx[::stride, ::stride], yy[::stride, ::stride],
            u[::stride, ::stride], v[::stride, ::stride],
            color="black", scale=25, width=0.0017,
        )
        rows = read_tsv(ARTIFACTS / "random.tsv")
        energy, enstrophy = min(rows, key=lambda row: abs(row[0] - frame["t"]))[1:]
        ax.set(
            xlabel="x", title=f"t={target}\nE={energy:.3f}, Z={enstrophy:.3f}",
            aspect="equal", xlim=(0, 2 * np.pi), ylim=(0, 2 * np.pi),
        )
    axes[0].set_ylabel("y")
    fig.colorbar(image, ax=axes, label="vorticity ω", shrink=0.84)
    fig.suptitle(f"Seeded random flow, fixed scale ±{vlim:.3f}")
    fig.savefig(EVIDENCE / "random.png", dpi=180)
    plt.close(fig)


def run_taylor_green_scan():
    initial = make_field("taylor-green", "--n", 64)
    scans = {}
    for dt in (0.032, 0.033):
        name = f"scan/taylor-green-rk4-dt{dt:.3f}"
        result = run_fluid(
            initial, "rk4", 0.1, dt, 8.0, 0.5, name,
            expect_failure=None,
        )
        scans[dt] = result
    return scans


def run_random_scan(field):
    runs = {}
    for dt in (0.038, 0.040):
        name = f"scan/random-rk4-dt{dt:.3f}"
        runs[dt] = run_fluid(
            field, "rk4", 0.004, dt, 10.0, 0.5, name,
            expect_failure=None,
        )

    # If this phase realization does not bracket the boundary at the printed
    # candidates, expand the scan in 0.002 increments while retaining every run.
    low, high = 0.038, 0.040
    for _ in range(6):
        if stable(runs[low]) and not stable(runs[high]):
            break
        if not stable(runs[low]) and not stable(runs[high]):
            high = low
            low = round(low - 0.002, 3)
            if low <= 0:
                raise RuntimeError("failed to find a stable random-flow RK4 step")
            if low not in runs:
                name = f"scan/random-rk4-dt{low:.3f}"
                runs[low] = run_fluid(
                    field, "rk4", 0.004, low, 10.0, 0.5, name,
                    expect_failure=None,
                )
        else:
            low = high
            high = round(high + 0.002, 3)
            name = f"scan/random-rk4-dt{high:.3f}"
            runs[high] = run_fluid(
                field, "rk4", 0.004, high, 10.0, 0.5, name,
                expect_failure=None,
            )
    if not (stable(runs[low]) and not stable(runs[high])):
        raise RuntimeError("random RK4 scan did not bracket a stable/unstable pair")
    runs["bracket"] = [low, high]
    runs["euler"] = run_fluid(
        field, "euler", 0.004, 0.01, 10.0, 0.5, "scan/random-euler-dt0.010",
        expect_failure=None,
    )
    return runs


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    random_field = make_field(
        "random", "--n", 128, "--seed", 2026, "--k-min", 2, "--k-max", 6
    )
    save_json(ARTIFACTS / "random-initial.json", random_field)
    baseline = run_fluid(
        random_field, "rk4", 0.004, 0.01, 10.0, 0.1, "random",
        expect_failure=False,
    )
    print("Random-flow run (head -2 and tail -1):")
    lines = baseline.stdout.splitlines()
    print("\n".join(lines[:2]))
    print(lines[-1])

    unstable = make_field("taylor-green", "--n", 64)
    initial_unstable = run_fluid(
        unstable, "rk4", 0.1, 0.04, 4.0, 0.1,
        "unstable/taylor-green", expect_failure=True,
    )
    print("\nTaylor-Green dt=0.04 (tail -2):")
    print("\n".join(initial_unstable.stdout.splitlines()[-2:]))

    speeds = np.hypot(random_field["u"], random_field["v"])
    umax = float(np.max(speeds))
    advective_bound = 2.83 / (umax * math.sqrt(2.0) * 42.0)
    print(f"largest speed of random initial field: {umax:.9f}")
    print(f"RK4 advective estimate 2.83/(Umax*|k|max): {advective_bound:.9f}")

    tg_scans = run_taylor_green_scan()
    random_scans = run_random_scan(random_field)
    bracket = random_scans["bracket"]
    measured_midpoint = sum(bracket) / 2
    print(f"Taylor-Green scan return codes: {[(dt, r.returncode) for dt, r in tg_scans.items()]}")
    print(f"Random RK4 boundary bracket: {bracket[0]:.3f} to {bracket[1]:.3f}")
    print(f"Boundary midpoint / advective bound: {measured_midpoint / advective_bound:.4f}")
    print(f"Random Euler dt=0.01 return code: {random_scans['euler'].returncode}")

    save_json(
        ARTIFACTS / "scan-summary.json",
        {
            "umax": umax,
            "advective_bound": advective_bound,
            "taylor_green_return_codes": {
                str(dt): run.returncode for dt, run in tg_scans.items()
            },
            "random_rk4_bracket": bracket,
            "random_rk4_return_codes": {
                str(key): value.returncode
                for key, value in random_scans.items()
                if key not in ("bracket", "euler")
            },
            "random_euler_return_code": random_scans["euler"].returncode,
        },
    )
    plot_random(random_field)
    print(f"wrote {EVIDENCE / 'random.png'}")


if __name__ == "__main__":
    main()
