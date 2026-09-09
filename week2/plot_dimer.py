#!/usr/bin/env python3
"""Plot the dimer energy-error time series to week2/dimer.png.

Two panels from data computed by the md crate (week2/md, example
`dimer`), which runs the two-atom Lennard-Jones simulation through the
shared Integrator trait:

  left  : energy error E(t)-E(0) for Euler and velocity-Verlet,
          both 500 steps at dt = 0.01 (Verlet stays below 1e-3;
          Euler drifts past 0.5).
  right : velocity-Verlet alone for 5000 steps, error scaled by 1000
          so the bounded 1e-3 band is visible.

Regenerate the figure with:

    python3 week2/plot_dimer.py
"""

import io
import pathlib
import shutil
import subprocess

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent          # week2/
MANIFEST = HERE / "md" / "Cargo.toml"
OUT = HERE / "dimer.png"

# Palette (documented diverging/categorical hues), recessive chrome.
BLUE = "#2a78d6"     # velocity-Verlet
ORANGE = "#eb6834"   # Euler
INK = "#26241f"
MUTED = "#6b6a64"
GRID = "#e3e2db"
BAND = "#f0efec"


def run_dimer_example():
    """Run md's dimer example and return {run: (t, dE)} arrays."""
    if shutil.which("cargo") is None:
        raise SystemExit(
            "cargo not found on PATH. Export PATH so it includes "
            "~/.cargo/bin, e.g.  export PATH=\"$HOME/.cargo/bin:$PATH\""
        )
    proc = subprocess.run(
        ["cargo", "run", "--quiet", "--example", "dimer",
         "--manifest-path", str(MANIFEST)],
        check=True, capture_output=True, text=True,
    )
    runs = {}
    for line in proc.stdout.splitlines():
        if not line or line.startswith("run,"):
            continue
        run, t, de = line.split(",")
        runs.setdefault(run, ([], []))[0].append(float(t))
        runs.setdefault(run, ([], []))[1].append(float(de))
    return {run: (np.array(t), np.array(de)) for run, (t, de) in runs.items()}


def style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(True, axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def main():
    print(f"running {MANIFEST} example 'dimer' ...")
    data = run_dimer_example()
    t_e, dE_e = data["euler500"]
    t_v, dE_v = data["verlet500"]
    t_l, dE_l = data["verlet5000"]
    print(
        f"samples: euler500={len(t_e)}, verlet500={len(t_v)}, "
        f"verlet5000={len(t_l)}"
    )
    print(
        f"Verlet max |dE| (500)   = {np.abs(dE_v).max():.3e}"
    )
    print(
        f"Verlet max |dE| (5000)  = {np.abs(dE_l).max():.3e}"
    )
    print(f"Euler final dE (500)   = {dE_e[-1]:.3e}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    fig.patch.set_facecolor("white")

    # ---- left panel: both integrators, 500 steps ----
    ax1.plot(t_e, dE_e, color=ORANGE, lw=1.8, label="Euler")
    ax1.plot(t_v, dE_v, color=BLUE, lw=1.8, label="velocity-Verlet")
    ax1.axhspan(-1e-3, 1e-3, color=BAND, zorder=0)
    ax1.set_title("Euler vs velocity-Verlet, 500 steps (dt = 0.01)",
                  fontsize=11)
    ax1.set_xlabel("time $t$ (reduced units)")
    ax1.set_ylabel("energy error $E(t) - E(0)$  (units of $\\varepsilon$)")
    ax1.legend(frameon=False, fontsize=9)
    style_axes(ax1)

    # ---- right panel: Verlet alone, 5000 steps, error scaled by 1000 ----
    ax2.plot(t_l, 1e3 * np.array(dE_l), color=BLUE, lw=1.4)
    ax2.axhspan(-1.0, 1.0, color=BAND, zorder=0)   # +-1e-3 in actual units
    ax2.set_title("velocity-Verlet, 5000 steps (dt = 0.01)",
                  fontsize=11)
    ax2.set_xlabel("time $t$ (reduced units)")
    ax2.set_ylabel("$1000 \\times [E(t) - E(0)]$  (units of $\\varepsilon$)")
    ax2.set_ylim(-1.5, 1.5)
    style_axes(ax2)

    fig.suptitle(
        "Two-atom LJ dimer: total-energy error $E(t)-E(0)$ over time\n"
        "both start from rest at $r=1.2\\,\\sigma$, so $E(0)=U(1.2)$; "
        "shaded band $=\\pm10^{-3}\\,\\varepsilon$",
        fontsize=11,
    )
    fig.text(
        0.01, 0.01,
        "Reduced units ($\\varepsilon=\\sigma=1$, $m=1$). Data from the md "
        "crate integrators via `cargo run --example dimer`.",
        fontsize=8, color=MUTED,
    )

    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    fig.savefig(OUT, dpi=160, facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
