#!/usr/bin/env python3
"""Render the Lennard-Jones pair field around one atom to week2/field.png.

The scalar data come from the `md` crate in week2/md/: the example
`field.rs` calls lj_energy and lj_force on a grid around a central atom
at the origin and prints ``x,y,V,Fx,Fy`` to stdout. This script runs that
example and draws the pair energy as colors and the pair force as arrows.

Regenerate the figure with:

    python3 week2/plot_field.py
"""

import io
import pathlib
import shutil
import subprocess

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

HERE = pathlib.Path(__file__).resolve().parent          # week2/
REPO = HERE.parent                                       # repo root
MANIFEST = HERE / "md" / "Cargo.toml"
OUT = HERE / "field.png"

# Energy display range (reduced units, epsilon = 1): the attractive well
# bottoms out at -1 at the equilibrium separation; V > 0 is repulsive.
VMIN, VMAX = -1.05, 6.0
CORE = 0.80      # mask the divergent r -> 0 core inside this radius
ARROWS_PER_SIDE = 14

# Visual style: diverging blue->red with a neutral centre at V = 0 (polarity:
# blue = attractive V < 0, red = repulsive V > 0), recessive axes, dark ink.
DIVERGING = "coolwarm"
INK = "#26241f"
MUTED = "#6b6a64"
GRID_LINE = "#d8d7d0"
ATOM = "#26241f"


def run_field_example():
    """Run md's field example and return numpy arrays on the square grid."""
    if shutil.which("cargo") is None:
        raise SystemExit(
            "cargo not found on PATH. Export PATH so it includes "
            "~/.cargo/bin, e.g.  export PATH=\"$HOME/.cargo/bin:$PATH\""
        )
    proc = subprocess.run(
        ["cargo", "run", "--quiet", "--example", "field",
         "--manifest-path", str(MANIFEST)],
        check=True, capture_output=True, text=True,
    )
    data = np.loadtxt(io.StringIO(proc.stdout), delimiter=",", skiprows=1)
    n = int(round(np.sqrt(data.shape[0])))
    cols = np.array(np.split(data, n, axis=0))   # (n, n, 5): [x, y, V, Fx, Fy]
    return cols[..., 0], cols[..., 1], cols[..., 2], cols[..., 3], cols[..., 4]


def main():
    print(f"running {MANIFEST} example 'field' ...")
    x, y, v, fx, fy = run_field_example()

    r = np.hypot(x, y)
    core = r < CORE

    # The figure: energy as color, force as arrows.
    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    fig.patch.set_facecolor("white")

    # --- energy field (colors) ---
    v_plot = np.where(core, np.nan, v)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=VMIN, vmax=VMAX)
    im = ax.imshow(
        v_plot, origin="lower", extent=[x.min(), x.max(), y.min(), y.max()],
        cmap=DIVERGING, norm=norm, interpolation="bilinear", aspect="equal",
    )

    # --- force arrows (direction + magnitude via length) ---
    step = max(1, (len(x) - 1) // (ARROWS_PER_SIDE - 1))
    xa, ya, fa_x, fa_y = x[::step, ::step], y[::step, ::step], \
        fx[::step, ::step], fy[::step, ::step]
    fmag = np.hypot(fa_x, fa_y)
    # Saturating length scale so the huge near-core repulsion cannot blow out
    # the well region: length ~ tanh(|F| / F0), direction unchanged.
    f0 = 4.0
    with np.errstate(divide="ignore", invalid="ignore"):
        disp = np.where(fmag > 0, np.tanh(fmag / f0), 0.0)
        u = np.where(fmag > 0, fa_x / fmag * disp, 0.0)
        w = np.where(fmag > 0, fa_y / fmag * disp, 0.0)
    ar = np.hypot(xa, ya)
    u[ar < CORE] = np.nan
    w[ar < CORE] = np.nan
    # quiver with units='xy': a vector of length 1 spans `1/scale` data units.
    cell = 2.0 * x.max() / (ARROWS_PER_SIDE - 1)
    ax.quiver(
        xa, ya, u, w, units="xy", scale=1.0 / (0.85 * cell),
        angles="xy", color=INK, width=0.004, headwidth=3.2, headlength=4.2,
        headaxislength=3.2, alpha=0.9, zorder=3,
    )

    # --- annotations: the atom and the equilibrium ring r0 = 2^(1/6) ---
    ax.scatter([0.0], [0.0], s=90, color=ATOM, zorder=5)
    r0 = 2.0 ** (1.0 / 6.0)
    ring = plt.Circle((0, 0), r0, fill=False, ls=(0, (4, 3)), lw=1.1,
                      edgecolor=INK, alpha=0.55, zorder=4)
    ax.add_patch(ring)
    ax.annotate("equilibrium ring\n$r_0 = 2^{1/6}\\,\\sigma$",
                xy=(r0, 0), xytext=(1.7, 1.75),
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9),
                color=MUTED, fontsize=9, ha="center")

    # --- axes, labels, colour bar ---
    ax.set_xlabel("x (in units of $\\sigma$)")
    ax.set_ylabel("y (in units of $\\sigma$)")
    ax.set_title(
        "Lennard-Jones pair field around one atom\n"
        "colours: pair energy $V(r)$; arrows: pair force $F(r)=-V'(r)$",
        fontsize=12,
    )
    ax.set_xlim(-2.4, 2.4)
    ax.set_ylim(-2.4, 2.4)
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_LINE)
    ax.tick_params(colors=MUTED, length=0)
    for spine in ax.spines.values():
        spine.set_color(GRID_LINE)

    cb = fig.colorbar(im, ax=ax, pad=0.02, extend="both")
    cb.set_label("pair energy $V(r)$  (units of $\\varepsilon$)")
    cb.ax.tick_params(colors=MUTED)
    cb.outline.set_visible(False)

    fig.text(
        0.02, 0.015,
        "Reduced units: $\\varepsilon=\\sigma=1$. Arrow length is a saturating "
        "function of |F|; the near-core repulsion is clipped.",
        fontsize=8, color=MUTED,
    )

    fig.tight_layout()
    fig.savefig(OUT, dpi=160, facecolor="white")
    print(f"wrote {OUT}")
    return fig, ax, x, y, v, fx, fy


if __name__ == "__main__":
    main()
