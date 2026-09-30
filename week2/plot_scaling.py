#!/usr/bin/env python3
"""Plot naive and cell-list seconds per step from benchmark-results.json."""

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = pathlib.Path(__file__).resolve().parent
DATA = json.loads((HERE / "benchmark-results.json").read_text())
ROWS = DATA["scaling"]

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
    }
)

figure, axis = plt.subplots(figsize=(5.6, 4.1))
colors = {"naive": "#cf5b48", "cells": "#3978b8"}
markers = {"naive": "o", "cells": "s"}
for method in ("naive", "cells"):
    n = np.array([row["n"] for row in ROWS], dtype=float)
    median = np.array([row[method]["median_s"] / 600.0 for row in ROWS])
    low = median - np.array([row[method]["min_s"] / 600.0 for row in ROWS])
    high = np.array([row[method]["max_s"] / 600.0 for row in ROWS]) - median
    axis.errorbar(
        n,
        median,
        yerr=np.vstack([low, high]),
        color=colors[method],
        marker=markers[method],
        markersize=5,
        linewidth=1.6,
        capsize=3,
        label=method,
    )

axis.set_xscale("log")
axis.set_yscale("log")
axis.set_xticks([100, 400, 1600], labels=["100", "400", "1600"])
axis.set_xlabel("atoms, N")
axis.set_ylabel("seconds per integration step")
axis.set_title("Cell lists reduce the growth of pair-search cost")
axis.grid(which="both", color="#e5e5e5", linewidth=0.7)
axis.legend(title="force path")
figure.tight_layout()
figure.savefig(HERE / "scaling.png", dpi=220, facecolor="white")
figure.savefig(HERE / "scaling.svg", facecolor="white")
print(f"wrote {HERE / 'scaling.png'} and {HERE / 'scaling.svg'}")
