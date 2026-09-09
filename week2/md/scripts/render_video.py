#!/usr/bin/env python3
"""Two-panel movie from a `md` run: atoms (left) and rolling g(r) (right).

Usage: python3 render_video.py <run-dir> <out.mp4>

Reads run.json and traj.jsonl, draws each saved frame, and encodes the movie
with matplotlib's ffmpeg writer. The g(r) panel shows a rolling average over
the most recent frames so the structure settles as the liquid equilibrates.
"""

import json
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

RUN_DIR, OUT = sys.argv[1], sys.argv[2]

meta = json.load(open(f"{RUN_DIR}/run.json"))
frames = [json.loads(l) for l in open(f"{RUN_DIR}/traj.jsonl") if l.strip()]
L = meta["box"]
N = meta["atoms"]
rc = min(meta["cutoff"], L / 2.0)
rho = N / (L * L)

NB = 80
EDGES = np.linspace(0.0, rc, NB + 1)
R = 0.5 * (EDGES[:-1] + EDGES[1:])
SHELL = rho * 2.0 * np.pi * R * np.diff(EDGES)  # expected neighbours per atom


def hist_frame(f):
    """Per-frame histogram of ordered minimum-image pair distances (r < rc)."""
    x = np.asarray(f["x"], dtype=float)
    y = np.asarray(f["y"], dtype=float)
    h = np.zeros(NB)
    for i in range(N):
        dx = x - x[i]
        dy = y - y[i]
        dx -= L * np.round(dx / L)
        dy -= L * np.round(dy / L)
        r = np.sqrt(dx * dx + dy * dy)
        r[i] = np.inf
        inside = r < rc
        if inside.any():
            h += np.histogram(r[inside], bins=EDGES)[0]
    return h


print("computing per-frame histograms ...")
H = np.array([hist_frame(f) for f in frames])
NF = len(frames)
WINDOW = 25


def g_rolling(i):
    lo = max(0, i - WINDOW + 1)
    counts = H[lo:i + 1].sum(axis=0)
    return counts / ((i - lo + 1) * N * SHELL)


fig, (ax_a, ax_g) = plt.subplots(1, 2, figsize=(9.6, 4.6))
fig.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.13)


def draw(i):
    ax_a.cla()
    ax_g.cla()
    f = frames[i]
    ax_a.scatter(f["x"], f["y"], s=14, color="#2a78d6", edgecolors="none")
    ax_a.set_xlim(0, L)
    ax_a.set_ylim(0, L)
    ax_a.set_aspect("equal")
    ax_a.set_title(f"t = {f['t']:.2f}   (frame {i} / {NF})")
    g = g_rolling(i)
    ax_g.plot(R, g, color="#eb6834", lw=1.2)
    ax_g.axhline(1.0, color="#b9b6ac", lw=0.8)
    ax_g.set_xlim(0, rc)
    ax_g.set_ylim(0.0, min(6.0, max(0.5, g.max() * 1.3)))
    ax_g.set_title("g(r), rolling window")
    ax_g.set_xlabel("r ($\\sigma$)")
    ax_g.set_ylabel("g(r)")


print("encoding ...")
writer = FFMpegWriter(fps=15)
with writer.saving(fig, OUT, dpi=110):
    for i in range(NF):
        draw(i)
        writer.grab_frame()
print(f"wrote {OUT} ({NF} frames)")
