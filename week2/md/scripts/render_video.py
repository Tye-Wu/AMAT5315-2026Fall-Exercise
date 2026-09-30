#!/usr/bin/env python3
"""Render saved atoms beside a rolling radial distribution function."""

import json
import pathlib
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter


run_dir = pathlib.Path(sys.argv[1])
output = pathlib.Path(sys.argv[2])
meta = json.loads((run_dir / "run.json").read_text())
frames = [json.loads(line) for line in (run_dir / "traj.jsonl").read_text().splitlines() if line]
n = meta["n"]
lx, ly = meta["box"]
rho = meta["rho"]
rmax = 0.5 * min(lx, ly)

bins = 80
edges = np.linspace(0.0, rmax, bins + 1)
radii = 0.5 * (edges[:-1] + edges[1:])
shell_area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)


def distance_counts(frame):
    pos = np.asarray(frame["pos"], dtype=float)
    counts = np.zeros(bins)
    for i in range(n):
        delta = pos - pos[i]
        delta[:, 0] -= lx * np.round(delta[:, 0] / lx)
        delta[:, 1] -= ly * np.round(delta[:, 1] / ly)
        distance = np.hypot(delta[:, 0], delta[:, 1])
        distance[i] = np.inf
        counts += np.histogram(distance[distance < rmax], bins=edges)[0]
    return counts


histograms = np.asarray([distance_counts(frame) for frame in frames])


def rolling_g(index, window=25):
    first = max(0, index - window + 1)
    count = histograms[first : index + 1].sum(axis=0)
    return count / ((index - first + 1) * n * rho * shell_area)


figure, (atoms, structure) = plt.subplots(1, 2, figsize=(9.2, 4.4))
figure.subplots_adjust(left=0.07, right=0.98, top=0.89, bottom=0.14, wspace=0.26)


def draw(index):
    atoms.clear()
    structure.clear()
    frame = frames[index]
    pos = np.asarray(frame["pos"], dtype=float)
    speeds = np.linalg.norm(np.asarray(frame["vel"], dtype=float), axis=1)
    atoms.scatter(pos[:, 0], pos[:, 1], c=speeds, s=14, cmap="viridis", vmin=0, vmax=2.5)
    atoms.set(xlim=(0, lx), ylim=(0, ly), xlabel="x", ylabel="y")
    atoms.set_aspect("equal")
    atoms.set_title(f"step {frame['step']}   t={frame['t']:.2f}")
    g = rolling_g(index)
    structure.plot(radii, g, color="#eb6834", lw=1.4)
    structure.axhline(1.0, color="#888888", lw=0.8)
    structure.set(xlim=(0, rmax), ylim=(0, max(5.5, float(g.max()) * 1.1)), xlabel="r", ylabel="g(r)")
    structure.set_title("pair structure (recent frames)")


writer = FFMpegWriter(
    fps=15,
    codec="libx264",
    bitrate=700,
    extra_args=[
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ],
)
with writer.saving(figure, output, dpi=100):
    for index in range(len(frames)):
        draw(index)
        writer.grab_frame()
print(f"wrote {output} ({len(frames)} frames)")
