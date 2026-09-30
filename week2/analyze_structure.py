#!/usr/bin/env python3
"""Recompute RDF and melting observables from cold, hot, and ramp trajectories."""

import json
import pathlib

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
BINS = 48


def load(directory):
    directory = pathlib.Path(directory)
    meta = json.loads((directory / "run.json").read_text())
    frames = [json.loads(line) for line in (directory / "traj.jsonl").read_text().splitlines() if line]
    return meta, frames


def frame_gr(frame, meta):
    pos = np.asarray(frame["pos"], dtype=float)
    lx, ly = meta["box"]
    rmax = min(lx, ly) / 2.0
    delta = pos[:, None, :] - pos[None, :, :]
    delta[:, :, 0] -= lx * np.round(delta[:, :, 0] / lx)
    delta[:, :, 1] -= ly * np.round(delta[:, :, 1] / ly)
    distance = np.sqrt(np.sum(delta * delta, axis=2))
    upper = distance[np.triu_indices(meta["n"], 1)]
    count, edges = np.histogram(upper[upper < rmax], bins=BINS, range=(0.0, rmax))
    rho = meta["n"] / (lx * ly)
    shell = rho * np.pi * (edges[1:] ** 2 - edges[:-1] ** 2) * meta["n"] / 2.0
    radii = 0.5 * (edges[:-1] + edges[1:])
    return radii, count / shell


def trajectory_gr(meta, frames):
    rows = [frame_gr(frame, meta)[1] for frame in frames]
    return frame_gr(frames[0], meta)[0], np.asarray(rows)


def speed_temperature(frame):
    vel = np.asarray(frame["vel"], dtype=float)
    return float(np.sum(vel * vel) / (2.0 * len(vel)))


def contrast(radii, g):
    selected = g[radii > 2.0]
    return float(np.sqrt(np.mean((selected - 1.0) ** 2)))


def rolling_contrast(radii, rows, window=10):
    result = []
    for index in range(len(rows)):
        start = max(0, index - window + 1)
        result.append(contrast(radii, rows[start : index + 1].mean(axis=0)))
    return np.asarray(result)


def main():
    cold_meta, cold_frames = load("/tmp/week2-cold")
    hot_meta, hot_frames = load("/tmp/week2-hot")
    ramp_meta, ramp_frames = load(ROOT / "docs")
    radii, cold_rows = trajectory_gr(cold_meta, cold_frames)
    _, hot_rows = trajectory_gr(hot_meta, hot_frames)
    ramp_radii, ramp_rows = trajectory_gr(ramp_meta, ramp_frames)
    ramp_temperature = np.asarray([speed_temperature(frame) for frame in ramp_frames])
    ramp_contrast = rolling_contrast(ramp_radii, ramp_rows)
    times = np.asarray([frame["t"] for frame in ramp_frames])

    result = {
        "rdf_bins": BINS,
        "cold_temperature_mean": float(np.mean([speed_temperature(f) for f in cold_frames])),
        "hot_temperature_mean": float(np.mean([speed_temperature(f) for f in hot_frames])),
        "cold_long_range_contrast": contrast(radii, cold_rows.mean(axis=0)),
        "hot_long_range_contrast": contrast(radii, hot_rows.mean(axis=0)),
        "ramp_temperature_first": float(ramp_temperature[0]),
        "ramp_temperature_last": float(ramp_temperature[-1]),
        "ramp_contrast_initial_10": contrast(ramp_radii, ramp_rows[:10].mean(axis=0)),
        "ramp_contrast_final_10": contrast(ramp_radii, ramp_rows[-10:].mean(axis=0)),
        "radii": radii.tolist(),
        "cold_g": cold_rows.mean(axis=0).tolist(),
        "hot_g": hot_rows.mean(axis=0).tolist(),
        "ramp_time": times.tolist(),
        "ramp_temperature": ramp_temperature.tolist(),
        "ramp_rolling_contrast": ramp_contrast.tolist(),
    }
    (HERE / "structure-results.json").write_text(json.dumps(result, indent=2) + "\n")

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
    figure, (rdf_axis, ramp_axis) = plt.subplots(1, 2, figsize=(9.2, 3.8))
    rdf_axis.plot(radii, cold_rows.mean(axis=0), color="#3978b8", lw=1.7, label="T=0.2 cold")
    rdf_axis.plot(radii, hot_rows.mean(axis=0), color="#cf5b48", lw=1.7, label="T=1.0 hot")
    rdf_axis.axhline(1.0, color="#888888", lw=0.8)
    rdf_axis.set(xlabel="pair distance, r", ylabel="g(r)", title="Heating removes distant neighbour shells")
    rdf_axis.set_xlim(0, radii[-1])
    rdf_axis.legend()
    rdf_axis.grid(color="#e5e5e5", linewidth=0.7)

    contrast_axis = ramp_axis.twinx()
    ramp_axis.plot(times, ramp_temperature, color="#cf5b48", lw=1.6, label="temperature")
    contrast_axis.plot(times, ramp_contrast, color="#3978b8", lw=1.6, label="long-range contrast")
    ramp_axis.set(xlabel="production time", ylabel="T", title="400-atom ramp: order falls as T rises")
    contrast_axis.set_ylabel("RMS[g(r)-1], r>2")
    ramp_axis.grid(color="#e5e5e5", linewidth=0.7)
    lines = ramp_axis.lines + contrast_axis.lines
    ramp_axis.legend(lines, [line.get_label() for line in lines], loc="center right")
    figure.tight_layout()
    figure.savefig(HERE / "melting.png", dpi=220, facecolor="white")
    figure.savefig(HERE / "melting.svg", facecolor="white")
    print(json.dumps({key: value for key, value in result.items() if not isinstance(value, list)}, indent=2))


if __name__ == "__main__":
    main()
