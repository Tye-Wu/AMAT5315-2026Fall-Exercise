from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "inputs" / "reflector.json"
OUT = ROOT / "artifacts"


def main() -> None:
    e = json.loads(INPUT.read_text())
    OUT.mkdir(exist_ok=True)
    dx_km = e["dx"] * e["length_unit_m"] / 1000
    time_s = e["time_unit_s"]
    bg = np.asarray(e["background"])
    dv = np.asarray(e["perturbation"])
    extent = [0, e["nx"] * dx_km, e["nz"] * dx_km, 0]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    im = ax[0].imshow(bg, extent=extent, aspect="equal", cmap="viridis")
    fig.colorbar(im, ax=ax[0], label="Background speed (km/s)")
    im2 = ax[1].imshow(dv, extent=extent, aspect="equal", cmap="RdBu_r")
    fig.colorbar(im2, ax=ax[1], label="Velocity perturbation (km/s)")
    for x, z in e["shots"]:
        ax[1].plot(x * dx_km, z * dx_km, "r*", ms=10)
    for x, z in e["receivers"]:
        ax[1].plot(x * dx_km, z * dx_km, "kv", ms=3)
    for a in ax[:2]:
        a.set(xlabel="Horizontal position (km)", ylabel="Depth (km)")
        a.invert_yaxis()
        a.grid(alpha=.15)
    t = np.arange(e["steps"]) * e["dt"] * time_s
    theta = np.pi * e["source_frequency"] * (np.arange(e["steps"]) * e["dt"] - e["source_peak_time"])
    pulse = e["source_amplitude"] * (1 - 2 * theta**2) * np.exp(-theta**2)
    ax[2].plot(t, pulse, color="black")
    ax[2].axvline(e["source_peak_time"] * time_s, color="tab:red", ls="--", label="peak time")
    ax[2].set(xlabel="Time (s)", ylabel="Source amplitude", title="Ricker source")
    ax[2].legend(frameon=False)
    fig.savefig(OUT / "inputs.png", dpi=180)
    plt.close(fig)

    run = json.loads((OUT / "forward" / "result.json").read_text())
    traces = np.load(OUT / "forward" / "traces.npy")
    fig, axes = plt.subplots(1, len(e["shots"]), figsize=(13, 4), sharey=True, constrained_layout=True)
    vmax = np.max(np.abs(traces))
    for s, a in enumerate(axes):
        a.imshow(traces[s], aspect="auto", origin="upper", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                 extent=[e["receivers"][0][0] * dx_km, e["receivers"][-1][0] * dx_km,
                         e["steps"] * e["dt"] * time_s, 0])
        a.set(title=f"Shot {s}; source x={e['shots'][s][0] * dx_km:.1f} km", xlabel="Receiver position (km)")
    axes[0].set_ylabel("Time (s), increasing downward")
    fig.colorbar(plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(-vmax, vmax)), ax=axes, label="Pressure")
    fig.savefig(OUT / "forward" / "gathers.png", dpi=180)
    plt.close(fig)

    wf = np.load(OUT / "forward" / "wavefield.npy")
    echo = np.load(OUT / "forward" / "echo.npy")
    frame = run["recording"]["steps"].index(150)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), constrained_layout=True)
    for a, arr, title in zip(axes, [wf[frame], echo[frame]], ["Background wavefield; step 150", "Reflector echo; step 150"]):
        lim = np.max(np.abs(arr))
        im = a.imshow(arr, extent=extent, aspect="equal", cmap="RdBu_r", vmin=-lim, vmax=lim)
        a.set(title=title, xlabel="Horizontal position (km)", ylabel="Depth (km)")
        a.invert_yaxis()
        fig.colorbar(im, ax=a, label="Pressure")
    direct = float(np.max(np.abs(wf[frame])))
    scattered = float(np.max(np.abs(echo[frame])))
    fig.suptitle(f"Echo/direct maximum = {scattered / direct:.3%}")
    fig.savefig(OUT / "forward" / "wavefield-echo-step150.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"step150_direct_max": direct, "step150_echo_max": scattered,
                      "step150_echo_fraction": scattered / direct,
                      "reference_trace_l2": run["trace_l2"]}, indent=2))


if __name__ == "__main__":
    main()
