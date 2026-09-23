#!/usr/bin/env python3
"""Run the Rust line experiments and make both Part 1 verification figures."""

import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


HERE = Path(__file__).resolve().parent
WEEK4 = HERE.parent
ROOT = WEEK4.parent
ARTIFACTS = WEEK4 / "artifacts" / "line-study"
EVIDENCE = WEEK4 / "evidence"


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["cargo", "run", "--offline", "--release", "--example", "line_study"],
        cwd=WEEK4,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr or result.stdout)
    data = json.loads(result.stdout)
    (ARTIFACTS / "line-study.json").write_text(json.dumps(data))
    plot_stability(data)
    plot_accuracy(data)
    print(f"RK4 line stability limit from all Fourier modes: {line_stability_limit():.6f}")


def rk4_growth(z):
    return abs(1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24)


def line_stability_limit():
    modes = np.arange(-32, 32)
    lam = -0.05 * modes**2 - 1j * modes
    lam[modes == -32] = -0.05 * 32**2

    def stable(dt):
        return np.all(np.asarray([rk4_growth(value * dt) for value in lam]) <= 1)

    low, high = 0.0, 0.1
    for _ in range(60):
        mid = (low + high) / 2
        if stable(mid):
            low = mid
        else:
            high = mid
    return (low + high) / 2


def plot_stability(data):
    stability = data["stability"]
    real = np.asarray(stability["real"])
    imag = np.asarray(stability["imag"])
    growth = np.asarray(stability["measured_rk4"])
    rr, ii = np.meshgrid(real, imag)
    z = rr + 1j * ii
    r_euler = np.abs(1 + z)
    r_midpoint = np.abs(1 + z + z**2 / 2)
    r_rk4 = np.abs(1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)
    ax = axes[0]
    im = ax.pcolormesh(
        real, imag, growth, shading="auto", norm=LogNorm(vmin=1e-3, vmax=1e3),
        cmap="coolwarm"
    )
    ax.contour(real, imag, r_rk4, levels=[1], colors="black", linewidths=2.0)
    ax.contour(real, imag, r_euler, levels=[1], colors="#f5d742", linestyles="--")
    ax.contour(real, imag, r_midpoint, levels=[1], colors="white", linestyles=":")
    for mode in stability["mode_points"]:
        points = np.asarray(mode["points"])
        color = "#21a179" if np.isclose(mode["dt"], 0.045) else "#ff4d6d"
        ax.scatter(points[:, 0], points[:, 1], s=8, color=color, alpha=0.7,
                   label=f"line modes, dt={mode['dt']:.3f}")
    ax.set(xlim=(-4, 0.8), ylim=(-4, 4), xlabel="Re(z)", ylabel="Im(z)",
           title="Measured RK4 growth per step")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=7, loc="lower left")
    fig.colorbar(im, ax=ax, label="|y₁| / |y₀|, log scale")

    x = np.arange(64) * 2 * np.pi / 64
    for ax, key, title in zip(
        axes[1:],
        ("stable_pulse", "unstable_pulse"),
        ("RK4, dt = 0.045", "RK4, dt = 0.056"),
    ):
        pulse = data[key]
        values = np.asarray(pulse["values"])
        times = np.asarray(pulse["times"])
        ax.imshow(
            values,
            origin="upper",
            aspect="auto",
            extent=(0, 2 * np.pi, times[-1], times[0]),
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            interpolation="nearest",
        )
        ax.set(xlim=(0, 2 * np.pi), ylim=(6, 0), xlabel="x", title=title)
        ax.set_ylabel("time t (downward)")
    fig.suptitle("Stability region and a periodic pulse below and above dt = 0.0494")
    fig.savefig(EVIDENCE / "line-stability.png", dpi=180)
    plt.close(fig)


def plot_accuracy(data):
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2), constrained_layout=True)
    x = np.asarray(data["accuracy_x"])
    exact = np.asarray(data["accuracy_exact"])
    profiles = data["accuracy_profiles"]
    labels = (
        "RK4, Fourier, dt=0.02",
        "RK4, centered, dt=0.02",
        "Euler, Fourier, dt=0.005",
    )
    ax = axes[0]
    ax.plot(x, exact, color="black", linewidth=2.2, label="exact")
    max_errors = []
    for label, profile in zip(labels, profiles):
        profile = np.asarray(profile)
        error = np.max(np.abs(profile - exact))
        max_errors.append(error)
        ax.plot(x, profile, linewidth=1.2, label=label)
        print(f"{label}: maximum absolute error = {error:.9e}")
    ax.set(xlabel="x", ylabel="u(x, 2π)", title="One lap of the pulse")
    ax.legend(fontsize=7)

    ax = axes[1]
    for series in data["error_series"]:
        steps = np.asarray(series["steps"])
        errors = np.asarray(series["errors"])
        slope, intercept = np.polyfit(np.log(steps), np.log(errors), 1)
        ax.loglog(steps, errors, "o", label=f"{series['method']}, slope {slope:.2f}")
        ax.loglog(steps, np.exp(intercept) * steps**slope, "-", linewidth=1.0)
        print(f"{series['method']}: fitted error slope = {slope:.4f}")
    ax.set(xlabel="time step dt", ylabel="maximum error at t=1",
           title="Time-integration accuracy")
    ax.legend(fontsize=8)
    fig.suptitle("Spatial error and time-step convergence")
    fig.savefig(EVIDENCE / "line-accuracy.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
