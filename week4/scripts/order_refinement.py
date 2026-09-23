#!/usr/bin/env python3
"""Measure RK4 order on Taylor-Green and choose a random-flow step by Richardson."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import ARTIFACTS, EVIDENCE, make_field, run_fluid, save_json


def last_frame(out):
    frames = [
        json.loads(line)
        for line in (ARTIFACTS / out / "fields.jsonl").read_text().splitlines()
    ]
    return frames[-1]


def relative_vector_error(actual, exact):
    actual = np.asarray(actual, dtype=float)
    exact = np.asarray(exact, dtype=float)
    return float(np.linalg.norm(actual - exact) / np.linalg.norm(exact))


def run_taylor_green_order():
    initial = make_field("taylor-green", "--n", 8)
    exact = make_field("taylor-green", "--n", 8, "--nu", 0.5, "--t", 2)
    steps = [0.4, 0.25, 0.2]
    runs = []
    for dt in steps:
        name = f"order/rk4-dt{dt:g}"
        result = run_fluid(
            initial, "rk4", 0.5, dt, 2.0, 2.0, name, expect_failure=False
        )
        if result.returncode:
            raise RuntimeError(f"Taylor-Green order run failed for dt={dt}")
        final = last_frame(name)
        error = relative_vector_error(
            final["u"] + final["v"], exact["u"] + exact["v"]
        )
        runs.append({"dt": dt, "relative_velocity_error": error, "out": name})
    slope, intercept = np.polyfit(
        np.log([item["dt"] for item in runs]),
        np.log([item["relative_velocity_error"] for item in runs]),
        1,
    )
    return runs, float(slope), float(intercept)


def run_random_refinement():
    initial = make_field(
        "random", "--n", 128, "--seed", 2026, "--k-min", 2, "--k-max", 6
    )
    save_json(ARTIFACTS / "convergence" / "initial-field.json", initial)
    viscosity = 0.004
    candidates = [0.02, 0.0125, 0.01]
    reference_dt = 0.0025
    outputs = {}
    for dt in candidates + [reference_dt]:
        tag = f"dt{dt:g}"
        out = f"convergence/{tag}"
        result = run_fluid(
            initial, "rk4", viscosity, dt, 2.0, 2.0, out,
            expect_failure=False,
        )
        if result.returncode:
            raise RuntimeError(f"random refinement run failed for dt={dt}")
        outputs[dt] = (out, last_frame(out))

    reference = outputs[reference_dt][1]["omega"]
    reference_norm = np.linalg.norm(reference)
    series = []
    for dt in candidates:
        out, frame = outputs[dt]
        error = float(
            np.linalg.norm(np.asarray(frame["omega"]) - np.asarray(reference))
            / reference_norm
        )
        series.append({"dt": dt, "relative_omega_error": error, "out": out})
    slope, intercept = np.polyfit(
        np.log([item["dt"] for item in series]),
        np.log([item["relative_omega_error"] for item in series]),
        1,
    )

    dt_coarse = 0.02
    dt_fine = 0.01
    omega_coarse = np.asarray(outputs[dt_coarse][1]["omega"], dtype=float)
    omega_fine = np.asarray(outputs[dt_fine][1]["omega"], dtype=float)
    richardson_error_at_fine = float(
        np.linalg.norm(omega_coarse - omega_fine)
        / (2**4 - 1)
        / np.linalg.norm(omega_fine)
    )
    threshold = 5e-6
    predictions = []
    for item in series:
        predicted = richardson_error_at_fine * (item["dt"] / dt_fine) ** 4
        predictions.append(
            {
                "dt": item["dt"],
                "predicted_relative_error": float(predicted),
                "measured_relative_error": item["relative_omega_error"],
            }
        )
    acceptable = [item for item in predictions if item["predicted_relative_error"] < threshold]
    choice = max(acceptable, key=lambda item: item["dt"])
    return {
        "seed": 2026,
        "n": 128,
        "k_band": [2, 6],
        "nu": viscosity,
        "t_end": 2.0,
        "reference": {"dt": reference_dt, "out": outputs[reference_dt][0]},
        "runs": series,
        "slope": float(slope),
        "fit_intercept": float(intercept),
        "richardson": {
            "order": 4,
            "threshold": threshold,
            "error_at_dt_0.01": richardson_error_at_fine,
            "predictions": predictions,
            "choice": choice,
        },
    }, float(slope), float(intercept)


def plot_order(runs, slope, intercept):
    steps = np.asarray([item["dt"] for item in runs])
    errors = np.asarray([item["relative_velocity_error"] for item in runs])
    line_x = np.linspace(steps.min() * 0.95, steps.max() * 1.05, 100)
    fig, ax = plt.subplots(figsize=(6.2, 4.5), constrained_layout=True)
    ax.loglog(steps, errors, "o", markersize=9, zorder=4,
              label=f"RK4, slope {slope:.3f}")
    ax.loglog(line_x, np.exp(intercept) * line_x**slope, "--", zorder=2,
              label="log-log fit")
    ax.set(
        xlabel="time step dt", ylabel="relative velocity error at t=2",
        title="Taylor-Green temporal order, N=8, ν=0.5",
    )
    ax.grid(True, which="both", alpha=0.2)
    ax.legend()
    fig.savefig(EVIDENCE / "order.png", dpi=180)
    plt.close(fig)


def plot_convergence(data, slope, intercept):
    runs = data["runs"]
    steps = np.asarray([item["dt"] for item in runs])
    errors = np.asarray([item["relative_omega_error"] for item in runs])
    fit_x = np.linspace(steps.min() * 0.95, steps.max() * 1.08, 120)
    choice = data["richardson"]["choice"]
    fig, ax = plt.subplots(figsize=(6.8, 4.8), constrained_layout=True)
    ax.loglog(steps, errors, "o", markersize=7, label=f"measured, slope {slope:.3f}")
    ax.loglog(fit_x, np.exp(intercept) * fit_x**slope, "--", label="log-log fit")
    ax.axhline(data["richardson"]["threshold"], color="black", linestyle=":",
               label="relative error threshold 5×10⁻⁶")
    selected = choice["dt"]
    ax.scatter(
        [selected], [choice["predicted_relative_error"]],
        marker="*", s=180, color="#e63946", zorder=5,
        label=f"chosen dt={selected:g}, predicted {choice['predicted_relative_error']:.2e}",
    )
    ax.annotate(
        f"measured {choice['measured_relative_error']:.2e}",
        (selected, choice["predicted_relative_error"]),
        xytext=(8, 10), textcoords="offset points", fontsize=8,
    )
    ax.set(
        xlabel="time step dt", ylabel="relative omega error at t=2",
        title="Random-flow refinement, N=128, ν=0.004",
    )
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=8)
    fig.savefig(EVIDENCE / "convergence.png", dpi=180)
    plt.close(fig)


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    order_runs, order_slope, order_intercept = run_taylor_green_order()
    print(f"Taylor-Green RK4 fitted slope: {order_slope:.6f}")
    for item in order_runs:
        print(
            f"dt={item['dt']:g}: relative velocity error "
            f"{item['relative_velocity_error']:.9e}"
        )
    plot_order(order_runs, order_slope, order_intercept)

    data, slope, intercept = run_random_refinement()
    save_json(ARTIFACTS / "../evidence/convergence.json", data)
    print(f"Random-flow relative omega-error slope: {slope:.6f}")
    for item in data["runs"]:
        print(
            f"dt={item['dt']:g}: relative omega error "
            f"{item['relative_omega_error']:.9e}"
        )
    print(
        "Richardson error at dt=0.01: "
        f"{data['richardson']['error_at_dt_0.01']:.9e}"
    )
    print("Predictions and measured checks:")
    for item in data["richardson"]["predictions"]:
        print(
            f"dt={item['dt']:g}: predicted={item['predicted_relative_error']:.9e}, "
            f"measured={item['measured_relative_error']:.9e}"
        )
    choice = data["richardson"]["choice"]
    print(
        f"chosen dt={choice['dt']:g}, predicted={choice['predicted_relative_error']:.9e}, "
        f"measured={choice['measured_relative_error']:.9e}"
    )
    plot_convergence(data, slope, intercept)
    print(f"wrote {EVIDENCE / 'order.png'}, {EVIDENCE / 'convergence.png'}")


if __name__ == "__main__":
    main()
