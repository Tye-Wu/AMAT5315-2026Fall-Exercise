#!/usr/bin/env python3
"""Add the prescribed vorticity ripple, measure sensitivity, and make Part 3 plots."""

import copy
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import ARTIFACTS, EVIDENCE, make_field, read_tsv, run_fluid, save_json


RIPPLE = 7e-5


def perturbed_velocity(field):
    altered = copy.deepcopy(field)
    n = field["n"]
    largest_component = max(
        max(abs(value) for value in field["u"]),
        max(abs(value) for value in field["v"]),
    )
    amplitude = RIPPLE * largest_component
    for j in range(n):
        y = 2 * np.pi * j / n
        for i in range(n):
            x = 2 * np.pi * i / n
            index = j * n + i
            altered["u"][index] += (4 * amplitude / 25) * math.cos(3 * x) * math.sin(4 * y)
            altered["v"][index] -= (3 * amplitude / 25) * math.sin(3 * x) * math.cos(4 * y)
    return altered, largest_component


def run_pair(case, field, viscosity):
    changed, largest_component = perturbed_velocity(field)
    base_name = f"sensitivity/{case}/base"
    changed_name = f"sensitivity/{case}/perturbed"
    base_run = run_fluid(
        field, "rk4", viscosity, 0.01, 20.0, 0.5, base_name,
        expect_failure=False,
    )
    perturbed_run = run_fluid(
        changed, "rk4", viscosity, 0.01, 20.0, 0.5, changed_name,
        expect_failure=False,
    )
    if base_run.returncode or perturbed_run.returncode:
        raise RuntimeError(f"sensitivity pair {case} did not reach t=20")

    base_frames = [
        json.loads(line)
        for line in (ARTIFACTS / base_name / "fields.jsonl").read_text().splitlines()
    ]
    changed_frames = [
        json.loads(line)
        for line in (ARTIFACTS / changed_name / "fields.jsonl").read_text().splitlines()
    ]
    times = []
    relative = []
    for base, changed_frame in zip(base_frames, changed_frames):
        times.append(base["t"])
        base_omega = np.asarray(base["omega"], dtype=float)
        difference = np.asarray(changed_frame["omega"], dtype=float) - base_omega
        relative.append(float(np.linalg.norm(difference) / np.linalg.norm(base_omega)))
    return {
        "case": case,
        "times": times,
        "relative_difference": relative,
        "largest_component": largest_component,
        "base_name": base_name,
        "perturbed_name": changed_name,
    }


def check_initial_ripple(result):
    base_dir = ARTIFACTS / result["base_name"]
    changed_dir = ARTIFACTS / result["perturbed_name"]
    base_field = json.loads((base_dir / "initial-field.json").read_text())
    n = base_field["n"]
    largest_component = max(
        max(abs(value) for value in base_field["u"]),
        max(abs(value) for value in base_field["v"]),
    )
    base = json.loads((base_dir / "fields.jsonl").read_text().splitlines()[0])
    changed = json.loads((changed_dir / "fields.jsonl").read_text().splitlines()[0])
    x = 2 * np.pi * np.arange(n) / n
    xx, yy = np.meshgrid(x, x)
    expected = -RIPPLE * largest_component * np.cos(3 * xx) * np.cos(4 * yy)
    actual = (
        np.asarray(changed["omega"], dtype=float)
        - np.asarray(base["omega"], dtype=float)
    ).reshape(n, n)
    absolute_error = float(np.max(np.abs(actual - expected)))
    relative_error = float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
    return {
        "largest_component": largest_component,
        "max_abs_error_after_six_decimal_storage": absolute_error,
        "relative_l2_error_after_six_decimal_storage": relative_error,
    }


def plot_sensitivity(results):
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    for result in results:
        values = np.asarray(result["relative_difference"], dtype=float)
        label = result["case"].replace("-", " ")
        if result["case"] == "taylor-green":
            values = np.maximum(values, 5e-6)
            label += " (6-decimal storage floor)"
        ax.semilogy(
            result["times"], values, label=label,
        )
    ax.axhline(5e-6, color="#555555", linestyle=":", linewidth=1,
               label="display floor = 5×10⁻⁶")
    ax.set(
        xlabel="time t", ylabel="relative vorticity difference",
        title="Physical sensitivity at a stable RK4 step",
        xlim=(0, 20),
    )
    ax.grid(True, which="both", alpha=0.2)
    ax.legend()
    fig.savefig(EVIDENCE / "sensitivity.png", dpi=180)
    plt.close(fig)


def energy_panel(ax, series, label, color, linestyle="-"):
    times = np.asarray([row[0] for row in series])
    energies = np.asarray([row[1] for row in series])
    valid = np.isfinite(energies) & (energies > 0)
    ax.semilogy(times[valid], energies[valid], color=color, linestyle=linestyle,
                linewidth=1.6, label=label)
    stop = times[-1] if times.size else float("nan")
    if times.size and not np.isfinite(series[-1][1]):
        ax.axvline(stop, color=color, linestyle=":", alpha=0.7)
        ax.annotate(f"stop {stop:.2f}", (stop, ax.get_ylim()[0]),
                    xytext=(4, 5), textcoords="offset points", rotation=90,
                    color=color, fontsize=7)


def plot_blowup_and_sensitivity(results):
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.1), constrained_layout=True)
    tg_axis, random_axis, sensitivity_axis = axes
    for dt, color in ((0.032, "#0077b6"), (0.033, "#d62828")):
        data = read_tsv(ARTIFACTS / f"scan/taylor-green-rk4-dt{dt:.3f}.tsv")
        energy_panel(tg_axis, data, f"RK4, dt={dt:.3f}", color)
    t = np.linspace(0, 8, 401)
    tg_axis.semilogy(t, 0.25 * np.exp(-0.4 * t), "k--", label="exact energy")
    tg_axis.set(xlabel="time t", ylabel="kinetic energy E",
                title="Taylor-Green, N=64, ν=0.1", xlim=(0, 8))
    tg_axis.legend(fontsize=7)

    summary = json.loads((ARTIFACTS / "scan-summary.json").read_text())
    low, high = summary["random_rk4_bracket"]
    energy_panel(
        random_axis,
        read_tsv(ARTIFACTS / f"scan/random-rk4-dt{low:.3f}.tsv"),
        f"RK4, dt={low:.3f}", "#0077b6",
    )
    energy_panel(
        random_axis,
        read_tsv(ARTIFACTS / f"scan/random-rk4-dt{high:.3f}.tsv"),
        f"RK4, dt={high:.3f}", "#d62828",
    )
    energy_panel(
        random_axis,
        read_tsv(ARTIFACTS / "scan/random-euler-dt0.010.tsv"),
        "Euler, dt=0.010", "#f4a261", "--",
    )
    random_axis.set(
        xlabel="time t", ylabel="kinetic energy E",
        title="Random flow, N=128, ν=0.004", xlim=(0, 10),
    )
    random_axis.legend(fontsize=7)

    for result in results:
        values = np.asarray(result["relative_difference"], dtype=float)
        label = result["case"].replace("-", " ")
        if result["case"] == "taylor-green":
            values = np.maximum(values, 5e-6)
            label += " (storage floor)"
        sensitivity_axis.semilogy(
            result["times"], values, label=label,
        )
    sensitivity_axis.axhline(
        5e-6, color="#555555", linestyle=":", linewidth=1,
        label="display floor = 5×10⁻⁶",
    )
    sensitivity_axis.set(
        xlabel="time t", ylabel="relative vorticity difference",
        title="Perturbation response", xlim=(0, 20),
    )
    sensitivity_axis.legend(fontsize=8)
    for ax in axes:
        ax.grid(True, which="both", alpha=0.18)
    fig.suptitle("Numerical instability and physical sensitivity")
    fig.savefig(EVIDENCE / "blowup.png", dpi=180)
    plt.close(fig)


def main():
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    summary_path = ARTIFACTS / "sensitivity-summary.json"
    if "--plots-only" in sys.argv:
        results = json.loads(summary_path.read_text())
    else:
        green = make_field("taylor-green", "--n", 64)
        random = make_field(
            "random", "--n", 128, "--seed", 2026, "--k-min", 2, "--k-max", 6
        )
        results = [
            run_pair("taylor-green", green, 0.1),
            run_pair("random", random, 0.004),
        ]
    for result in results:
        result["initial_ripple_check"] = check_initial_ripple(result)
        check = result["initial_ripple_check"]
        print(
            f"{result['case']} initial ripple: max |error| after six-decimal "
            f"storage = {check['max_abs_error_after_six_decimal_storage']:.3e}, "
            f"relative L2 error = {check['relative_l2_error_after_six_decimal_storage']:.3e}"
        )
    save_json(summary_path, results)
    for result in results:
        ratios = np.asarray(result["relative_difference"])
        print(
            f"{result['case']}: initial separation {ratios[0]:.6e}, "
            f"maximum {ratios.max():.6e}, final {ratios[-1]:.6e}"
        )
    random_growth = (
        results[1]["relative_difference"][-1]
        / results[1]["relative_difference"][0]
    )
    green_growth = (
        results[0]["relative_difference"][-1]
        / results[0]["relative_difference"][0]
    )
    print(f"random perturbation amplification at t=20: {random_growth:.4g}x")
    print(f"Taylor-Green relative perturbation factor at t=20: {green_growth:.4g}x")
    plot_sensitivity(results)
    plot_blowup_and_sensitivity(results)
    print(f"wrote {EVIDENCE / 'sensitivity.png'} and {EVIDENCE / 'blowup.png'}")


if __name__ == "__main__":
    main()
