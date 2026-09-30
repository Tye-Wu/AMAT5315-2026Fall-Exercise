#!/usr/bin/env python3
"""Time full LJ cluster gradients in JAX forward and reverse mode."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "ad"
SIZES = [64, 128, 256, 512, 1024]
SPACING = 2.0 ** (1.0 / 6.0)


def cluster(n: int, seed: int = 2026) -> np.ndarray:
    side = math.ceil(n ** (1.0 / 3.0))
    grid = np.array(
        [(i, j, k) for i in range(side) for j in range(side) for k in range(side)],
        dtype=np.float64,
    )[:n]
    rng = np.random.default_rng(seed)
    return (SPACING * grid + rng.normal(0.0, 0.05, size=grid.shape)).reshape(-1)


def model(n: int):
    i_np, j_np = np.triu_indices(n, k=1)
    i = jnp.asarray(i_np)
    j = jnp.asarray(j_np)

    def energy(coords: jax.Array) -> jax.Array:
        xyz = coords.reshape((n, 3))
        delta = xyz[i] - xyz[j]
        r2 = jnp.sum(delta * delta, axis=1)
        r6_inv = r2**-3
        return jnp.sum(4.0 * (r6_inv * r6_inv - r6_inv))

    def analytic_gradient(coords: np.ndarray) -> np.ndarray:
        xyz = coords.reshape((n, 3))
        delta = xyz[i_np] - xyz[j_np]
        r2 = np.sum(delta * delta, axis=1)
        inv_r = 1.0 / np.sqrt(r2)
        inv_r6 = inv_r**6
        d_u = 24.0 * (inv_r**7 - 2.0 * inv_r**13)
        pair_grad = (d_u * inv_r)[:, None] * delta
        grad = np.zeros_like(xyz)
        np.add.at(grad, i_np, pair_grad)
        np.add.at(grad, j_np, -pair_grad)
        return grad.reshape(-1)

    return energy, analytic_gradient


def seconds(function, *args) -> float:
    start = time.perf_counter()
    value = function(*args)
    jax.block_until_ready(value)
    return time.perf_counter() - start


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in SIZES:
        coords_np = cluster(n)
        coords = jnp.asarray(coords_np)
        energy, analytic_gradient = model(n)
        energy_jit = jax.jit(energy)
        reverse_jit = jax.jit(jax.grad(energy))
        forward_jit = jax.jit(lambda x, v: jax.jvp(energy, (x,), (v,)))
        p = coords_np.size
        basis = jnp.eye(p, dtype=jnp.float64)

        # Compile and synchronize each path before timing.
        energy_jit(coords).block_until_ready()
        grad = reverse_jit(coords).block_until_ready()
        forward_jit(coords, basis[0])[1].block_until_ready()
        e_times = [seconds(energy_jit, coords) for _ in range(3)]
        r_times = [seconds(reverse_jit, coords) for _ in range(3)]
        forward_start = time.perf_counter()
        for direction in range(p):
            _, tangent = forward_jit(coords, basis[direction])
            tangent.block_until_ready()
        f_time = time.perf_counter() - forward_start

        grad_np = np.asarray(grad)
        reference = analytic_gradient(coords_np)
        max_component = float(np.max(np.abs(reference)))
        rel_error = float(np.max(np.abs(grad_np - reference)) / max_component)
        e_time = float(np.median(e_times))
        r_time = float(np.median(r_times))
        row = {
            "atoms": n,
            "inputs": p,
            "energy_seconds": e_time,
            "forward_gradient_seconds": f_time,
            "reverse_gradient_seconds": r_time,
            "forward_over_energy": f_time / e_time,
            "reverse_over_energy": r_time / e_time,
            "forward_over_reverse": f_time / r_time,
            "largest_relative_gradient_error": rel_error,
        }
        rows.append(row)
        print(json.dumps(row))

    (OUT / "scaling.json").write_text(json.dumps({"seed": 2026, "spacing": SPACING, "runs": rows}, indent=2) + "\n")
    inputs = np.asarray([row["inputs"] for row in rows])
    forward_ratio = np.asarray([row["forward_over_energy"] for row in rows])
    reverse_ratio = np.asarray([row["reverse_over_energy"] for row in rows])
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.loglog(inputs, forward_ratio, "o-", color="#dc6843", lw=1.8, label="forward mode: one JVP per input")
    ax.loglog(inputs, reverse_ratio, "o-", color="#3572b0", lw=1.8, label="reverse mode: one VJP")
    ax.loglog(inputs, forward_ratio[0] * inputs / inputs[0], "--", color="#667085", lw=1.0, label="proportional to P")
    ax.set(xlabel="number of coordinate inputs P=3N", ylabel="gradient time / one energy time", title="Cluster energy gradient scaling")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "scaling.png", dpi=200, facecolor="white")
    plt.close(fig)
    print(json.dumps({"largest_forward_reverse_ratio_at_P3072": rows[-1]["forward_over_reverse"], "max_relative_errors": [x["largest_relative_gradient_error"] for x in rows]}))


if __name__ == "__main__":
    main()
