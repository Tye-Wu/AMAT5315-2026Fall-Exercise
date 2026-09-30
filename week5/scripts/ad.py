#!/usr/bin/env python3
"""Hand-compose JAX forward and reverse passes for one LJ pair."""

from __future__ import annotations

import json
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


def pair_energy(r: jax.Array) -> jax.Array:
    a = r**-6
    b = a**2
    c = b - a
    return 4.0 * c


def forward_pass(r: jax.Array) -> tuple[jax.Array, dict[str, jax.Array]]:
    """One local JVP per node, including both inputs to the subtraction."""
    one = jnp.ones_like(r)
    a, da = jax.jvp(lambda x: x**-6, (r,), (one,))
    b, db = jax.jvp(lambda x: x**2, (a,), (da,))
    c, dc = jax.jvp(lambda x, y: x - y, (b, a), (db, da))
    energy, denergy = jax.jvp(lambda x: 4.0 * x, (c,), (dc,))
    return energy, {"r": one, "a": da, "b": db, "c": dc, "U": denergy}


def reverse_pass(r: jax.Array) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Forward-evaluate nodes, then compose local VJPs in reverse order."""
    a, pull_a = jax.vjp(lambda x: x**-6, r)
    b, pull_b = jax.vjp(lambda x: x**2, a)
    c, pull_c = jax.vjp(lambda x, y: x - y, b, a)
    energy, pull_u = jax.vjp(lambda x: 4.0 * x, c)

    (bar_c,) = pull_u(jnp.ones_like(energy))
    (bar_b, bar_a_from_c) = pull_c(bar_c)
    (bar_a_from_b,) = pull_b(bar_b)
    bar_a = bar_a_from_c + bar_a_from_b
    (bar_r,) = pull_a(bar_a)
    return energy, {
        "U": jnp.ones_like(energy),
        "c": bar_c,
        "b": bar_b,
        "a": bar_a,
        "r": bar_r,
    }


def draw_jaxpr_graph(closed: jax.core.ClosedJaxpr, path: Path, title: str) -> None:
    """Draw one operation node per JAX primitive in a traced graph."""
    jaxpr = closed.jaxpr
    producer: dict[object, str] = {var: "input" for var in jaxpr.invars}
    labels: dict[str, str] = {"input": "r"}
    ranks: dict[str, int] = {"input": 0}
    edges: list[tuple[str, str]] = []
    for index, equation in enumerate(jaxpr.eqns):
        name = f"op{index}"
        params = equation.params
        label = equation.primitive.name
        if equation.primitive.name == "integer_pow" and "y" in params:
            label += f"[y={params['y']}]"
        labels[name] = label
        parent_nodes: list[str] = []
        for variable in equation.invars:
            try:
                source = producer.get(variable)
            except TypeError:  # JAX Literal constants are not hashable.
                source = None
            if source is not None:
                parent_nodes.append(source)
        rank = 1 + max((ranks[node] for node in parent_nodes), default=0)
        ranks[name] = rank
        for source in parent_nodes:
            edges.append((source, name))
        for variable in equation.outvars:
            producer[variable] = name

    output_nodes = []
    for i, variable in enumerate(jaxpr.outvars):
        name = f"output{i}"
        labels[name] = "output" if len(jaxpr.outvars) == 1 else f"output {i}"
        try:
            source = producer.get(variable, "input")
        except TypeError:
            source = "input"
        ranks[name] = ranks[source] + 1
        edges.append((source, name))
        output_nodes.append(name)

    columns: dict[int, list[str]] = {}
    for name, rank in ranks.items():
        columns.setdefault(rank, []).append(name)
    positions: dict[str, tuple[float, float]] = {}
    for x, names in columns.items():
        names.sort(key=lambda node: (node.startswith("output"), node))
        for index, name in enumerate(names):
            positions[name] = (float(x), -(index - (len(names) - 1) / 2.0) * 1.25)

    fig, ax = plt.subplots(figsize=(max(7.0, len(columns) * 1.55), max(3.2, len(labels) * 0.37)))
    for source, target in edges:
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        ax.annotate(
            "",
            xy=(x1 - 0.27, y1),
            xytext=(x0 + 0.27, y0),
            arrowprops={"arrowstyle": "->", "color": "#667085", "lw": 1.1, "shrinkA": 0, "shrinkB": 0},
        )
    for name, (x, y) in positions.items():
        is_op = name.startswith("op")
        fc = "#dcecff" if is_op else ("#f5f1e8" if name == "input" else "#e4f3e8")
        ax.text(
            x,
            y,
            labels[name],
            ha="center",
            va="center",
            fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.38", "facecolor": fc, "edgecolor": "#475467", "linewidth": 0.8},
            zorder=3,
        )
    ax.set_title(title, fontsize=12)
    ax.set_xlim(-0.8, max(columns) + 0.8)
    ax.set_ylim(min(y for _, y in positions.values()) - 0.8, max(y for _, y in positions.values()) + 0.8)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    r0 = jnp.asarray(1.3, dtype=jnp.float64)
    energy, tangents = forward_pass(r0)
    _, adjoints = reverse_pass(r0)
    jax_grad = jax.grad(pair_energy)(r0)
    record = {
        "r": float(r0),
        "energy": float(energy),
        "tangents": {key: float(value) for key, value in tangents.items()},
        "adjoints": {key: float(value) for key, value in adjoints.items()},
        "jax_grad": float(jax_grad),
    }
    (OUT / "derivatives.json").write_text(json.dumps(record, indent=2) + "\n")

    r = jnp.linspace(0.95, 2.5, 601, dtype=jnp.float64)
    energy_fn = pair_energy
    forward = jax.jit(jax.vmap(lambda x: forward_pass(x)[1]["U"]))(r)
    reverse = jax.jit(jax.vmap(lambda x: reverse_pass(x)[1]["r"]))(r)
    analytic = 24.0 * (r**-7 - 2.0 * r**-13)
    h = 1e-6
    finite = (energy_fn(r + h) - energy_fn(r - h)) / (2.0 * h)
    error_f = np.abs(np.asarray(forward - analytic))
    error_r = np.abs(np.asarray(reverse - analytic))
    error_fd = np.abs(np.asarray(finite - analytic))

    r_np = np.asarray(r)
    analytic_np = np.asarray(analytic)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9))
    axes[0].plot(r_np, analytic_np, color="#101828", lw=1.8, label="analytic dU/dr")
    axes[0].plot(r_np, np.asarray(forward), color="#3572b0", lw=1.0, ls="--", label="hand forward JAX")
    axes[0].plot(r_np, np.asarray(reverse), color="#dc6843", lw=0.9, ls=":", label="hand reverse JAX")
    axes[0].plot(r_np, np.asarray(finite), color="#799d59", lw=0.8, alpha=0.8, label="centered FD, h=1e-6")
    axes[0].axhline(0.0, color="#98a2b3", lw=0.7)
    axes[0].axvline(2.0 ** (1.0 / 6.0), color="#98a2b3", lw=0.7, ls="--")
    axes[0].set(xlabel="separation r (sigma)", ylabel="dU/dr (epsilon/sigma)", title="Lennard-Jones derivative")
    axes[0].legend(frameon=False, fontsize=7.5)
    axes[0].grid(alpha=0.2)
    axes[1].semilogy(r_np, np.maximum(error_f, 1e-18), color="#3572b0", lw=1.0, label="forward AD")
    axes[1].semilogy(r_np, np.maximum(error_r, 1e-18), color="#dc6843", lw=0.9, label="reverse AD")
    axes[1].semilogy(r_np, np.maximum(error_fd, 1e-18), color="#799d59", lw=0.8, label="centered FD")
    axes[1].set(xlabel="separation r (sigma)", ylabel="absolute error", title="Error against analytic derivative")
    axes[1].legend(frameon=False, fontsize=7.5)
    axes[1].grid(alpha=0.2, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "modes.png", dpi=200, facecolor="white")
    plt.close(fig)

    graph_value = jax.make_jaxpr(pair_energy)(r0)
    graph_grad = jax.make_jaxpr(jax.grad(pair_energy))(r0)
    draw_jaxpr_graph(graph_value, OUT / "graph.png", "JAX jaxpr: Lennard-Jones energy")
    draw_jaxpr_graph(graph_grad, OUT / "grad-graph.png", "JAX jaxpr: reverse gradient (look for add_any)")

    maxima = {
        "forward": float(error_f.max()),
        "reverse": float(error_r.max()),
        "finite_difference": float(error_fd.max()),
    }
    print(json.dumps({"derivatives": record, "max_errors": maxima}, indent=2))


if __name__ == "__main__":
    main()
