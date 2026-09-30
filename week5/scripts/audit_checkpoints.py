from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
FULL = ART / "adjoint" / "image.npy"
REFERENCE_CALLS = {1: 28680, 3: 1695, 5: 990, 10: 642}


def audit_schedule(path: Path, steps: int, budget: int) -> dict:
    actions = json.loads(path.read_text())
    saved = {0}
    current = None
    grads: list[int] = []
    invalid_restore = duplicate_store = bad_call = bad_fetch = budget_overrun = 0
    peak = len(saved)
    for action in actions:
        op, n = action["action"], action["step"]
        if op == "restore":
            if n not in saved:
                invalid_restore += 1
            current = n
        elif op == "call":
            if current != n or n >= steps:
                bad_call += 1
            current = n + 1
        elif op == "store":
            if current != n or n in saved:
                duplicate_store += 1
            saved.add(n)
        elif op == "grad":
            if n not in saved:
                invalid_restore += 1
            grads.append(n)
        elif op == "fetch":
            if n == 0 or n not in saved:
                bad_fetch += 1
            saved.discard(n)
        else:
            raise ValueError(f"unknown action {op!r} in {path}")
        peak = max(peak, len(saved))
        if len(saved) > budget + 1:
            budget_overrun += 1
        if len(saved) != action["saved_states"]:
            raise AssertionError(f"recorded saved-state count mismatch in {path}: {action}")
    expected = list(range(steps - 1, -1, -1))
    if grads != expected:
        raise AssertionError(f"reverse gradients are not exactly {expected[0]}..0 in {path}")
    if saved != {0}:
        raise AssertionError(f"scheduler left unexpected saved states in {path}: {saved}")
    calls = sum(a["action"] == "call" for a in actions)
    if (invalid_restore, duplicate_store, bad_call, bad_fetch, budget_overrun) != (0, 0, 0, 0, 0):
        raise AssertionError(f"invalid scheduler actions in {path}")
    if peak != budget + 1 or calls != REFERENCE_CALLS[budget]:
        raise AssertionError(f"unexpected peak/work in {path}: peak={peak}, calls={calls}")
    return {"actions_file": path.name, "calls": calls, "grad_steps": len(grads),
            "peak_saved_states": peak, "invalid_restore": invalid_restore,
            "duplicate_store": duplicate_store, "bad_call": bad_call,
            "bad_fetch": bad_fetch, "budget_overrun": budget_overrun}


def main() -> None:
    full_image = np.load(FULL)
    budgets = sorted(REFERENCE_CALLS)
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    colors = {"restore": "tab:purple", "call": "tab:orange", "store": "tab:blue",
              "grad": "tab:green", "fetch": "tab:red"}
    for ax, budget in zip(axes.flat, budgets):
        folder = ART / f"checkpoint-{budget}"
        result = json.loads((folder / "result.json").read_text())
        image = np.load(folder / "image.npy")
        relative_error = float(np.linalg.norm(image - full_image) / np.linalg.norm(full_image))
        if relative_error >= 1e-9:
            raise AssertionError(f"checkpoint-{budget} image differs from full history: {relative_error}")
        if result["peak_saved_states"] != budget + 1:
            raise AssertionError(f"wrong state peak in checkpoint-{budget}")
        if result["scheduler_forward_calls"] != REFERENCE_CALLS[budget] * 3:
            raise AssertionError(f"wrong aggregate replay work in checkpoint-{budget}")
        if result["reverse_calls"] != 240 * 3:
            raise AssertionError(f"wrong aggregate reverse work in checkpoint-{budget}")
        per_shot = []
        for shot in range(3):
            report = audit_schedule(folder / f"actions-{shot}.json", 240, budget + 0)
            if result["per_shot"][shot]["actions_file"] != report["actions_file"]:
                raise AssertionError("result.json does not identify its action log")
            if result["per_shot"][shot]["scheduler_forward_calls"] != report["calls"]:
                raise AssertionError("per-shot work metadata mismatch")
            per_shot.append(report)
        rows.append({"extra_slots": budget, "peak_saved_states": result["peak_saved_states"],
                     "peak_saved_bytes": result["peak_saved_bytes"],
                     "scheduler_forward_calls_per_shot": REFERENCE_CALLS[budget],
                     "reverse_calls_per_shot": 240,
                     "image_relative_l2_error_vs_full_history": relative_error,
                     "transpose_relative_error": result["transpose_relative_error"],
                     "per_shot_audits": per_shot})
        acts = json.loads((folder / "actions-0.json").read_text())
        for op, color in colors.items():
            selected = [(i, a["step"]) for i, a in enumerate(acts) if a["action"] == op]
            if selected:
                xx, yy = zip(*selected)
                ax.scatter(xx, yy, s=2 if op == "call" else 9, alpha=.55,
                           color=color, label=op, rasterized=True)
        ax.set(title=f"{budget} extra slots; {REFERENCE_CALLS[budget]:,} replay calls/shot",
               xlabel="Action sequence index", ylabel="Time-step index")
        ax.invert_yaxis()
        ax.legend(ncol=3, fontsize=7, frameon=False)
    fig.suptitle("Treeverse action schedules (shot 0)")
    fig.savefig(ART / "checkpoint-actions.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    x = np.array(budgets)
    calls = np.array([REFERENCE_CALLS[k] for k in budgets])
    states = x + 1
    ax[0].plot(x, calls, "o-", label="Treeverse replay calls/shot")
    ax[0].axhline(240, color="tab:red", ls="--", label="Full history: 240 forward steps")
    ax[0].set(xlabel="Extra checkpoint slots", ylabel="Forward replays per shot",
              title="Compute–memory tradeoff", yscale="log")
    ax[0].legend(frameon=False, fontsize=8)
    ax[1].plot(x, states, "o-", color="tab:purple")
    ax[1].set(xlabel="Extra checkpoint slots", ylabel="Peak saved states (including initial state)",
              title="Peak memory: small reflector model")
    for k, n in zip(x, states):
        ax[1].annotate(f"{n} states\n{n*2*41*41*8:,} B", (k, n), xytext=(5, 8),
                       textcoords="offset points", fontsize=7)
    fig.savefig(ART / "checkpoint-work.png", dpi=180)
    plt.close(fig)

    full = json.loads((ART / "adjoint" / "result.json").read_text())
    summary = {"full_history_states": 241,
               "full_history_saved_bytes": 241 * 2 * 41 * 41 * 8,
               "full_history_transpose_relative_error": full["transpose_relative_error"],
               "budgets": rows,
               "overall": "PASS"}
    (ART / "checkpoint-audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
