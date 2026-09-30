from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "inputs" / "marmousi.json"
BORN = ROOT / "artifacts" / "marmousi-born"
ADJOINT = ROOT / "artifacts" / "marmousi-image"


def main() -> None:
    exp = json.loads(INPUT.read_text())
    result = json.loads((ADJOINT / "result.json").read_text())
    background = np.asarray(exp["background"])
    perturbation = np.asarray(exp["perturbation"])
    born = np.load(BORN / "born_data.npy")
    image = np.load(ADJOINT / "image.npy")
    dx_km = exp["dx"] * exp["length_unit_m"] / 1000
    dt_s = exp["dt"] * exp["time_unit_s"]
    extent = [0, exp["nx"] * dx_km, exp["nz"] * dx_km, 0]
    shot = min(range(len(exp["shots"])), key=lambda i: abs(exp["shots"][i][0] * dx_km - 10.0))
    shot_x_km = exp["shots"][shot][0] * dx_km

    fig, ax = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    vmin, vmax = np.percentile(background, [1, 99])
    p = ax[0, 0].imshow(background, extent=extent, aspect="auto", cmap="viridis",
                        vmin=vmin, vmax=vmax)
    ax[0, 0].set(title="Marmousi background velocity", xlabel="Distance (km)", ylabel="Depth (km)")
    fig.colorbar(p, ax=ax[0, 0], label="Velocity (km/s)")

    bound = float(np.max(np.abs(perturbation)))
    p = ax[0, 1].imshow(perturbation, extent=extent, aspect="auto", cmap="RdBu_r",
                        vmin=-bound, vmax=bound)
    ax[0, 1].set(title="Velocity perturbation", xlabel="Distance (km)", ylabel="Depth (km)")
    fig.colorbar(p, ax=ax[0, 1], label="Δvelocity (km/s)")

    gather = born[shot]
    t = np.arange(exp["steps"]) * dt_s
    rx = np.asarray([point[0] * dx_km for point in exp["receivers"]])
    clip = float(np.percentile(np.abs(gather), 99.5))
    p = ax[1, 0].imshow(gather, extent=[rx[0], rx[-1], t[-1], t[0]], aspect="auto",
                        cmap="RdBu_r", vmin=-clip, vmax=clip)
    ax[1, 0].set(title=f"Born data; shot x={shot_x_km:.2f} km (nearest to 10 km)",
                 xlabel="Receiver position (km)", ylabel="Time (s), increasing downward")
    fig.colorbar(p, ax=ax[1, 0], label="Scattered pressure")

    image_lim = float(np.percentile(np.abs(image), 99.5))
    p = ax[1, 1].imshow(image, extent=extent, aspect="auto", cmap="RdBu_r",
                        vmin=-image_lim, vmax=image_lim)
    ax[1, 1].set(title="Raw Treeverse adjoint image", xlabel="Distance (km)", ylabel="Depth (km)")
    fig.colorbar(p, ax=ax[1, 1], label="Image amplitude (a.u.)")
    fig.suptitle("Marmousi Born modeling and checkpointed adjoint imaging")
    output = ROOT / "artifacts" / "marmousi.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)

    summary = {
        "image_shape": list(image.shape),
        "image_l2_norm": float(np.linalg.norm(image)),
        "expected_image_l2_norm": 6.7037741e-4,
        "relative_norm_error": float(abs(np.linalg.norm(image) - 6.7037741e-4) / 6.7037741e-4),
        "born_shape": list(born.shape),
        "selected_shot_index": shot,
        "selected_shot_x_km": shot_x_km,
        "scheduler_forward_calls": result["scheduler_forward_calls"],
        "reverse_calls": result["reverse_calls"],
        "peak_saved_states": result["peak_saved_states"],
        "peak_saved_bytes": result["peak_saved_bytes"],
        "transpose_relative_error": result["transpose_relative_error"],
        "full_history_not_used": result["storage"] == "treeverse",
    }
    (ROOT / "artifacts" / "marmousi-validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
