from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
exp = json.loads((ROOT / "inputs" / "reflector.json").read_text())
out = ROOT / "artifacts" / "adjoint"
image = np.load(out / "image.npy")
perturbation = np.asarray(exp["perturbation"])
dx_km = exp["dx"] * exp["length_unit_m"] / 1000
z0, z1 = 10, 34
x0, x1 = 7, 34
z = np.arange(z0, z1) * dx_km
x = np.arange(x0, x1) * dx_km
im = image[z0:z1, x0:x1]
dv = perturbation[z0:z1, x0:x1]
profile = np.linalg.norm(im, axis=1)
peak_i = int(np.argmax(profile))
peak_z = z[peak_i]
true_i = 21
depth_err = abs((z0 + peak_i) - true_i) * dx_km

fig, ax = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)
extent = [x[0], x[-1] + dx_km, z[-1] + dx_km, z[0]]
for a, data, title, label, cmap in [
    (ax[0], dv, "Known velocity perturbation", "Velocity change (km/s)", "RdBu_r"),
    (ax[1], im, "Raw RTM image", "Image amplitude (arbitrary units)", "RdBu_r"),
]:
    bound = float(np.max(np.abs(data)))
    pic = a.imshow(data, extent=extent, aspect="auto", cmap=cmap, vmin=-bound, vmax=bound)
    a.axhline(true_i * dx_km, color="black", ls="--", lw=1, label="True reflector")
    a.set(xlabel="Horizontal position (km)", ylabel="Depth (km)", title=title)
    fig.colorbar(pic, ax=a, label=label)
ax[2].plot(profile, z, color="black")
ax[2].axhline(true_i * dx_km, color="tab:red", ls="--", label="True depth: 2.1 km")
ax[2].axhline(peak_z, color="tab:blue", ls=":", label=f"Peak: {peak_z:.1f} km")
ax[2].set(xlabel="Row L2 norm", ylabel="Depth (km)", title="Image depth profile")
ax[2].legend(frameon=False, fontsize=8)
fig.suptitle(f"Reflector depth error: {depth_err:.3f} km")
fig.savefig(out / "image.png", dpi=180)
plt.close(fig)

run = json.loads((out / "run.json").read_text())
recording = run["recording"]
step = 132
frame_i = recording["steps"].index(step)
wavefield = np.load(out / "wavefield.npy")[frame_i]
fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
lim = float(np.max(np.abs(wavefield)))
pic = ax.imshow(wavefield, extent=[0, exp["nx"]*dx_km, exp["nz"]*dx_km, 0],
                aspect="equal", cmap="RdBu_r", vmin=-lim, vmax=lim)
ax.axhline(2.1, color="yellow", ls="--", lw=1, label="Reflector")
ax.set(xlabel="Horizontal position (km)", ylabel="Depth (km)",
       title=f"Adjoint field at reverse step {step}; t={recording['times'][frame_i]:.2f} s")
ax.legend(frameon=False)
fig.colorbar(pic, ax=ax, label="Pressure adjoint")
fig.savefig(out / "wavefield.png", dpi=180)
plt.close(fig)

print(json.dumps({"row_peak_index": int(z0 + peak_i), "row_peak_depth_km": float(peak_z),
                  "true_depth_index": true_i, "depth_error_km": float(depth_err),
                  "transpose_relative_error": json.loads((out / "result.json").read_text())["transpose_relative_error"]}, indent=2))
