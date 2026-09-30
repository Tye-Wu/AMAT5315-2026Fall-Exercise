#!/usr/bin/env python3
"""Summarize saved samply profiles and render auditable sample-breakdown PNGs.

On macOS, samply 0.13 records correct addresses but may leave Rust names
unsymbolicated in the JSON. This script resolves frames with `atos`, then
classifies inclusive samples. The raw `.json.gz` files remain the authoritative
profiles and can be opened with `samply load`.
"""

import gzip
import json
import pathlib
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = pathlib.Path(__file__).resolve().parent
BINARY = HERE / "md" / "target" / "release" / "md"


def resolve_md_frames(thread):
    frame_table = thread["frameTable"]
    function_table = thread["funcTable"]
    frame_ids = []
    addresses = []
    for frame_id, function_id in enumerate(frame_table["func"]):
        if function_table["resource"][function_id] == 1:  # the `md` resource
            frame_ids.append(frame_id)
            addresses.append(frame_table["address"][frame_id])
    command = [
        "atos",
        "-o",
        str(BINARY),
        "-arch",
        "arm64",
        "-l",
        "0x100000000",
        *[hex(0x100000000 + address) for address in addresses],
    ]
    names = subprocess.check_output(command, text=True).splitlines()
    return dict(zip(frame_ids, names))


def summarize(profile_path):
    with gzip.open(profile_path, "rt") as handle:
        profile = json.load(handle)
    thread = profile["threads"][0]
    resolved = resolve_md_frames(thread)
    stack_table = thread["stackTable"]
    samples = thread["samples"]
    count = {"forces": 0, "write trajectory": 0, "everything else": 0}
    for stack_id in samples["stack"]:
        labels = []
        while stack_id is not None:
            labels.append(resolved.get(stack_table["frame"][stack_id], ""))
            stack_id = stack_table["prefix"][stack_id]
        joined = " ".join(labels)
        if "interactions_with_method" in joined:
            count["forces"] += 1
        elif "write_output" in joined or "serde_json" in joined:
            count["write trajectory"] += 1
        else:
            count["everything else"] += 1
    total = samples["length"]
    elapsed = (max(samples["time"]) - min(samples["time"])) / 1000.0
    return {
        "samples": total,
        "elapsed_s": elapsed,
        "share_percent": {key: 100.0 * value / total for key, value in count.items()},
    }


def draw(version, result):
    categories = ["forces", "write trajectory", "everything else"]
    values = [result["share_percent"][category] for category in categories]
    colors = ["#3978b8", "#cf8b48", "#a8a8a8"]
    figure, axis = plt.subplots(figsize=(7.2, 3.2))
    bars = axis.barh(categories[::-1], values[::-1], color=colors[::-1], height=0.62)
    for bar, value in zip(bars, values[::-1]):
        axis.text(value + 0.8, bar.get_y() + bar.get_height() / 2, f"{value:.1f}%", va="center")
    axis.set_xlim(0, 105)
    axis.set_xlabel("inclusive CPU samples (%)")
    axis.set_title(
        f"Samply profile: {version} force path, N=400\n"
        f"{result['samples']} samples, elapsed {result['elapsed_s']:.3f} s"
    )
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    axis.grid(axis="x", color="#e5e5e5", linewidth=0.7)
    figure.tight_layout()
    figure.savefig(HERE / f"profile-{version}.png", dpi=200, facecolor="white")


def main():
    results = {}
    for version in ("naive", "cells"):
        results[version] = summarize(HERE / f"profile-{version}.json.gz")
        draw(version, results[version])
    (HERE / "profile-results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
