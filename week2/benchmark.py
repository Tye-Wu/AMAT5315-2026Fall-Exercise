#!/usr/bin/env python3
"""Reproduce the Week 2 timing and force-scaling tables.

The script performs three wall-clock runs per row and writes raw timings plus
median/min/max summaries to benchmark-results.json. Run it after building both
debug and release binaries.
"""

import json
import pathlib
import platform
import statistics
import subprocess
import sys
import tempfile
import time


HERE = pathlib.Path(__file__).resolve().parent
DEBUG = HERE / "md" / "target" / "debug" / "md"
RELEASE = HERE / "md" / "target" / "release" / "md"
BASELINE = HERE / "week2-sim.py"
OUTPUT = HERE / "benchmark-results.json"


def timed(command, cwd=None):
    started = time.perf_counter()
    subprocess.run(
        [str(part) for part in command],
        cwd=cwd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return time.perf_counter() - started


def three_runs(builder):
    values = [builder() for _ in range(3)]
    return {
        "runs_s": values,
        "median_s": statistics.median(values),
        "min_s": min(values),
        "max_s": max(values),
    }


def contract_timing():
    def run_python():
        with tempfile.TemporaryDirectory(prefix="week2_numpy_") as directory:
            return timed([sys.executable, BASELINE], cwd=directory)

    def run_rust(binary):
        def execute():
            with tempfile.TemporaryDirectory(prefix="week2_rust_") as directory:
                return timed([binary, "run", "--out", pathlib.Path(directory) / "artifacts"])

        return execute

    return {
        "numpy": three_runs(run_python),
        "rust_debug": three_runs(run_rust(DEBUG)),
        "rust_release": three_runs(run_rust(RELEASE)),
    }


def scaling():
    rows = []
    for n in (100, 400, 1600):
        row = {"n": n}
        for force in ("naive", "cells"):
            def execute(force=force, n=n):
                with tempfile.TemporaryDirectory(prefix=f"week2_{force}_{n}_") as directory:
                    return timed(
                        [
                            RELEASE,
                            "run",
                            "--force",
                            force,
                            "--n",
                            n,
                            "--eq-steps",
                            100,
                            "--steps",
                            500,
                            "--out",
                            pathlib.Path(directory) / "run",
                        ]
                    )

            row[force] = three_runs(execute)
        row["speedup"] = row["naive"]["median_s"] / row["cells"]["median_s"]
        rows.append(row)
    return rows


def main():
    for binary in (DEBUG, RELEASE):
        if not binary.exists():
            raise SystemExit(f"missing {binary}; build debug and release first")
    result = {
        "platform": platform.platform(),
        "contract": "N=100, eq_steps=2000, steps=10000, sample_every=50",
        "timing": contract_timing(),
        "scaling_contract": "eq_steps=100, steps=500, default parameters otherwise",
        "scaling": scaling(),
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

