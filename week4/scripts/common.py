"""Shared helpers for the reproducible Week 4 run and plot scripts."""

import json
import os
import shutil
import subprocess
from pathlib import Path


WEEK4 = Path(__file__).resolve().parent.parent
ROOT = WEEK4.parent
ARTIFACTS = WEEK4 / "artifacts"
EVIDENCE = WEEK4 / "evidence"
BIN_DIR = Path.home() / ".cargo" / "bin"


def binary(name):
    found = shutil.which(name)
    if found:
        return found
    candidate = WEEK4 / "target" / "release" / name
    if candidate.exists():
        return str(candidate)
    raise RuntimeError("Build the Week 4 commands with: cargo install --path . --quiet")


def make_field(*arguments):
    result = subprocess.run(
        [binary("field"), *map(str, arguments)],
        cwd=WEEK4,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, separators=(",", ":")) + "\n")


def run_fluid(field, method, viscosity, dt, t_end, every, out, expect_failure=None):
    out_path = ARTIFACTS / out
    out_path.mkdir(parents=True, exist_ok=True)
    save_json(out_path / "initial-field.json", field)
    relative_out = os.path.relpath(out_path, WEEK4)
    result = subprocess.run(
        [
            binary("fluid"),
            "--method", method,
            "--nu", str(viscosity),
            "--dt", str(dt),
            "--t-end", str(t_end),
            "--every", str(every),
            "--out", relative_out,
        ],
        cwd=WEEK4,
        input=json.dumps(field),
        capture_output=True,
        text=True,
    )
    (out_path.parent / f"{out_path.name}.tsv").write_text(result.stdout)
    (out_path.parent / f"{out_path.name}.stderr.txt").write_text(result.stderr)
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr or result.stdout)
    if result.returncode == 1 and expect_failure is False:
        raise RuntimeError(f"unexpected non-finite stop in {out}:\n{result.stdout}")
    if result.returncode == 0 and expect_failure is True:
        print(f"Note: {out} stayed finite through t={t_end:g}.")
        return result
    return result


def read_tsv(path):
    rows = []
    for line in Path(path).read_text().splitlines()[1:]:
        fields = line.split("\t")
        if len(fields) >= 3:
            try:
                rows.append((float(fields[0]), float(fields[1]), float(fields[2])))
            except ValueError:
                rows.append((float(fields[0]), float("nan"), float("nan")))
    return rows


def latest_stop(path):
    lines = Path(path).read_text().splitlines()
    for line in reversed(lines):
        try:
            return float(line.split("\t", 1)[0])
        except (ValueError, IndexError):
            continue
    return float("nan")
