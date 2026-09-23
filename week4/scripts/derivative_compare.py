#!/usr/bin/env python3
"""Print the Part 2 derivative comparison from the solver's Rust routines."""

import subprocess
from pathlib import Path

WEEK4 = Path(__file__).resolve().parent.parent

result = subprocess.run(
    ["cargo", "run", "--offline", "--release", "--example", "derivative_compare"],
    cwd=WEEK4,
    text=True,
    check=True,
)
