"""Tests for the Monte Carlo estimate of pi in week1/pi.py."""

import math

from pi import estimate_pi


def test_estimate_pi():
    assert abs(estimate_pi(1_000_000, seed=2026) - math.pi) < 1e-2
