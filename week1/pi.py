"""Estimate pi with a Monte Carlo dartboard.

Darts are thrown uniformly at the unit square [0, 1] x [0, 1]. Those
landing within distance 1 of the origin fall inside a quarter circle of
radius 1, whose area is pi / 4. So pi is roughly four times the
fraction of darts that land inside the quarter circle.
"""

import random


def estimate_pi(n, seed):
    """Estimate pi by throwing n random darts at the unit square.

    Parameters
    ----------
    n : int
        Number of darts to throw.
    seed : int
        Seed for the random number generator, so the same call always
        returns the same value.

    Returns
    -------
    float
        Four times the fraction of darts within distance 1 of the
        origin, i.e. an estimate of pi.
    """
    rng = random.Random(seed)
    inside = 0
    for _ in range(n):
        x = rng.random()  # uniform in [0, 1)
        y = rng.random()
        if x * x + y * y <= 1.0:
            inside += 1
    return 4.0 * inside / n
