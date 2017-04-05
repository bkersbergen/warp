from fwarp import simple_canonical, spline_warp
import numpy as np


def test_simple_canonical():
    knots = np.array([1 / 3, 2 / 3])
    num_signals = 100
    warps = [
        spline_warp(np.sin, knots, np.random.normal(knots, .2), 2 * np.pi)
        for _ in range(num_signals)
    ]

    x = np.linspace(0, 10)
    signals = [w(x) for w in warps]

    results = simple_canonical(signals, 10, 5, 5)
