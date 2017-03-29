from .warp import *
import numpy as np

def test_best_bezier():
    # given
    knots = [1/3, 2/3]
    warp_yvalues = [.1, .9]
    scale_yvalues = [.6, 1.2, 1.7, .5]
    warped = bezier_warp(np.sin, knots, warp_yvalues, np.pi * 2)
    distorted = bezier_scale(warped, knots, scale_yvalues, np.pi * 2)

    # when
    print("about to optimize")
    best = find_best_bezier(
        np.sin, distorted, np.linspace(0, 2*np.pi), knots, to_return='weights',
        optimization_options={'maxfev': 10}
    )

    # then
    assert np.allclose(best['warp'], warp_yvalues)
    assert np.allclose(best['scale'], scale_yvalues)

