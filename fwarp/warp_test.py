from .warp import *
import cProfile
import numpy as np

def test_best_spline():
    # given
    knots = [1/3, 2/3]
    warp_yvalues = [.2, .8]
    scale_yvalues = [.6, 1.2, 1.7, .5]
    warped = spline_warp(np.sin, knots, warp_yvalues, np.pi * 2)
    distorted = spline_scale(warped, knots, scale_yvalues, np.pi * 2)

    # when
    best = find_best_spline(
        np.sin, distorted, np.linspace(0, 2*np.pi), 2, to_return='weights',
    )

    # then
    print(best)
    assert np.allclose(best['warp'], warp_yvalues, atol=.1)
    assert np.allclose(best['scale'], scale_yvalues, atol=.1)

if __name__ == '__main__':
    #cProfile.run('test_best_spline()', sort='cumulative')
    pass

