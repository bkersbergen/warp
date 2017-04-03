import sys
from math import fabs
from operator import sub

import numpy as np
import scipy.optimize as sop
from scipy.interpolate import CubicSpline

from .warp import functional_scale, functional_warp


def generate_spline_distortion(knots, yvalues):
    yvalues = [0] + list(yvalues) + [1]
    return generate_spline(knots, yvalues)


def generate_spline_scale(knots, yvalues):
    return generate_spline(knots, yvalues)


def generate_spline(knots, yvalues):
    knots = [0] + list(knots) + [1]
    return CubicSpline(knots, yvalues)


def spline_warp(func, knots, yvalues, period):
    distort = generate_spline_distortion(knots, yvalues)
    return functional_warp(func, distort, period)


def spline_scale(func, knots, yvalues, period):
    scale = generate_spline_scale(knots, yvalues)
    return functional_scale(func, scale, period)


def find_best_spline(func1,
                     func2,
                     x,
                     interior_knots,
                     cost=None,
                     log=False,
                     optimization_method='Powell',
                     to_return='function',
                     optimization_options=None,
                     starting_weights=None):
    # interior knots does not include the two on the edges (0 and 1)
    # assumes x starts as zero
    if cost is None:

        def cost(x, y):
            return np.abs(x.flatten() - y.flatten())**2

    knot_points = [(i + 1) / (interior_knots + 1)
                   for i in range(interior_knots)]
    diagonal = np.linspace(0, 1, interior_knots + 2)

    def distance(params):
        # TODO: deal with log case
        warp_params, scale_params = np.split(params, [len(knot_points)])

        all_warp = np.zeros(interior_knots + 2)
        all_warp[1:] = np.concatenate((warp_params, np.ones(1)))
        # conditions: warp can only be 1 / interior_knots away from diagonal
        # (adding more knots then keeps close to diagonal - don't add too many)
        # scale control points can only jump .5 from one to one
        # scale must stay between .5 and 2
        # NOTE: At the extremes allowed here with 2 interior knots, the
        # distortion function can be induced to briefly have negative slope.
        # Fix this be evaluating the gradient everywhere and rejecting any negatve values?
        # (too computationally intensive?)
        if (np.abs(all_warp - diagonal) > (1 / (2*(interior_knots + 2)))).any() \
            or any(fabs(sub(*a)) > .5 for a in zip(scale_params, scale_params[1:])) \
            or (scale_params > 2).any() or (scale_params < 0.5).any():
            return sys.maxsize

        intermediary = spline_warp(func1, knot_points, warp_params, max(x))
        new_func1 = spline_scale(intermediary, knot_points, scale_params,
                                 max(x))
        total_cost = np.sum(cost(new_func1(x), func2(x)))
        # print(total_cost)
        return total_cost

    if starting_weights is None:
        starting_warp = np.linspace(0, 1, len(knot_points) + 2)[1:-1]
        starting_scale = np.ones(len(knot_points) + 2)
        starting = np.concatenate((starting_warp, starting_scale))
    else:
        starting = np.concatenate((starting_weights['warp'],
                                   starting_weights['scale']))

    optimal = sop.minimize(
        distance,
        starting,
        method=optimization_method,
        options=optimization_options)
    assert optimal.success, 'Optimization failed'

    weights = optimal.x
    if log:
        weights = np.exp(weights)

    warp_weights, scale_weights = np.split(weights, [len(knot_points)])
    both_weights = {'warp': warp_weights, 'scale': scale_weights}
    if to_return == 'weights':
        return both_weights
    elif to_return == 'function':
        intermediary = spline_warp(func1, knot_points, both_weights['warp'],
                                   max(x))
        return spline_scale(intermediary, knot_points, both_weights['scale'],
                            max(x))
