import sys

import bezier
import numpy as np
import scipy.optimize as sop

from .warp import functional_scale, functional_warp, \
    invert_monotonic_over_unit_interval


def generate_bezier_distortion(knots, yvalues):
    if max(yvalues) > 1 or min(yvalues) < 0:
        raise ValueError('yvalues must be between 0 and 1')

    yvalues = [0] + list(yvalues) + [1]
    return generate_bezier(knots, yvalues)


def generate_bezier_scale(knots, yvalues):
    if max(yvalues) > 2 or min(yvalues) < 0.5:
        raise ValueError('yvalues must be between 0.5 and 2')

    return generate_bezier(knots, yvalues)


def generate_bezier(knots, yvalues):
    knots = [0] + list(knots) + [1]
    nodes = np.array(list(zip(knots, yvalues)))
    curve = bezier.Curve.from_nodes(nodes)

    def distort(x):
        def corresponding_x(i):
            return curve.evaluate(i)[0][0]

        find_x = invert_monotonic_over_unit_interval(corresponding_x)
        new_x = find_x(x)
        points = curve.evaluate(new_x)
        return points[0][1]

    return np.vectorize(distort)


def bezier_warp(func, knots, yvalues, period):
    distort = generate_bezier_distortion(knots, yvalues)
    return functional_warp(func, distort, period)


def bezier_scale(func, knots, yvalues, period):
    scale = generate_bezier_scale(knots, yvalues)
    return functional_scale(func, scale, period)


def find_best_bezier(func1,
                     func2,
                     x,
                     knot_points,
                     cost=None,
                     log=False,
                     optimization_method='Powell',
                     to_return='function',
                     optimization_options=None):
    # assumes x starts as zero
    if cost is None:

        def cost(x, y):
            return np.abs(x.flatten() - y.flatten())**2

    def distance(params):
        # TODO: deal with log case
        warp_params, scale_params = np.split(params, [len(knot_points)])
        if np.max(warp_params) > 1 or np.min(warp_params) < 0 \
           or np.max(scale_params) > 2 or np.min(scale_params) < .5:
            return sys.maxsize
        intermediary = bezier_warp(func1, knot_points, warp_params, max(x))
        new_func1 = bezier_scale(intermediary, knot_points, scale_params,
                                 max(x))
        total_cost = np.sum(cost(new_func1(x), func2(x)))
        print(total_cost)
        return total_cost

    starting_warp = np.linspace(0, 1, len(knot_points) + 2)[1:-1]
    starting_scale = np.ones(len(knot_points) + 2)
    starting = np.concatenate((starting_warp, starting_scale))
    optimal = sop.minimize(
        distance,
        starting,
        method=optimization_method,
        options=optimization_options)
    assert optimal.success, 'Optimization failed'

    # print(weights)
    weights = optimal.x
    if log:
        weights = np.exp(weights)

    warp_weights, scale_weights = np.split(weights, [len(knot_points)])
    both_weights = {'warp': warp_weights, 'scale': scale_weights}
    if to_return == 'weights':
        return both_weights
    elif to_return == 'function':
        intermediary = bezier_warp(func1, knot_points, both_weights['warp'],
                                   max(x))
        return bezier_scale(intermediary, knot_points, both_weights['scale'],
                            max(x))
