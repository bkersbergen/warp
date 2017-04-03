import sys

import bezier
import numpy as np
import scipy.optimize as sop

from .warp import functional_scale, functional_warp, \
    invert_monotonic_over_unit_interval
from .best_warp import best_warp


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


def find_best_bezier(func1, func2, period, **kwargs):
    def warp_constraints(warp_weights, scale_weights, knot_points):
        return np.max(warp_weights) > 1 or np.min(warp_weights) < 0 \
           or np.max(scale_weights) > 2 or np.min(scale_weights) < .5

    kwargs['warp_constraints'] = warp_constraints
    return best_warp(func1, func2, period, bezier_warp, bezier_scale,
                     generate_bezier_distortion, generate_bezier_scale,
                     **kwargs)
