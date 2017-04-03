import sys
from math import fabs
from operator import sub

import numpy as np
import scipy.optimize as sop
from scipy.interpolate import CubicSpline

from .warp import functional_scale, functional_warp
from .best_warp import best_warp


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


def find_best_spline(func1, func2, period, **kwargs):
    def warp_constraints(warp_weights, scale_weights, knot_points):
        warp_near_diag = np.max(np.abs(warp_weights - knot_points[1:-1])) < 0.1
        scale_near_one = np.max(np.abs(scale_weights - 1)) < 0.5
        scale_changes_slow = max(
            abs(sub(*pair))
            for pair in zip(scale_weights, scale_weights[1:])) < .5

        return warp_near_diag and scale_near_one and scale_changes_slow

    kwargs['warp_constraints'] = warp_constraints
    return best_warp(func1, func2, period, spline_warp, spline_scale,
                     generate_spline_distortion, generate_spline_scale,
                     **kwargs)
