import sys
from math import isclose
from .warp import functional_warp, functional_scale

import numpy as np
import scipy.optimize as sop


def linear_warp(func, knots, weights, period):
    """
    params
    ------
    func: The function to be warped.
    knots: A list of numbers between 0 and 1
    weights: A list of length 1 greater than knots, representing how much of
        the function should fall between each knot. Weights must sum to 1.
    period: The absolute length at which the warps repeat

    returns:
    ------
    A new function where the weights determine how much of the old function is
    gotten by each of the knots.
    """
    distort = generate_linear_distortion(knots, weights)
    return functional_warp(func, distort, period)


def linear_scale(func, knots, scales, period):
    """
    params:
    ------
    scales: scaling factors, assumed continuous
    knots: the breakpoints between which different scalinf factors are applied
    period: same as above
    """
    scale_func = generate_linear_scale(knots, scales)
    return functional_scale(func, scale_func, period)


def generate_linear_distortion(knots, weights):
    # generates a distortion function from the parameters to the `warp` function
    if not isclose(sum(weights), 1):
        raise ValueError("Weights must sum to 1")
    if max(knots) >= 1 or min(knots) <= 0:
        raise ValueError("Knots must fall between 0 and 1")

    def distortion_function(x):
        start = 0
        ks = [0] + knots + [1]
        for i in range(len(ks)):
            if x < ks[i]:
                start = 0 if i < 2 else sum(weights[:i - 1])
                return start + weights[i - 1] * (
                    (x - ks[i - 1]) / (ks[i] - ks[i - 1]))

    return np.vectorize(distortion_function)


def generate_linear_scale(knots, weights):
    # generates a scale function from the parameters to the 'scale' function
    # weights is two longer than weights
    def scale(x):
        ks = [0] + knots + [1]
        for i in range(len(ks)):
            if x < ks[i]:
                slope = (weights[i] - weights[i - 1]) / (ks[i] - ks[i - 1])
                return (slope * (x - ks[i - 1])) + weights[i - 1]

    return np.vectorize(scale)


def find_best_linear_warp(func1,
                          func2,
                          x,
                          num_knots=3,
                          cost=None,
                          optimization_method="SLSQP",
                          log=False,
                          to_return='functions'):
    return find_best_linear(func1, func2, x, num_knots, cost, 'warp',
                            optimization_method, log, to_return)


def find_best_linear_scale(func1,
                           func2,
                           x,
                           num_knots=3,
                           cost=None,
                           optimization_method="SLSQP",
                           log=False,
                           to_return='functions'):
    return find_best_linear(func1, func2, x, num_knots, cost, 'scale',
                            optimization_method, log, to_return)


def find_best_linear(func1,
                     func2,
                     x,
                     knot_points,
                     cost=None,
                     manipulation='both',
                     optimization_method="Powell",
                     log=False,
                     to_return='functions'):
    if cost is None:
        def cost(x, y): return np.abs(x.flatten() - y.flatten())**2

    knots = len(knot_points)

    def distance(params):
        if log:
            params = np.exp(params)
            # sometimes the params are too big
            if sum(np.isinf(params)) != 0 or sum(np.isnan(params) != 0):
                return sys.maxsize

        if manipulation == 'warp':
            params = params / params.sum()
            dist_func1 = linear_warp(func1, knot_points, params, max(x))

        elif manipulation == 'scale':
            dist_func1 = linear_scale(func1, knot_points, params, max(x))

        elif manipulation == 'both':
            params_1 = params[:knots + 1]
            params_1 = params_1 / params_1.sum()
            intermediary = linear_warp(func1, knot_points, params_1, max(x))

            params_2 = params[knots + 1:]
            dist_func1 = linear_scale(intermediary, knot_points, params_2,
                                      max(x))

        total_cost = np.sum(cost(dist_func1(x), func2(x)))
        return total_cost

    if manipulation == 'warp':
        starting = np.ones(knots + 1)
    elif manipulation == 'scale':
        starting = np.ones(knots + 2)
    elif manipulation == 'both':
        starting = np.ones((knots * 2) + 3)
    optimal = sop.minimize(distance, starting, method=optimization_method)
    assert optimal.success, 'Optimization failed'

    weights = optimal.x
    if log:
        weights = np.exp(weights)

    if to_return == 'weights':
        if manipulation == 'warp':
            return weights / weights.sum()
        if manipulation == 'scale':
            return weights
        if manipulation == 'both':
            warp_weights = weights[:knots + 1] / weights[:knots + 1].sum()
            return {'warp': warp_weights, 'scale': weights[knots + 1:]}

    elif to_return == 'functions':
        if manipulation == 'warp':
            weights = weights / weights.sum()
            return linear_warp(func1, knot_points, weights, max(x))
        elif manipulation == 'scale':
            return linear_scale(func1, knot_points, weights, max(x))
        elif manipulation == 'both':
            warp_weights = weights[:knots + 1] / weights[:knots + 1].sum()
            intermediary = linear_warp(func1, knot_points, warp_weights,
                                       max(x))
            return linear_scale(intermediary, knot_points, weights[knots + 1:],
                                max(x))
