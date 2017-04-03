import sys
from math import isclose
from .warp import functional_warp, functional_scale
from .best_warp import best_warp

import numpy as np


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


def find_best_linear(func1, func2, period, **kwargs):
    return best_warp(func1, func2, period, linear_warp, linear_scale,
                     generate_linear_distortion, generate_linear_scale,
                     **kwargs)
