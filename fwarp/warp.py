import matplotlib.pyplot as plt
import numpy as np
from math import isclose

def warp(func, knots, weights, period):
    """
    params
    ------
    func: The function to be warped.
    knots: A list of numbers between 0 and 1
    weights: A list of length 1 greater than knots, representing how much of the
        function should fall between each knot. Weights must sum to 1.
    period: The absolute length at which the warps repeat

    returns:
    ------
    A new function where the weights determine how much of the old function is
    gotten by each of the knots.
    """
    if not isclose(sum(weights), 1):
        raise ValueError("Weights must sum to 1")
    if max(knots) >= 1 or min(knots) <= 0:
        raise ValueError("Knots must fall between 0 and 1")

    knots = [0] + knots + [1]
    def distort(x):
        num_periods = x // period
        remainder_fraction = (x % period) / period

        for i, knot in enumerate(knots):
            if remainder_fraction <= knot:
                break

        i -= 1
        start = sum(weights[:i])
        proportion = (remainder_fraction - knots[i]) / (knots[i + 1] - knots[i])
        new_remainder = start + (weights[i] * proportion)

        return (num_periods * period) + (new_remainder * period)

    return build_warped(func, distort)

def compress(func, factor=2):
    def distort(x):
        return factor * x
    return build_warped(func, distort)

def elongate(func, factor=2):
    return compress(func, factor=1/factor)

def add_noise(func, sd=1):
    def noisy(x):
        result = np.array(func(x))
        return result + np.random.normal(0, sd, len(result))
    return noisy

def scale(func, knots, scales, period):
    """
    params:
    ------
    scales: scaling factors
    knots: the breakpoints between which different scalinf factors are applied
    period: same as above
    """

    def scaled(x):
        result = func(x)

        for i, x in enumerate(x):
            remainder_fraction = (x % period) / period
            for j, knot in enumerate(knots + [1]):
                if knot >= remainder_fraction:
                    break
            result[i] *=  scales[j]
        return result

    return scaled

def build_warped(func, distortion_func):
    def warped(x):
        distorted_x = np.vectorize(distortion_func)(x)
        return func(distorted_x)
    return warped
