import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as sop
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

    distort = generate_distortion_function(knots, weights)
    return fwarp(func, distort, period)

def fwarp(func, distortion_func, period):
   def distort(x):
       num_periods = x // period
       percent_through = (x % period) / period
       return (distortion_func(percent_through) + num_periods) * period
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
    scales: scaling factors, assumed continuous
    knots: the breakpoints between which different scalinf factors are applied
    period: same as above
    """
    scale_func = generate_scale_function(knots, scales)
    return fscale(func, scale_func, period)

def fscale(func, scaling_func, period):
    # scaling_func maps from the interval 0 -> 1 to the
    # scale at that percentage through the period
    def scaled(x):
        y = (x % period) / period
        result = func(x)
        return scaling_func(y) * result
    return np.vectorize(scaled)

def build_warped(func, distortion_func):
    def warped(x):
        distorted_x = np.vectorize(distortion_func)(x)
        return func(distorted_x)
    return warped

def generate_distortion_function(knots, weights):
    # generates a distortion function from the parameters to the `warp` function
    def distortion_function(x):
        start = 0
        ks = [0] + knots + [1]
        for i in range(len(ks)):
            if x < ks[i]:
                start = 0 if i < 2 else sum(weights[:i - 1])
                return start + weights[i - 1] * ((x - ks[i - 1]) / (ks[i] - ks[i - 1]))
    return np.vectorize(distortion_function)

def generate_scale_function(knots, weights):
    # generates a scale function from the parameters to the 'scale' function
    # weights is two longer than weights
    def scale(x):
        ks = [0] + knots + [1]
        for i in range(len(ks)):
            if x < ks[i]:
                slope = (weights[i] - weights[i - 1]) / (ks[i] - ks [i - 1]) 
                return (slope * (x - ks[i - 1])) + weights[i - 1]
    return np.vectorize(scale)

def invert_scale_function(scale):
    def inverted(x):
        return 1 / scale(x)
    return inverted

def invert_distortion_function(distort):
    def inverse_distort(x):
        candidate = .5
        max = 1
        min = 0
        
        while np.abs(distort(candidate) - x) >= 1e-5:
            if distort(candidate) < x:
                min = candidate
                candidate = (candidate + max) / 2

            elif distort(candidate) > x:
                max = candidate
                candidate = (candidate - min) / 2
        return candidate

    return np.vectorize(inverse_distort)

def find_best_warp(func1, func2, x, num_knots=3, cost=None,
                   optimization_method="SLSQP", log=False, to_return='functions'):
    return find_best('warp', func1, func2, x, num_knots, cost,
                     optimization_method, log, to_return)

def find_best_scale(func1, func2, x, num_knots=3, cost=None,
                    optimization_method="SLSQP", log=False, to_return='functions'):
    return find_best('scale', func1, func2, x, num_knots, cost,
                     optimization_method, log, to_return)

def find_best_warp_scale(func1, func2, x, num_knots=3, cost=None,
                        optimization_method="SLSQP", log=False, to_return='functions'):
    return find_best('both', func1, func2, x, num_knots, cost,
                     optimization_method, log, to_return)


def find_best(manipulation, func1, func2, x, knots=3, cost=None,
              optimization_method="SLSQP", log=False, to_return='functions'):
    if cost is None:
        cost = lambda x, y: np.abs(x - y)**2

    knot_points = [(i + 1)/(knots + 1) for i in range(knots)]
    def distance(params):
        if log:
            params = np.exp(params)
            # sometimes the params are too big
            if sum(np.isinf(params)) != 0 or sum(np.isnan(params) != 0):
                return sys.maxsize

        if manipulation == 'warp':
            params = params / params.sum()
            dist_func1 = warp(func1, knot_points, params, max(x))

        elif manipulation == 'scale':
            dist_func1 = scale(func1, knot_points, params, max(x))

        elif manipulation == 'both':
            params_1 = params[:knots + 1]
            params_1 = params_1 / params_1.sum()
            intermediary = warp(func1, knot_points, params_1, max(x))

            params_2 = params[knots + 1:]
            dist_func1 = scale(intermediary, knot_points, params_2, max(x))

        total_cost = np.sum(cost(dist_func1(x).flatten(), func2(x).flatten()))
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
            return {'warp': warp_weights,
                    'scale': weights[knots + 1:]}
    
    elif to_return == 'functions':
        if manipulation == 'warp':
            weights = weights / weights.sum()
            return warp(func1, knot_points, weights, max(x))
        elif manipulation == 'scale':
            return scale(func1, knot_points, weights, max(x))
        elif manipulation == 'both':
            warp_weights = weights[:knots + 1] / weights[:knots + 1].sum()
            intermediary = warp(func1, knot_points, warp_weights, max(x))
            return scale(intermediary, knot_points, weights[knots + 1:], max(x))

