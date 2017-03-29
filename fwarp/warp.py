import sys
import bezier
import numpy as np
import scipy.optimize as sop
from operator import sub
from math import isclose, fabs
from scipy.interpolate import CubicSpline

def functional_warp(func, distortion_func, period):
   def distort(x):
       num_periods = x // period
       percent_through = (x % period) / period
       return (distortion_func(percent_through) + num_periods) * period
   return build_warped(func, distort)

def functional_scale(func, scaling_func, period):
    # scaling_func maps from the interval 0 -> 1 to the
    # scale at that percentage through the period
    def scaled(x):
        y = (x % period) / period
        result = func(x)
        return scaling_func(y) * result
    return np.vectorize(scaled)

def linear_warp(func, knots, weights, period):
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
                return start + weights[i - 1] * ((x - ks[i - 1]) / (ks[i] - ks[i - 1]))
    return np.vectorize(distortion_function)

def generate_linear_scale(knots, weights):
    # generates a scale function from the parameters to the 'scale' function
    # weights is two longer than weights
    def scale(x):
        ks = [0] + knots + [1]
        for i in range(len(ks)):
            if x < ks[i]:
                slope = (weights[i] - weights[i - 1]) / (ks[i] - ks [i - 1]) 
                return (slope * (x - ks[i - 1])) + weights[i - 1]
    return np.vectorize(scale)

def find_best_linear_warp(func1, func2, x, num_knots=3, cost=None,
                   optimization_method="SLSQP", log=False, to_return='functions'):
    return find_best(func1, func2, x, num_knots, cost, 'warp',
                     optimization_method, log, to_return)

def find_best_linear_scale(func1, func2, x, num_knots=3, cost=None,
                    optimization_method="SLSQP", log=False, to_return='functions'):
    return find_best(func1, func2, x, num_knots, cost, 'scale', 
                     optimization_method, log, to_return)

def find_best_linear(func1, func2, x, knot_points, cost=None, manipulation='both',
              optimization_method="Powell", log=False, to_return='functions'):
    if cost is None:
        cost = lambda x, y: np.abs(x.flatten() - y.flatten())**2

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
            dist_func1 = linear_scale(intermediary, knot_points, params_2, max(x))

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
            return {'warp': warp_weights,
                    'scale': weights[knots + 1:]}
    
    elif to_return == 'functions':
        if manipulation == 'warp':
            weights = weights / weights.sum()
            return linear_warp(func1, knot_points, weights, max(x))
        elif manipulation == 'scale':
            return linear_scale(func1, knot_points, weights, max(x))
        elif manipulation == 'both':
            warp_weights = weights[:knots + 1] / weights[:knots + 1].sum()
            intermediary = linear_warp(func1, knot_points, warp_weights, max(x))
            return linear_scale(intermediary, knot_points, weights[knots + 1:], max(x))

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
        corresponding_x = lambda i: curve.evaluate(i)[0][0]
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

def find_best_bezier(func1, func2, x, knot_points, cost=None , log=False,
                     optimization_method='Powell', to_return='function',
                     optimization_options=None):
    # assumes x starts as zero
    if cost is None:
        def cost(x, y):
            return np.abs(x.flatten() - y.flatten())**2

    def distance(params):
        #TODO: deal with log case
        warp_params, scale_params = np.split(params, [len(knot_points)])
        if np.max(warp_params) > 1 or np.min(warp_params) < 0 \
           or np.max(scale_params) > 2 or np.min(scale_params) < .5:
            return sys.maxsize
        intermediary = bezier_warp(func1, knot_points, warp_params, max(x))
        new_func1 = bezier_scale(intermediary, knot_points, scale_params, max(x))
        total_cost = np.sum(cost(new_func1(x), func2(x)))
        print(total_cost)
        return total_cost

    starting_warp = np.linspace(0, 1, len(knot_points) + 2)[1:-1]
    starting_scale = np.ones(len(knot_points) + 2)
    starting = np.concatenate((starting_warp, starting_scale))
    optimal = sop.minimize(distance, starting, method=optimization_method,
                           options=optimization_options)
    # assert optimal.success, 'Optimization failed'

    weights = optimal.x
    if log:
        weights = np.exp(weights)

    warp_weights, scale_weights = np.split(weights, [len(knot_points)])
    both_weights = {'warp': warp_weights, 'scale': scale_weights}
    if to_return == 'weights':
        return both_weights
    elif to_return == 'function':
        intermediary = bezier_warp(func1, knot_points, both_weights['warp'], max(x))
        return bezier_scale(intermediary, knot_points, both_weights['scale'], max(x))


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

def find_best_spline(func1, func2, x, knot_points, cost=None , log=False,
                     optimization_method='Powell', to_return='function',
                     optimization_options=None):
    # assumes x starts as zero
    if cost is None:
        def cost(x, y):
            return np.abs(x.flatten() - y.flatten())**2

    num_knots = len(knot_points)
    diagonal = np.linspace(0, 1, num_knots + 2)
    def distance(params):
        #TODO: deal with log case
        warp_params, scale_params = np.split(params, [len(knot_points)])

        all_warp = np.zeros(num_knots + 2)
        all_warp[1:] = np.concatenate((warp_params, np.ones(1)))
        if (np.abs(all_warp - diagonal) > (1 / (2*(num_knots + 2)))).any() \
            or any(fabs(sub(*a)) > .75 for a in zip(scale_params, scale_params[1:])) \
            or (scale_params > 2).any() or (scale_params < 0.5).any():  
            return sys.maxsize

        intermediary = spline_warp(func1, knot_points, warp_params, max(x))
        new_func1 = spline_scale(intermediary, knot_points, scale_params, max(x))
        total_cost = np.sum(cost(new_func1(x), func2(x)))
        # print(total_cost)
        return total_cost

    starting_warp = np.linspace(0, 1, len(knot_points) + 2)[1:-1]
    starting_scale = np.ones(len(knot_points) + 2)
    starting = np.concatenate((starting_warp, starting_scale))
    optimal = sop.minimize(distance, starting, method=optimization_method,
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
        intermediary = spline_warp(func1, knot_points, both_weights['warp'], max(x))
        return spline_scale(intermediary, knot_points, both_weights['scale'], max(x))





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

def build_warped(func, distortion_func):
    def warped(x):
        distorted_x = np.vectorize(distortion_func)(x)
        return func(distorted_x)
    return warped

def invert_scale_function(scale):
    def inverted(x):
        return 1 / scale(x)
    return inverted

def invert_distortion_function(distort):
    return np.vectorize(invert_monotonic_over_unit_interval(distort))

def invert_monotonic_over_unit_interval(func):
    def inverted(x):
        candidate = .5
        max = 1
        min = 0
        
        while np.abs(func(candidate) - x) >= 1e-5:
            if func(candidate) < x:
                min = candidate
                candidate = (candidate + max) / 2

            elif func(candidate) > x:
                max = candidate
                candidate = (candidate - min) / 2
        return candidate

    return inverted
