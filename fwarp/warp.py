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
    return scaled

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
    
