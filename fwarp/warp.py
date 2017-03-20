import matplotlib.pyplot as plt
import numpy as np

def expand_contract(func, period=1, factor=.3):
    """ factor of .5 gives original function """
    if factor < 0 or factor > 1:
        raise ValueError("factor must be between 0 and 1")

    def distort(x):
        num_periods = x // period
        remainder_fraction = (x % period) / period
        if remainder_fraction <= factor:
            new_remainder = .5 * remainder_fraction / factor
        else:
            new_remainder = .5 + .5 * (remainder_fraction - factor) / (1 - factor)
        return (num_periods * period) + (new_remainder * period)

    return warp(func, distort)

def compress(func, factor=2):
    def distort(x):
        return factor * x
    return warp(func, distort)

def elongate(func, factor=2):
    return compress(func, factor=1/factor)

def warp(func, distortion_func):
    def warped(x):
        distorted_x = np.vectorize(distortion_func)(x)
        return func(distorted_x)
    return warped
