import numpy as np


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


def find_canonical(funcs, num_interior_knots):

    pass


def compress(func, factor=2):
    def distort(x):
        return factor * x

    return build_warped(func, distort)


def elongate(func, factor=2):
    return compress(func, factor=1 / factor)


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
