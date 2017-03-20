import matplotlib.pyplot as plt
import numpy as np

def expand_extract(func, period=1, factor=.3):
    """ factor of .5 gives original function """
    if factor < 0 or factor > 1:
        raise ValueError("factor must be between 0 and 1")

    def distort(x):
        num_periods = x // period
        remainder = x % period

        if remainder <= factor * period:
            new_remainder = ((1 - ((factor - (remainder / period)) / factor))
                            * (period / 2))
        else:
            new_remainder = (period / 2) + (
                ((remainder / period) - factor) * (
                    (period / 2) / ((1 - factor) * period))
            )

        return (num_periods * period) + new_remainder

    print(distort(.3 * np.pi * 2))
    print(distort(.301 * np.pi * 2))

    x = np.linspace(0, 2*np.pi, 1000)
    plt.plot(x, np.vectorize(distort)(x))

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
