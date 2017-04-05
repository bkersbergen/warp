from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
import numpy as np


def canonical_mean(signals, allocation, num_knots=16):
    period = get_period(signals)
    num_motifs = len(np.unique(allocation))
    averages = np.zeros((num_motifs, period))

    for i in range(num_motifs):
        if not sum(allocation == i):
            averages[i] = np.zeros(period)
        else:
            averages[i] = np.mean(signals[allocation == i], axis=0)

    average_splines = [CubicSpline(np.arange(period), a) for a in averages]
    knot_points = get_knot_points(num_knots, period)

    y_values = [a(knot_points) for a in average_splines]
    return [CubicSpline(knot_points, y) for y in y_values]


def canonical_spline(signals,
                     num_motifs,
                     num_knots=16,
                     cost=np.linalg.norm,
                     optimizer_kwargs={}):

    # start with random allocation
    allocation = np.random.randint(0, num_motifs, len(signals))

    # seed with mean
    canonical = canonical_mean(signals, allocation, num_knots)

    period = get_period(signals)
    x = np.linspace(0, period, 20)

    funcs = np.array([CubicSpline(range(period), s) for s in signals])
    evaluated = np.array([f(x) for f in funcs])

    knot_points = get_knot_points(num_knots, period)

    # exit when there are minimal changes in canonical motifs
    changes = np.ones(num_motifs)
    while max(changes) > 1e-6:
        # step 1
        for i in range(num_motifs):

            def distance(yvalues):
                candidate = CubicSpline(knot_points, yvalues)
                return sum(
                    cost(candidate(x) - e) for e in evaluated[allocation == i])

            result = minimize(distance,
                              canonical[i](knot_points), **optimizer_kwargs)

            assert result.success, 'Optimization failed'
            new_spline = CubicSpline(knot_points, result.x)
            change = np.abs(canonical[i](knot_points) - new_spline(knot_points))
            print(change)
            changes[i] = change
            canonical[i] = new_spline

        c_evaluated = np.array([c(x) for c in canonical])

        # step 2
        for i, e in enumerate(evaluated):
            costs = [cost(ce - e) for ce in c_evaluated]
            allocation[i] = costs.index(min(costs))

    return canonical


def get_period(signals):
    lengths = set(len(s) for s in signals)

    if len(lengths) > 1:
        raise ValueError('All signals must be of the same length')

    period = list(lengths)[0]
    return period


def get_knot_points(num_knots, period):
    return ((np.array(range(num_knots + 1)) / num_knots)) * period
