from scipy.interpolate import CubicSpline
from sklearn.mixture import GaussianMixture
import numpy as np
import scipy.stats as st


def canonical_mean(signals, allocation, num_motifs, num_knots=16):
    period = get_signal_length(signals)
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


def gmm(signals, num_motifs, num_knots=16):
    period = get_signal_length(signals)
    knot_points = get_knot_points(num_knots, period)

    funcs = np.array([CubicSpline(range(period), s) for s in signals])
    occurrences = np.array([c(knot_points) for c in funcs])

    model = GaussianMixture(num_motifs)
    model.fit(occurrences)
    return model


def simple_canonical(signals, num_motifs, period, num_knots):

    # number of places at which a motif can begin
    # TODO: Allow motifs to overrun the start and end of the signal
    num_signals = len(signals)
    signal_length = get_signal_length(signals)
    cols = signal_length - period + 1

    # sparse allocation matrix
    allocation = np.array([[-1] * cols] * num_signals)

    # motifs can overlap, so there are slightly more than we strictly need
    num_non_empty = int(num_signals * (signal_length / period) * 1.2)
    high = [num_signals, cols, num_motifs]
    low = [0, 0, 0]
    non_empty = st.randint.rvs(low, high, size=(num_non_empty, 3))
    for s, i, m in non_empty:
        allocation[s][i] = m

    initial = starting_canonical(signals, allocation, num_motifs, period,
                                 num_knots)
    return initial


def starting_canonical(signals, allocation, num_motifs, period, num_knots):
    occurrences = {i: [] for i in range(num_motifs)}
    for s in range(allocation.shape[0]):
        for i in range(allocation.shape[1]):
            motif_id = allocation[s][i]
            if motif_id > -1:
                motif_occurrence = signals[s][i:i + period]
                occurrences[motif_id].append(motif_occurrence)

    averages = np.zeros((num_motifs, period))
    for i, o in occurrences.items():
        averages[i] = np.mean(o, axis=0)

    average_splines = [CubicSpline(np.arange(period), a) for a in averages]
    knot_points = get_knot_points(num_knots, period)

    y_values = [a(knot_points) for a in average_splines]
    return y_values


def build_splines(knot_points, y_values):
    return [CubicSpline(knot_points, y) for y in y_values]


def get_signal_length(signals):
    lengths = set(len(s) for s in signals)

    if len(lengths) > 1:
        raise ValueError('All signals must be of the same length')

    period = list(lengths)[0]
    return period


def get_knot_points(num_knots, period):
    return ((np.array(range(num_knots + 1)) / num_knots)) * period
