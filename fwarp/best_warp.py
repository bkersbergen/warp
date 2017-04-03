import numpy as np
import sys
from .warp import functional_warp, functional_scale
from scipy.optimize import minimize


def best_warp(func1,
              func2,
              period,
              warp_func,
              scale_func,
              generate_warp_func,
              generate_scale_func,
              num_knots=4,
              optimization_kwargs={},
              starting_weights=None,
              evaluation_points=None,
              distance_cost=None,
              warp_cost=None,
              warp_constraints=None,
              return_type='joint_function'):
    """
    params
    ---
    func1: function to be warped
    func2: function that func1 will be warped to try to match
    period: the period over which the distortions will be applied
    warp_func: the function that warps the target function
    scale_func: the function that scales the target function
    num_knots: the number of breakpoints in the distortion functions
    optimization_kwargs: kwargs that get passed into scipy's minimize function
    starting_weights: a dictionary containing the weights at which to start the
        minimization function. should have keys 'warp' and 'scale'
    evaluation_points: the points at which functions are evaluated to compute
        their cost. defaults to np.linspace(0, period, 20)
    distance_cost: a function that gets passed two arrays of the two functions
        evaluated at the to_evaluate points, and computes some cost between
        them
    warp_cost: a function that gets passed the warp_weights and scale_weights,
        and computes their cost (e.g. warping further from diagonal is less
        likely). also passed knot points for context
    warp_constraitns: a function that gets passed the warp_weights and
        scale_weights, and returns True if they are 'allowed' and false
        otherwise. Also passed knot_points for context.
    return_type: either 'joint_function' or 'independent_functions'. If the
        former, this function returns an optimal distortion function. If the
        later, this function returns the warp and scale functions that must be
        applied in that order to achieve the optimal warp.
    """

    valid_returns = ('joint_function', 'independent_functions', 'weights')
    if return_type not in valid_returns:
        raise ValueError('Invalid return type provided - must be in {}'
                         .format(valid_returns))

    # define default cost functions
    if distance_cost is None:

        def distance_cost(x, y):
            return np.sum((x - y)**2)

    if warp_constraints is None:

        def warp_constraints(warp_weights, scale_weights, knot_points):
            return True

    if warp_cost is None:

        def warp_cost(warp_weights, scale_weights, knot_points):
            return 0

    if evaluation_points is None:
        evaluation_points = np.linspace(0, period, 20)

    knot_points = np.linspace(0, 1, num_knots)

    def distance(params):
        warp_weights, scale_weights = np.split(params, [num_knots - 2])
        if not warp_constraints(warp_weights, scale_weights, knot_points):
            return sys.maxsize

        warped = warp_func(func1, knot_points[1:-1], warp_weights, period)
        warped_scaled = scale_func(warped, knot_points[1:-1], scale_weights,
                                   period)

        warped_scaled_evaluated = warped_scaled(evaluation_points).flatten()
        func2_evaluated = func2(evaluation_points).flatten()
        cost = distance_cost(warped_scaled_evaluated, func2_evaluated)
        cost += warp_cost(warp_weights, scale_weights, knot_points)
        return cost

    if starting_weights is None:
        starting_weights = {
            'warp': knot_points[1:-1],
            'scale': np.ones(num_knots)
        }

    x0 = np.concatenate((starting_weights['warp'], starting_weights['scale']))

    optimal = minimize(distance, x0, **optimization_kwargs)

    assert optimal.success, 'Optimization failed'

    warp_weights, scale_weights = np.split(optimal.x, [num_knots - 2])

    if return_type == 'weights':
        return {'warp': warp_weights, 'scale': scale_weights}

    warp_func = generate_warp_func(knot_points[1:-1], warp_weights)
    scale_func = generate_scale_func(knot_points[1:-1], scale_weights)

    if return_type == 'joint_function':
        warped = functional_warp(func1, warp_func, period)
        return functional_scale(warped, scale_func, period)

    if return_type == 'independent_functions':
        return {'warp': warp_func, 'scale': scale_func}
