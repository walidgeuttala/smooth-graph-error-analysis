"""Exact symmetry reductions and sufficient certificates for sigmoid reachability.

These are mathematical diagnostics, not trained models. Reduced simulations
represent large graph families without constructing their full adjacency.
"""
from numbers import Integral

import numpy as np


def sigmoid(value):
    return np.exp(-np.logaddexp(0., -np.asarray(value, dtype=float)))


def _arguments(size, steps, beta, minimum):
    if isinstance(size, bool) or not isinstance(size, Integral) or size < minimum:
        raise ValueError('invalid graph family size')
    if isinstance(steps, bool) or not isinstance(steps, Integral) or steps < 1:
        raise ValueError('steps must be a positive integer')
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError('beta must be finite and positive')


def _readout(row_sums):
    sums = np.asarray(row_sums)
    steps = len(sums)-1
    denominator = sums[-1]-1
    if np.any(denominator <= 0) or not np.all(np.isfinite(sums)):
        raise ValueError('invalid nonisolated denominator')
    return (steps*sums[-1]-sums[:-1].sum(axis=0))/denominator


def barbell(clique_size, steps, beta):
    """Two size-s cliques joined by one edge; four exact vertex orbits.

    Orbit order: left peripheral, left bridge, right bridge, right peripheral.
    At H1 every orbit-pair block is constant, including the diagonal blocks.
    Thereafter the closed-neighborhood count matrix gives exact updates.
    """
    _arguments(clique_size, steps, beta, 2)
    s = float(clique_size)
    sizes = np.array([s-1, 1, 1, s-1])
    count = np.array([[s-1, 1, 0, 0], [s-1, 1, 1, 0],
                      [0, 1, 1, s-1], [0, 0, 1, s-1]], dtype=float)
    hidden = sigmoid(beta*((count > 0).astype(float)-.5))
    row_sums = [np.ones(4), hidden@sizes]
    max_state_errors = []
    exact = (count > 0).astype(float)
    max_state_errors.append(float(np.abs(hidden-exact).max()))
    for _ in range(1, steps):
        hidden = sigmoid(beta*(count@hidden-.5))
        exact = (count@exact > 0).astype(float)
        row_sums.append(hidden@sizes)
        max_state_errors.append(float(np.abs(hidden-exact).max()))
    prediction = _readout(row_sums)
    target = np.array([2., (3*s-2)/(2*s-1), (3*s-2)/(2*s-1), 2.])
    return {'prediction': prediction, 'target': target, 'orbit_sizes': sizes,
            'row_sums': np.asarray(row_sums), 'max_state_errors': max_state_errors,
            'mae': float(np.dot(sizes, np.abs(prediction-target))/sizes.sum())}


def path_copies(copies, steps, beta):
    """Disjoint union of copies of P3, using own/foreign 3x3 blocks.

    The output is one representative component, ordered endpoint/center/endpoint.
    For copies=1 the foreign block is computed but contributes no mass or error.
    """
    _arguments(copies, steps, beta, 1)
    closed = np.array([[1., 1, 0], [1., 1, 1], [0., 1, 1]])
    own, foreign, exact = np.eye(3), np.zeros((3, 3)), np.eye(3)
    row_sums, max_state_errors = [np.ones(3)], []
    for _ in range(steps):
        own = sigmoid(beta*(closed@own-.5))
        foreign = sigmoid(beta*(closed@foreign-.5))
        exact = (closed@exact > 0).astype(float)
        row_sums.append(own.sum(1)+float(copies-1)*foreign.sum(1))
        max_state_errors.append(float(max(np.abs(own-exact).max(),
                                         foreign.max() if copies > 1 else 0.)))
    prediction = _readout(row_sums)
    target = np.array([1.5, 1., 1.5])
    return {'prediction': prediction, 'target': target, 'row_sums': np.asarray(row_sums),
            'max_state_errors': max_state_errors,
            'mae': float(np.abs(prediction-target).mean())}


def certificate_beta(max_degree, state_error, perturbation=0.):
    """Sufficient beta for ||H_k-R_k||max <= state_error at every depth.

    Update: sigmoid(beta*((A+I)H + E_k - 1/2)); ||E_k||max<=perturbation.
    H0=I, simple undirected graph, max degree <= max_degree. E_k may depend
    on the current state; its value bound suffices for this forward statement.
    """
    if isinstance(max_degree, bool) or not isinstance(max_degree, Integral) or max_degree < 0:
        raise ValueError('max_degree must be a nonnegative integer')
    if not np.isfinite(state_error) or not 0 < state_error < .5:
        raise ValueError('state_error must lie in (0,1/2)')
    if not np.isfinite(perturbation) or perturbation < 0:
        raise ValueError('perturbation must be finite and nonnegative')
    margin = .5-perturbation-(max_degree+1)*state_error
    if margin <= 0:
        raise ValueError('certificate has no positive threshold margin')
    return float((np.log1p(-state_error)-np.log(state_error))/margin)


def centrality_error_bound(n, reachable_other, steps, state_error):
    """Absolute bound after component eccentricity: 3*K*n*delta/(m-n*delta).

    For nonisolates only, epsilon=0. Requires exact H0 and the uniform state
    error hypothesis; returns no certificate for an unprotected denominator.
    """
    if (not 0 < reachable_other < n or steps < 1 or not 0 <= state_error < .5):
        raise ValueError('invalid centrality-bound arguments')
    denominator = reachable_other-n*state_error
    if denominator <= 0:
        raise ValueError('state error does not protect denominator')
    return float(3*steps*n*state_error/denominator)


def state_jacobian_bound(max_degree, beta, state_error, residual_lipschitz=0.):
    """Conditional local state-Jacobian infinity-norm bound in certified tube.

    Assumes E is differentiable with ||DE||infinity<=residual_lipschitz.
    This is not a bound on every parameter gradient or on optimization success.
    """
    if max_degree < 0 or beta <= 0 or not 0 <= state_error < .5 or residual_lipschitz < 0:
        raise ValueError('invalid Jacobian-bound arguments')
    return float(beta*state_error*(1-state_error)*(max_degree+1+residual_lipschitz))
