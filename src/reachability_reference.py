"""Reference all-pairs reachability, separate from the trained models.

The target is mean finite distance to other vertices in the same component,
with isolated vertices assigned zero. Each state has quadratic memory cost;
this transparent implementation stores all K+1 states (O(K n^2) memory) and
uses dense multiplication (O(K n^3) time). It is not a scalable BFS replacement.
Hard updates are exact; finite-temperature sigmoid updates are approximations.
"""

from numbers import Integral

import numpy as np


def _adjacency(adjacency):
    a = np.asarray(adjacency, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("adjacency must be square")
    if not np.all(np.isfinite(a)) or not np.all((a == 0) | (a == 1)):
        raise ValueError("adjacency must be finite and binary")
    if not np.array_equal(a, a.T) or np.any(np.diag(a)):
        raise ValueError("adjacency must be symmetric with zero diagonal")
    return a


def reachability_states(adjacency, steps, *, beta=None, threshold=0.5):
    """Return H_0,...,H_steps, initialized with the identity.

    beta=None uses Boolean positive-threshold expansion. Otherwise each update
    is sigmoid(beta * ((A + I) H - threshold)). threshold=1 reproduces the
    manuscript's legacy activation for diagnostics, not an exact algorithm.
    """
    a = _adjacency(adjacency)
    if isinstance(steps, bool) or not isinstance(steps, Integral) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    if beta is not None and (not np.isfinite(beta) or beta <= 0):
        raise ValueError("beta must be finite and positive")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    states = [np.eye(len(a))]
    for _ in range(steps):
        expanded = a @ states[-1] + states[-1]
        if beta is None:
            state = (expanded > 0).astype(float)
        else:
            z = beta * (expanded - threshold)
            # Stable sigmoid without overflow in either tail.
            state = np.exp(-np.logaddexp(0.0, -z))
        states.append(state)
    return np.stack(states)


def reachability_readout(states, *, isolated_mask=None, epsilon=0.0):
    """Compute (K b_K - sum_{k<K} b_k)/(b_K - 1 + epsilon).

    Isolates must be identified by the caller and are assigned zero explicitly.
    Invalid nonisolated denominators raise instead of silently returning a
    centrality. Positive epsilon is an optional, biased diagnostic convention;
    exact recovery requires epsilon=0. At K=0 nonisolated nodes are undefined.
    """
    h = np.asarray(states, dtype=float)
    if h.ndim != 3 or h.shape[0] < 1 or h.shape[1] != h.shape[2]:
        raise ValueError("states must have shape (K+1, n, n)")
    if not np.all(np.isfinite(h)):
        raise ValueError("states must be finite")
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and nonnegative")
    n = h.shape[1]
    isolates = np.zeros(n, dtype=bool) if isolated_mask is None else np.asarray(isolated_mask, dtype=bool)
    if isolates.shape != (n,):
        raise ValueError("isolated_mask must have shape (n,)")
    k = h.shape[0] - 1
    counts = h.sum(axis=2)
    numerator = k * counts[-1] - counts[:-1].sum(axis=0)
    denominator = counts[-1] - 1 + epsilon
    active = ~isolates
    if np.any(denominator[active] <= 0):
        raise ValueError("nonisolated readout denominator must be positive")
    result = np.zeros(n)
    result[active] = numerator[active] / denominator[active]
    return result


def inverse_closeness(adjacency, steps=None, *, beta=None, threshold=0.5, epsilon=0.0):
    """Mean finite distance, or its truncated/soft reachability approximation.

    With Boolean updates and the default n-1 steps this is exact on every
    undirected simple graph, including disconnected graphs. For shorter runs,
    Boolean output averages only vertices reached within the chosen horizon.
    """
    a = _adjacency(adjacency)
    if steps is None:
        steps = max(len(a) - 1, 0)
    states = reachability_states(a, steps, beta=beta, threshold=threshold)
    return reachability_readout(states, isolated_mask=a.sum(axis=1) == 0, epsilon=epsilon)
