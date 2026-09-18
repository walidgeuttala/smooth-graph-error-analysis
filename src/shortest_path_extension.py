"""Finite-horizon shortest-path diagnostics, not a trained APSP solver.

Soft Bellman--Ford propagates exact-length walks, then combines lengths once.
There are no artificial zero-cost self loops or duplicated 'stay' histories.
Absent edges remain structurally absent. Positive edge weights are required.
"""
import numpy as np
from numbers import Integral


def trajectory_distances(states, reachable):
    """K H_K - sum_{k<K} H_k, with exact support and zero diagonal.

Unreachable entries are infinity, never a finite distance zero. The caller
must supply true support and sufficient depth for a full distance estimate.
"""
    h = np.asarray(states, dtype=float)
    support = np.asarray(reachable, dtype=bool)
    if (h.ndim != 3 or h.shape[0] < 2 or h.shape[1] != h.shape[2]
            or support.shape != h.shape[1:] or not np.isfinite(h).all()):
        raise ValueError('invalid trajectory or support')
    d = (len(h)-1)*h[-1]-h[:-1].sum(axis=0)
    d = np.where(support, d, np.inf)
    np.fill_diagonal(d, 0.)
    return d


def distance_outputs(distances, reachable):
    """Mean finite distance and reciprocal mean, with explicit validity flags.

Support is fixed externally. Invalid nonpositive means are not clipped.
Closeness is NaN there, and zero for isolates by convention.
"""
    d = np.asarray(distances, dtype=float)
    mask = np.asarray(reachable, dtype=bool).copy()
    if d.ndim != 2 or d.shape[0] != d.shape[1] or mask.shape != d.shape:
        raise ValueError('invalid distance matrix or support')
    np.fill_diagonal(mask, False)
    count = mask.sum(axis=1)
    total = np.where(mask, d, 0.).sum(axis=1)
    mean = np.divide(total, count, out=np.zeros(len(d)), where=count > 0)
    valid = (count == 0) | (np.isfinite(mean) & (mean > 0))
    closeness = np.full(len(d), np.nan)
    np.divide(1., mean, out=closeness, where=(count > 0) & valid)
    closeness[count == 0] = 0.
    return mean, closeness, valid


def soft_bellman_ford(weights, steps, tau=0., normalize=False):
    """All-pairs finite-horizon softmin over walks of lengths 0,...,K.

weights[i,j] is a positive edge cost or +inf for absence, diagonal +inf.
tau=0 yields hard min-plus. normalize=True averages over ALL admissible
walks per pair, not just locally finite predecessor values. Returned
log_counts exposes this pair-dependent normalization. Intermediate states
are exact-length costs. Dense reference cost O(K n^3), storage O(K n^2).
"""
    w = np.asarray(weights, dtype=float)
    if (w.ndim != 2 or w.shape[0] != w.shape[1] or len(w) == 0
            or np.isnan(w).any() or np.isneginf(w).any()
            or np.any(w[np.isfinite(w)] <= 0)
            or np.isfinite(np.diag(w)).any()):
        raise ValueError('positive costs, absent diagonal, square matrix required')
    if isinstance(steps, bool) or not isinstance(steps, Integral) or steps < 0:
        raise ValueError('steps must be a nonnegative integer')
    if not np.isfinite(tau) or tau < 0:
        raise ValueError('tau must be finite and nonnegative')
    n = len(w)
    state = np.full((n, n), np.inf)
    np.fill_diagonal(state, 0.)
    count = np.full((n, n), -np.inf)
    np.fill_diagonal(count, 0.)
    total, total_count = state.copy(), count.copy()
    states = [state.copy()]
    edges = np.where(np.isfinite(w), 0., -np.inf)
    for _ in range(steps):
        costs = state[:, :, None] + w[None, :, :]
        state = costs.min(axis=1) if tau == 0 else -tau*np.logaddexp.reduce(-costs/tau, axis=1)
        count = np.logaddexp.reduce(count[:, :, None]+edges[None, :, :], axis=1)
        total = np.minimum(total, state) if tau == 0 else -tau*np.logaddexp(-total/tau, -state/tau)
        total_count = np.logaddexp(total_count, count)
        states.append(state.copy())
    if normalize and tau:
        finite = np.isfinite(total)
        total[finite] += tau*total_count[finite]
    np.fill_diagonal(total, 0.)
    return total, np.stack(states), total_count


def error_metrics(predicted, exact):
    support = np.isfinite(exact)
    mask = support & ~np.eye(len(exact), dtype=bool)
    errors = np.abs(predicted[mask]-exact[mask])
    mean, close, valid = distance_outputs(predicted, support)
    truth, true_close, _ = distance_outputs(exact, support)
    return {'distance_max': float(errors.max()) if errors.size else 0.,
            'distance_mae': float(errors.mean()) if errors.size else 0.,
            'mean_max': float(np.abs(mean-truth).max()),
            'mean_mae': float(np.abs(mean-truth).mean()),
            'closeness_mae': float(np.abs(close-true_close).mean()) if valid.all() else None,
            'invalid_closeness_nodes': int((~valid).sum())}


def differentiable_soft_bellman_ford(weights, steps, tau, normalize=False):
    """PyTorch autograd version on fixed support, for strictly positive tau.

All-infinite reductions are guarded so unreachable pairs do not create NaN
gradients. Edge insertion/deletion is discrete and is not differentiated.
"""
    import torch
    if not isinstance(weights, torch.Tensor) or not weights.is_floating_point():
        raise ValueError('floating torch weights required')
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError('positive tau required for differentiable execution')
    # Reuse validation and obtain topology-only counts for normalization.
    _, _, counts = soft_bellman_ford(weights.detach().cpu().numpy(), steps, 0.)

    def reduce(costs, dim):
        finite=torch.isfinite(costs).any(dim=dim,keepdim=True)
        safe=torch.where(finite,costs,torch.zeros_like(costs))
        result=-tau*torch.logsumexp(-safe/tau,dim=dim)
        return torch.where(finite.squeeze(dim),result,torch.full_like(result,float('inf')))

    n=len(weights)
    diagonal=torch.eye(n,dtype=torch.bool,device=weights.device)
    state=torch.where(diagonal,torch.zeros_like(weights),torch.full_like(weights,float('inf')))
    total=state
    for _ in range(steps):
        state=reduce(state[:,:,None]+weights[None,:,:],1)
        total=reduce(torch.stack((total,state)),0)
    if normalize:
        correction=torch.as_tensor(np.where(np.isfinite(counts),counts,0.),
                                   dtype=weights.dtype,device=weights.device)
        total=total+tau*correction
    return torch.where(diagonal,torch.zeros_like(total),total)
