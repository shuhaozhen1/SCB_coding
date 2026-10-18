"""Exact, faster evaluation of the existing local-linear multiplier band.

The estimator, common-bandwidth fitted residuals, uncentered subject scores,
and Gaussian multiplier quantile are the same as in ``scb_core``.  This module
changes only how their finite sums and linear algebra are evaluated.  In
particular it introduces no leverage correction, pilot residuals, or inflation.
"""
from __future__ import annotations

import numpy as np


def _observed_beta(model, h, chunk_size=2048):
    """Solve the local normal equations without constructing unused inverses."""
    out = np.empty((len(model.t), model.p))
    for start in range(0, len(model.t), chunk_size):
        stop = min(start + chunk_size, len(model.t))
        points = model.t[start:stop]
        s0, s1, s2 = [model.xx.moment(points, h, a) for a in range(3)]
        gram = np.concatenate(
            [np.concatenate([s0, s1], axis=2),
             np.concatenate([s1, s2], axis=2)], axis=1)
        rhs = np.concatenate(
            [model.xy.moment(points, h, 0),
             model.xy.moment(points, h, 1)], axis=1)
        eig = np.linalg.eigvalsh(gram)
        if np.any(eig[:, 0] <= 1e-10 * np.maximum(eig[:, -1], 1)):
            raise np.linalg.LinAlgError(
                'Singular or ill-conditioned local design; no silent regularization.')
        out[start:stop] = np.linalg.solve(gram, rhs[..., None])[..., 0][:, :model.p]
    return out


def _subject_scores(model, grid, h, residual):
    """Evaluate both kernel moments for every subject in one operation.

    Prefix sums restart within each subject, keeping the numerical cancellation
    at the same scale as the original per-subject implementation.  Unequal
    counts with very large padding retain that original implementation.
    """
    from scb_core import WindowSums

    grid = np.asarray(grid, dtype=float)
    counts = np.bincount(model.sid, minlength=model.n)
    largest = int(counts.max())
    if model.n * largest > 4 * len(model.t):
        out = np.empty((model.n, len(grid), 2 * model.p))
        for subject in range(model.n):
            ix = model.sid == subject
            marks = model.x[ix] * (model.w[ix] * residual[ix])[:, None]
            prefix = WindowSums(model.t[ix], marks, 3)
            out[subject] = np.concatenate(
                [prefix.moment(grid, h, 0), prefix.moment(grid, h, 1)], axis=1)
        return out

    order = np.lexsort((model.t, model.sid))
    subject = model.sid[order]
    starts = np.r_[0, np.cumsum(counts)]
    within = np.arange(len(order)) - starts[subject]
    times = model.t[order]
    marks = model.x[order] * (model.w[order] * residual[order])[:, None]
    # Search in disjoint subject intervals.  The spacing also handles arbitrary
    # finite observation-time ranges and bandwidths without cross-subject hits.
    span = float(max(times.max(), grid.max()) - min(times.min(), grid.min())
                 + 2 * h + 1)
    keys = subject * span + times
    targets = np.arange(model.n)[:, None] * span + grid[None, :]
    lo = np.searchsorted(keys, targets - h, side='left') - starts[:-1, None]
    hi = np.searchsorted(keys, targets + h, side='right') - starts[:-1, None]
    lo = np.clip(lo, 0, counts[:, None])
    hi = np.clip(hi, 0, counts[:, None])
    rows = np.arange(model.n)[:, None]
    sums = []
    z = times - 0.5
    for degree in range(4):
        padded = np.zeros((model.n, largest + 1, model.p))
        padded[subject, within + 1] = marks * z[:, None] ** degree
        np.cumsum(padded, axis=1, out=padded)
        sums.append(padded[rows, hi] - padded[rows, lo])
    c = (grid - 0.5)[None, :, None]
    a0 = sums[0]
    a1 = (sums[1] - c * sums[0]) / h
    a2 = (sums[2] - 2 * c * sums[1] + c * c * sums[0]) / h ** 2
    a3 = (sums[3] - 3 * c * sums[2] + 3 * c * c * sums[1]
          - c ** 3 * sums[0]) / h ** 3
    return np.concatenate([(.75 / h) * (a0 - a2),
                           (.75 / h) * (a1 - a3)], axis=2)


def multiplier_band(model, grid, h, rng=None, B=2000, alpha=.05,
                    components=None, multipliers=None, observed_chunk_size=2048):
    """Return the same fields and target-specific band quantile as scb_core.

    ``multipliers`` can be supplied for an exact paired numerical check.  The
    matrix product only includes targeted components; other returned component
    bands use that same critical value, matching the original API.
    """
    grid = np.asarray(grid, dtype=float)
    if h <= 0 or not 0 < alpha < 1:
        raise ValueError('Require h > 0 and 0 < alpha < 1.')
    if components is None:
        components = np.arange(model.p)
    components = np.asarray(components, dtype=int)
    if components.ndim != 1 or len(components) == 0:
        raise ValueError('components must be a nonempty one-dimensional index array.')
    if np.any((components < 0) | (components >= model.p)):
        raise ValueError('A component index lies outside the model dimension.')
    fit = model.fit(grid, h)
    observed = _observed_beta(model, h, observed_chunk_size)
    residual = model.y - np.sum(model.x * observed, axis=1)
    score = _subject_scores(model, grid, h, residual)
    psi = model.n * np.einsum('gkr,ngr->ngk', fit.inverse[:, :model.p], score,
                             optimize=True)
    sigma = np.sqrt(np.mean(psi ** 2, axis=0))
    if np.any(sigma <= 1e-12):
        raise FloatingPointError('Degenerate fitted variance')
    if multipliers is None:
        if rng is None:
            raise ValueError('Supply an RNG or an explicit multiplier matrix.')
        multipliers = rng.normal(size=(B, model.n))
    multipliers = np.asarray(multipliers, dtype=float)
    if multipliers.ndim != 2 or multipliers.shape[1] != model.n:
        raise ValueError('multipliers must have shape (B, number of subjects).')
    standardized = psi[:, :, components] / sigma[None, :, components]
    process = multipliers @ standardized.reshape(model.n, -1) / np.sqrt(model.n)
    maxima = np.max(np.abs(process), axis=1)
    q = float(np.quantile(maxima, 1 - alpha, method='inverted_cdf'))
    se = sigma / np.sqrt(model.n)
    return dict(beta=fit.beta, se=se, lower=fit.beta - q * se,
                upper=fit.beta + q * se, q=q, psi=psi, residual=residual,
                maxima=maxima, min_eigenvalue=fit.min_eigenvalue)
