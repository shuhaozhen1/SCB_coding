"""Sparse-matrix acceleration of the existing scalar Cai--Hu comparator.

Candidate knots, BIC, least squares, uncentered subject-score covariance,
Gaussian calibration, and empirical quantiles match comparison_design.cai_hu.
Only the representation of the B-spline and aggregation matrices changes.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline
from scipy.sparse import csr_matrix


def spline_basis(t, J):
    knots = np.r_[np.zeros(4), np.linspace(0, 1, J+2)[1:-1], np.ones(4)]
    return BSpline.design_matrix(t, knots, 3, extrapolate=False)


def fit_cai_hu(model, grid):
    """Fit and expose quantities needed for independent numerical checks."""
    N, n = len(model.t), model.n
    # Sorted points let the B-spline evaluator reuse neighboring knot spans.
    # Row permutation preserves every least-squares and subject-score sum.
    order = np.argsort(model.t, kind='stable')
    t = model.t[order]
    y = model.y[order]
    sid = model.sid[order]
    low = max(1, int(np.ceil(.5*N**(1/6))))
    high = max(low, int(np.floor(2*N**.25)))
    best = None
    candidates = []
    for J in range(low, high+1):
        A = spline_basis(t, J)
        gram = (A.T @ A).toarray()
        theta = np.linalg.solve(gram, A.T @ y)
        res = y - A @ theta
        bic = np.log(np.mean(res**2)) + J*np.log(N)/N
        candidates.append((J, float(bic)))
        if best is None or bic < best[0]:
            best = (bic, J, A, gram, theta, res)
    bic, J, A, gram, theta, res = best
    subject_map = csr_matrix((np.ones(N), (sid, np.arange(N))),
                             shape=(n, N))
    sums = (subject_map @ A.multiply(res[:, None])).toarray()
    influence = sums @ np.linalg.inv(gram)
    covariance = influence.T @ influence
    Ag = spline_basis(grid, J).toarray()
    se = np.sqrt(np.einsum('gi,ij,gj->g', Ag, covariance, Ag))
    val, vec = np.linalg.eigh(covariance)
    if val.min() < -1e-9:
        raise ValueError('Non-PSD spline covariance')
    root = vec*np.sqrt(np.maximum(val, 0))
    original_residual = np.empty(N)
    original_residual[order] = res
    return dict(beta=Ag @ theta, se=se, J=J, bic=float(bic),
                candidate_bic=candidates, covariance=covariance, root=root,
                grid_basis=Ag, theta=theta, residual=original_residual,
                subject_scores=sums, influence=influence)


def cai_hu(model, grid, rng, B):
    fit = fit_cai_hu(model, grid)
    root, Ag, se = fit['root'], fit['grid_basis'], fit['se']
    draws = rng.normal(size=(B, root.shape[0])) @ root.T @ Ag.T / se
    q = float(np.quantile(np.max(np.abs(draws), axis=1), .95,
                          method='inverted_cdf'))
    return fit['beta'], se, q, fit['J']
