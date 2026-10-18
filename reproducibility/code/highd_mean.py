"""Blockwise implementation of the manuscript's multivariate mean bands.

The scalar local-linear design is shared by all responses. Contributions use
fitted residuals at every observed time, uncentered subject squares, and the
same multiplier for every coordinate of a subject. No covariance approximation,
ridge, bias correction, coordinate subsampling, or marginal-band adjustment.
"""
from __future__ import annotations

import time
import numpy as np
from scb_core import epan, LocalLinear, WindowSums


def mean_truth(t, coordinates):
    t = np.asarray(t, float)
    phase = 2 * np.pi * (np.asarray(coordinates) % 20) / 20
    return (np.sin(np.pi * t[:, None])
            + 0.5 * t[:, None] * np.cos(phase)
            + 0.25 * t[:, None]**2 * np.sin(phase))


def local_weights(t, sid, evaluation, h):
    counts = np.bincount(sid)
    u = (np.asarray(t)[None, :] - np.asarray(evaluation)[:, None]) / h
    kernel = epan(u) / h / counts[sid][None, :]
    s0 = kernel.sum(axis=1)
    s1 = (kernel * u).sum(axis=1)
    s2 = (kernel * u**2).sum(axis=1)
    gram = np.empty((len(evaluation), 2, 2))
    gram[:, 0, 0], gram[:, 0, 1] = s0, s1
    gram[:, 1, 0], gram[:, 1, 1] = s1, s2
    eigenvalues = np.linalg.eigvalsh(gram)
    if np.any(eigenvalues[:, 0] <= 1e-10 * np.maximum(eigenvalues[:, 1], 1)):
        raise np.linalg.LinAlgError('Singular local mean design; no regularization.')
    weights = kernel * (s2[:, None] - s1[:, None] * u)
    weights /= (s0 * s2 - s1**2)[:, None]
    return weights, float(eigenvalues[:, 0].min() / len(counts))


class SharedMeanFit:
    def __init__(self, sid, t, grid, h, multipliers):
        _, self.sid = np.unique(sid, return_inverse=True)
        self.n = int(self.sid.max() + 1)
        self.t = np.asarray(t, float)
        self.grid = np.asarray(grid, float)
        self.h = float(h)
        self.multipliers = np.asarray(multipliers, float)
        self.weights, eg = local_weights(self.t, self.sid, self.grid, h)
        # Exact polynomial prefix sums avoid an observations-by-observations
        # smoothing matrix when each subject has repeated observations.
        scalar = LocalLinear(self.sid, self.t, np.ones((len(self.t),1)),
                             np.zeros(len(self.t)))
        observed = scalar.fit(self.t, h)
        self.observed_inverse_row = observed.inverse[:,0,:]
        self.observation_weights = scalar.w
        self.min_eigenvalue = min(eg, observed.min_eigenvalue)
        self.windows = []
        for weights in self.weights:
            obs = np.flatnonzero(weights != 0)
            subjects, inverse = np.unique(self.sid[obs], return_inverse=True)
            self.windows.append((obs, weights[obs], inverse, len(subjects),
                                 np.ascontiguousarray(self.multipliers[:, subjects])))
        basis = lambda x: np.column_stack([np.sin(np.pi*x), x, x*x])
        self.bias_basis = self.weights @ basis(self.t) - basis(self.grid)

    def fitted_block(self, response):
        moments = WindowSums(self.t, response*self.observation_weights[:,None], 3)
        fitted = (self.observed_inverse_row[:,0,None]*moments.moment(self.t,self.h,0)
                  +self.observed_inverse_row[:,1,None]*moments.moment(self.t,self.h,1))
        return self.weights @ response, response-fitted

    def calibrate_nested(self, response_block, dimensions, block_size=128,
                         alpha=0.05, return_maxima=False):
        dimensions = tuple(sorted(set(int(p) for p in dimensions)))
        if not dimensions or dimensions[0] < 1:
            raise ValueError('Positive response dimensions are required.')
        maxima = np.zeros(len(self.multipliers))
        observed_maximum = 0.0
        sum_se = 0.0
        min_se, max_se = float('inf'), 0.0
        max_bias_se = 0.0
        rows = []
        start = time.perf_counter()
        lo = 0
        for dimension in dimensions:
            while lo < dimension:
                hi = min(lo + block_size, dimension)
                coordinates = np.arange(lo, hi)
                response = np.asarray(response_block(lo, hi), float)
                if response.shape != (len(self.t), hi-lo):
                    raise ValueError('Incorrect response-block shape.')
                beta, residual = self.fitted_block(response)
                truth = mean_truth(self.grid, coordinates)
                phase = 2*np.pi*(coordinates % 20)/20
                coefficients = np.vstack([np.ones(hi-lo), 0.5*np.cos(phase),
                                           0.25*np.sin(phase)])
                bias = self.bias_basis @ coefficients
                for g, (obs, weights, inverse, ns, xi) in enumerate(self.windows):
                    weighted_residual = weights[:, None] * residual[obs]
                    if ns == len(obs) and np.array_equal(inverse, np.arange(ns)):
                        scores = weighted_residual
                    else:
                        scores = np.zeros((ns, hi-lo))
                        np.add.at(scores, inverse, weighted_residual)
                    se = np.sqrt(np.einsum('ik,ik->k', scores, scores))
                    if np.any(se <= 1e-12) or not np.isfinite(se).all():
                        raise FloatingPointError('Degenerate fitted mean variance.')
                    process = (xi @ scores) / se[None, :]
                    maxima = np.maximum(maxima, np.max(np.abs(process), axis=1))
                    observed_maximum = max(observed_maximum,
                                           float(np.max(np.abs(beta[g]-truth[g])/se)))
                    sum_se += float(se.sum())
                    min_se = min(min_se, float(se.min()))
                    max_se = max(max_se, float(se.max()))
                    max_bias_se = max(max_bias_se, float(np.max(np.abs(bias[g])/se)))
                lo = hi
            critical = float(np.quantile(maxima, 1-alpha, method='inverted_cdf'))
            row = dict(p=dimension, cp=int(observed_maximum <= critical),
                       aw=2*critical*sum_se/(len(self.grid)*dimension), q=critical,
                       observed_maximum=observed_maximum, min_se=min_se,
                       max_se=max_se, max_bias_se=max_bias_se,
                       min_eigenvalue=self.min_eigenvalue,
                       calibration_seconds=time.perf_counter()-start)
            if return_maxima:
                row['maxima'] = maxima.copy()
            rows.append(row)
        return rows
