"""Exact common-moment acceleration of CLZ whole-subject prediction CV.

Only unique symmetric XX entries are stored. Global-minus-subject prefix
differences are formed once and translated to kernel moments 0,1,2 together.
The candidate grid, data, full observation CV event, and minimization criterion
are unchanged. This helper is separate from the original frozen source code.
"""
from __future__ import annotations

import numpy as np
from clz_source_audit import LeaveSubjectMoments
from scb_core import WindowSums


def loso_select(model, candidates):
    candidates = np.asarray(candidates, float)
    p, N = model.p, len(model.t)
    ii, jj = np.triu_indices(p)
    size = len(ii)
    marks = np.concatenate([model.x[:, ii]*model.x[:, jj],
                            model.x*model.y[:, None]], axis=1)
    total = WindowSums(model.t, marks, 4)
    own = LeaveSubjectMoments(model, marks, 4)
    c = (model.t-.5)[:, None]
    c2, c3, c4 = c*c, c*c*c, c*c*c*c
    losses = []
    for h in candidates:
        lo = np.searchsorted(total.t, model.t-h)
        hi = np.searchsorted(total.t, model.t+h, side='right')
        own_lo = np.searchsorted(own.sorted, own.shifted-h)
        own_hi = np.searchsorted(own.sorted, own.shifted+h, side='right')
        s0, s1, s2, s3, s4 = [(a[hi]-a[lo])-(b[own_hi]-b[own_lo])
                              for a, b in zip(total.prefix, own.prefix)]
        second = (s2-2*c*s1+c2*s0)/h**2
        m0 = .75/h*(s0-second)
        m1 = .75/h*((s1-c*s0)/h-(s3-3*c*s2+3*c2*s1-c3*s0)/h**3)
        m2 = .75/h*(second-(s4-4*c*s3+6*c2*s2-4*c3*s1+c4*s0)/h**4)
        gram = np.empty((N, 2*p, 2*p))
        gram[:, ii, jj] = m0[:, :size]
        gram[:, jj, ii] = m0[:, :size]
        gram[:, ii+p, jj+p] = m2[:, :size]
        gram[:, jj+p, ii+p] = m2[:, :size]
        gram[:, ii, jj+p] = m1[:, :size]
        gram[:, jj, ii+p] = m1[:, :size]
        gram[:, ii+p, jj] = m1[:, :size]
        gram[:, jj+p, ii] = m1[:, :size]
        rhs = np.concatenate([m0[:, size:], m1[:, size:]], axis=1)
        try:
            theta = np.linalg.solve(gram, rhs[:, :, None])[:, :, 0]
            prediction = np.sum(model.x*theta[:, :p], axis=1)
            loss = float(np.mean((model.y-prediction)**2))
            if not np.isfinite(loss):
                loss = float('inf')
        except np.linalg.LinAlgError:
            loss = float('inf')
        losses.append(loss)
    losses = np.asarray(losses)
    if not np.isfinite(losses).any():
        raise np.linalg.LinAlgError('All CLZ LOSO bandwidth candidates failed')
    return float(candidates[int(np.argmin(losses))]), losses
