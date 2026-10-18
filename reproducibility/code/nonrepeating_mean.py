"""Exact joint bands for distinct bounded sine/cosine means and one shared error."""
import time
import numpy as np
from scipy.spatial import cKDTree
from highd_mean import local_weights
from scb_core import LocalLinear, WindowSums

BOUNDS = ((.75, 1.25), (-.25, .25))


def coefficients(p):
    k = np.arange(1, p+1, dtype=float)
    a = .75+.5*np.remainder(k*(np.sqrt(5)-1)/2, 1)
    b = -.25+.5*np.remainder(k*(np.sqrt(2)-1), 1)
    return np.vstack([np.ones(p), a, b])


def mean_truth(t, coordinates):
    coordinates = np.asarray(coordinates, int)
    c = coefficients(int(coordinates.max())+1)[:, coordinates]
    return np.column_stack([np.sin(np.pi*np.asarray(t)),
                            np.cos(2*np.pi*np.asarray(t))])@c[1:]


def rectangle_maximum(v, covariance):
    """Max |v.(1,a,b)|/sqrt((1,a,b)'V(1,a,b)) over a fixed rectangle.

    A maximum occurs at a corner, a stationary point of an edge, or an
    interior direction proportional to V^{-1}v. Singular V needs no interior
    candidate: an interior maximizer can be extended to the boundary.
    """
    v = np.atleast_2d(v)
    maximum = np.zeros(len(v))

    def consider(c, valid=None):
        nonlocal maximum
        c = np.broadcast_to(c, v.shape)
        variance = np.einsum('bi,ij,bj->b', c, covariance, c)
        if np.any(variance <= 0):
            raise FloatingPointError('Nonpositive variance in coefficient envelope.')
        values = np.abs(np.einsum('bi,bi->b', v, c))/np.sqrt(variance)
        if valid is not None:
            values = np.where(valid, values, 0)
        maximum = np.maximum(maximum, values)

    for a in BOUNDS[0]:
        for b in BOUNDS[1]:
            consider(np.array([1., a, b]))
    for fixed, variable in [(1, 2), (2, 1)]:
        for value in BOUNDS[fixed-1]:
            e = np.array([1., 0., 0.]); e[fixed] = value
            A = v@e
            B = v[:, variable]
            C = e@covariance@e
            D = e@covariance[:, variable]
            E = covariance[variable, variable]
            denominator = B*D-A*E
            x = np.divide(A*D-B*C, denominator, out=np.full(len(v), np.nan),
                          where=denominator != 0)
            lo, hi = BOUNDS[variable-1]
            valid = np.isfinite(x) & (x >= lo) & (x <= hi)
            candidate = np.broadcast_to(e, v.shape).copy()
            candidate[:, variable] = np.where(valid, x, (lo+hi)/2)
            consider(candidate, valid)
    try:
        stationary = np.linalg.solve(covariance, v.T).T
        stationary = np.divide(stationary, stationary[:, :1],
                               out=np.full_like(stationary, np.nan),
                               where=stationary[:, :1] != 0)
        valid = np.isfinite(stationary).all(axis=1)
        for j, (lo, hi) in enumerate(BOUNDS, 1):
            valid &= (stationary[:, j] >= lo) & (stationary[:, j] <= hi)
        stationary[~valid] = [1., 1., 0.]
        consider(stationary, valid)
    except np.linalg.LinAlgError:
        pass
    return maximum


def exact_direction_maximum(z, directions, method='auto'):
    """All-coordinate maximum; the exact tree query only accelerates the search."""
    if method == 'direct' or (method == 'auto' and directions.shape[1] <= 256):
        return np.max(np.abs(z@directions), axis=1)
    norm = np.linalg.norm(z, axis=1)
    query = np.divide(z, norm[:, None], out=np.zeros_like(z), where=norm[:, None] > 0)
    points = np.concatenate([directions.T, -directions.T], axis=0)
    tree = cKDTree(points)
    _, selected = tree.query(query, k=1, eps=0, workers=1)
    return np.abs(np.einsum('ij,ij->i', z, directions[:, selected % directions.shape[1]].T))


class NonrepeatingMeanFit:
    def __init__(self, sid, t, grid, h, multipliers):
        _, self.sid = np.unique(sid, return_inverse=True)
        self.n = int(self.sid.max()+1)
        self.t = np.asarray(t, float)
        self.grid = np.asarray(grid, float)
        self.h = float(h)
        self.xi = np.asarray(multipliers, float)
        self.weights, eg = local_weights(self.t, self.sid, self.grid, h)
        scalar = LocalLinear(self.sid, self.t, np.ones((len(t), 1)), np.zeros(len(t)))
        observed = scalar.fit(self.t, h)
        self.inverse_row = observed.inverse[:, 0, :]
        self.observation_weights = scalar.w
        self.min_eigenvalue = min(eg, observed.min_eigenvalue)

    def calibrate(self, error, dimensions, method='auto', return_maxima=False):
        start = time.perf_counter()
        requested = sorted(set(dimensions))
        dims = sorted(set([50]+requested))
        c = coefficients(max(dims))
        basis = np.column_stack([error, np.sin(np.pi*self.t), np.cos(2*np.pi*self.t)])
        moments = WindowSums(self.t, basis*self.observation_weights[:, None], 3)
        fitted = (self.inverse_row[:, 0, None]*moments.moment(self.t, self.h, 0)
                  +self.inverse_row[:, 1, None]*moments.moment(self.t, self.h, 1))
        residual = basis-fitted
        basis_estimate = self.weights@basis
        bias_basis = basis_estimate[:, 1:]-np.column_stack([
            np.sin(np.pi*self.grid), np.cos(2*np.pi*self.grid)])
        maxima = {p:np.zeros(len(self.xi)) for p in dims}
        diagnostic = {p:dict(observed=0., sum_se=0., min_se=np.inf, max_se=0., bias_se=0.) for p in dims}
        envelope = np.zeros(len(self.xi))
        envelope_observed = 0.
        for g, weights in enumerate(self.weights):
            obs = np.flatnonzero(weights != 0)
            scores = np.column_stack([np.bincount(
                self.sid[obs], weights=weights[obs]*residual[obs, j], minlength=self.n)
                for j in range(3)])
            u, singular, vt = np.linalg.svd(scores, full_matrices=False)
            transformed = singular[:, None]*(vt@c)
            se = np.linalg.norm(transformed, axis=0)
            if np.any(se <= 1e-12) or not np.isfinite(se).all():
                raise FloatingPointError('Degenerate coordinate variance.')
            directions = transformed/se[None, :]
            z = self.xi@u
            v = np.r_[basis_estimate[g, 0], bias_basis[g]]
            observed = np.abs(v@c)/se
            bias = np.abs(bias_basis[g]@c[1:])/se
            covariance = scores.T@scores
            envelope = np.maximum(envelope, rectangle_maximum(
                (z*singular[None, :])@vt, covariance))
            envelope_observed = max(envelope_observed, float(rectangle_maximum(v, covariance)[0]))
            for p in dims:
                maxima[p] = np.maximum(maxima[p], exact_direction_maximum(z, directions[:, :p], method))
                d = diagnostic[p]
                d['observed'] = max(d['observed'], float(observed[:p].max()))
                d['sum_se'] += float(se[:p].sum())
                d['min_se'] = min(d['min_se'], float(se[:p].min()))
                d['max_se'] = max(d['max_se'], float(se[:p].max()))
                d['bias_se'] = max(d['bias_se'], float(bias[:p].max()))
        q50 = float(np.quantile(maxima[50], .95, method='inverted_cdf'))
        q_envelope = float(np.quantile(envelope, .95, method='inverted_cdf'))
        rows = []
        for p in requested:
            d = diagnostic[p]
            if np.min(envelope-maxima[p]) < -2e-8 or envelope_observed < d['observed']-2e-8:
                raise FloatingPointError('Continuum envelope failed its finite-set bound.')
            q = float(np.quantile(maxima[p], .95, method='inverted_cdf'))
            row = dict(p=p, cp=int(d['observed'] <= q), q=q,
                       aw=2*q*d['sum_se']/(len(self.grid)*p),
                       observed_maximum=d['observed'], min_se=d['min_se'], max_se=d['max_se'],
                       max_bias_se=d['bias_se'], min_eigenvalue=self.min_eigenvalue,
                       q50=q50, q_envelope=q_envelope, statistic_envelope=envelope_observed,
                       cp_envelope=int(envelope_observed <= q_envelope),
                       cp_all_p_lowerbound=int(envelope_observed <= q50),
                       calibration_seconds=time.perf_counter()-start)
            if return_maxima:
                row['maxima'] = maxima[p].copy()
                row['envelope_maxima'] = envelope.copy()
            rows.append(row)
        return rows
