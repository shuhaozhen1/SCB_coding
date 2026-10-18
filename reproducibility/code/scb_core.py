"""Equal-subject local-linear bands, exactly matching the revised manuscript.

No ridge, leave-one-out residuals, influence centering, or separate bandwidth.
Prefix sums evaluate the Epanechnikov polynomial exactly (no binning).
"""
from __future__ import annotations
from dataclasses import dataclass
from math import comb
import numpy as np


def bandwidth(counts, scale=1.0):
    counts = np.asarray(counts, float)
    n = len(counts)
    mh = 1 / np.mean(1 / counts)
    return float(scale * max(n**(-0.25), (n*mh)**(-0.2)) / np.sqrt(np.log(np.e*n)))


def epan(u):
    return 0.75 * np.maximum(1-u*u, 0)


class WindowSums:
    def __init__(self, t, marks, degree):
        ix = np.argsort(t, kind='stable')
        self.t = np.asarray(t)[ix]
        marks = np.asarray(marks)[ix]
        self.shape = marks.shape[1:]
        marks = marks.reshape(len(t), -1)
        z = self.t - 0.5
        self.prefix = []
        for k in range(degree+1):
            vals = marks * z[:, None]**k
            self.prefix.append(np.vstack([np.zeros((1, marks.shape[1])), np.cumsum(vals, axis=0)]))

    def moment(self, grid, h, degree):
        grid = np.asarray(grid)
        lo = np.searchsorted(self.t, grid-h, side='left')
        hi = np.searchsorted(self.t, grid+h, side='right')
        c = grid - 0.5
        sums = [v[hi]-v[lo] for v in self.prefix[:degree+3]]
        def power(a):
            return sum(comb(a,k)*(-c[:,None])**(a-k)*sums[k] for k in range(a+1)) / h**a
        out = (0.75/h)*(power(degree)-power(degree+2))
        return out.reshape((len(grid),)+self.shape)


@dataclass
class Fit:
    beta: np.ndarray
    inverse: np.ndarray
    min_eigenvalue: float


class LocalLinear:
    def __init__(self, sid, t, x, y, weighting='subject'):
        _, self.sid = np.unique(sid, return_inverse=True)
        self.t = np.asarray(t, float)
        self.x = np.asarray(x, float)
        self.y = np.asarray(y, float)
        self.n = self.sid.max()+1
        self.p = self.x.shape[1]
        self.counts = np.bincount(self.sid)
        self.w = 1/self.counts[self.sid] if weighting=='subject' else np.full(len(t), self.n/len(t))
        xx = self.x[:,:,None]*self.x[:,None,:]*self.w[:,None,None]
        self.xx = WindowSums(self.t, xx, 4)
        self.xy = WindowSums(self.t, self.x*(self.w*self.y)[:,None], 3)

    def fit(self, grid, h):
        s0, s1, s2 = [self.xx.moment(grid,h,k) for k in range(3)]
        gram = np.concatenate([np.concatenate([s0,s1],axis=2), np.concatenate([s1,s2],axis=2)],axis=1)
        rhs = np.concatenate([self.xy.moment(grid,h,0), self.xy.moment(grid,h,1)],axis=1)
        eig = np.linalg.eigvalsh(gram)
        if np.any(eig[:,0] <= 1e-10*np.maximum(eig[:,-1],1)):
            raise np.linalg.LinAlgError('Singular or ill-conditioned local design; no silent regularization.')
        inverse = np.linalg.inv(gram)
        beta = np.einsum('gij,gj->gi', inverse, rhs)[:,:self.p]
        return Fit(beta, inverse, float(eig[:,0].min()/self.n))

    def contributions(self, grid, h, fit=None):
        if fit is None: fit = self.fit(grid,h)
        observed = self.fit(self.t,h)
        residual = self.y-np.sum(self.x*observed.beta,axis=1)
        psi = np.empty((self.n,len(grid),self.p))
        for i in range(self.n):
            ix = self.sid==i
            wxr = self.x[ix]*(self.w[ix]*residual[ix])[:,None]
            pref = WindowSums(self.t[ix],wxr,3)
            score = np.concatenate([pref.moment(grid,h,0),pref.moment(grid,h,1)],axis=1)
            psi[i] = self.n*np.einsum('gij,gj->gi',fit.inverse[:,:self.p,:],score)
        return psi, residual


def multiplier_band(model, grid, h, rng, B=1999, alpha=.05, components=None, multipliers=None):
    fit = model.fit(grid,h)
    psi, residual = model.contributions(grid,h,fit)
    sigma = np.sqrt(np.mean(psi**2,axis=0))
    if np.any(sigma<=1e-12): raise FloatingPointError('Degenerate fitted variance')
    standardized = psi/sigma[None,:,:]
    if multipliers is None: multipliers = rng.normal(size=(B,model.n))
    process = (multipliers @ standardized.reshape(model.n,-1)/np.sqrt(model.n)).reshape(len(multipliers),len(grid),model.p)
    if components is None: components = np.arange(model.p)
    maxima = np.max(np.abs(process[:,:,components]),axis=(1,2))
    # Empirical inverse CDF, not interpolated percentile.
    q = float(np.quantile(maxima,1-alpha,method='inverted_cdf'))
    se = sigma/np.sqrt(model.n)
    return dict(beta=fit.beta,se=se,lower=fit.beta-q*se,upper=fit.beta+q*se,q=q,
                psi=psi,residual=residual,maxima=maxima,min_eigenvalue=fit.min_eigenvalue)


def coverage(beta,se,q,truth):
    return bool(np.all(np.abs(beta-truth)<=q*se))
