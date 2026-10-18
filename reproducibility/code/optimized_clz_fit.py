"""Algebraically identical CLZ fit with shared exact window moments.

This separate execution helper changes neither the estimator nor the variance:
observed-time normal equations are solved for the required coefficient vector,
unique symmetric raw XX marks are stored once, and window searches are shared.
The original frozen implementation remains the reference.
"""
from __future__ import annotations

from math import comb
import numpy as np
from scb_core import LocalLinear, WindowSums
from clz_source_audit import density_bandwidth


def shared_moments(prefix, grid, h, maximum):
    """Compute the original polynomial sums once for all requested moments."""
    grid = np.asarray(grid)
    lo = np.searchsorted(prefix.t, grid-h, side='left')
    hi = np.searchsorted(prefix.t, grid+h, side='right')
    sums = [v[hi]-v[lo] for v in prefix.prefix[:maximum+3]]
    c = (grid-.5)[:, None]
    powers = [sum(comb(a,k)*(-c)**(a-k)*sums[k] for k in range(a+1))/h**a
              for a in range(maximum+3)]
    return [(.75/h*(powers[k]-powers[k+2])).reshape((len(grid),)+prefix.shape)
            for k in range(maximum+1)]


def observed_beta(pooled, h):
    """Solve the same normal equations, retaining the original eigenvalue test."""
    s0,s1,s2 = shared_moments(pooled.xx,pooled.t,h,2)
    gram = np.concatenate([np.concatenate([s0,s1],axis=2),
                           np.concatenate([s1,s2],axis=2)],axis=1)
    rhs = np.concatenate(shared_moments(pooled.xy,pooled.t,h,1),axis=1)
    eig = np.linalg.eigvalsh(gram)
    if np.any(eig[:,0] <= 1e-10*np.maximum(eig[:,-1],1)):
        raise np.linalg.LinAlgError('Singular or ill-conditioned local design; no silent regularization.')
    return np.linalg.solve(gram,rhs[:,:,None])[:,:pooled.p,0]


def clz_fit(model, grid, h, density_h=None):
    """Original CLZ beta/SE/residual/NR-density-bandwidth return signature."""
    N,p = len(model.t),model.p
    # Equal-count subject weighting is exactly the original pooled weighting.
    target_weight = np.full(N,model.n/N)
    pooled = model if np.array_equal(model.w,target_weight) else LocalLinear(
        model.sid,model.t,model.x,model.y,weighting='observation')
    fit = pooled.fit(grid,h)
    beta_observed = observed_beta(pooled,h)
    residual = model.y-np.sum(model.x*beta_observed,axis=1)
    if density_h is None:
        density_h = density_bandwidth(model.t)
    f = WindowSums(model.t,np.ones(N),2).moment(grid,density_h,0)/N
    ii,jj = np.triu_indices(p)
    width = len(ii)
    xx = model.x[:,ii]*model.x[:,jj]
    marks = np.concatenate([xx,xx*residual[:,None]**2],axis=1)
    moments = WindowSums(model.t,marks,2).moment(grid,h,0)/N
    omega,lam = np.empty((len(grid),p,p)),np.empty((len(grid),p,p))
    omega[:,ii,jj] = omega[:,jj,ii] = moments[:,:width]
    lam[:,ii,jj] = lam[:,jj,ii] = moments[:,width:]
    inv = np.linalg.inv(omega)
    var = np.diagonal(inv@lam@inv,axis1=1,axis2=2)*.6/(N*h*f[:,None])
    if np.any(var <= 0) or not np.isfinite(var).all():
        raise FloatingPointError('Invalid CLZ variance')
    return fit.beta,np.sqrt(var),residual,float(density_h)
