"""Exact local-linear joint bands for genuinely coordinate-specific responses.

Every response column is fitted, residualized and included in the multiplier
maximum. There is no shared-error rank shortcut, coordinate subsampling,
individual-curve smoothing, binning, ridge or variance correction.
"""
from __future__ import annotations

import time
import numpy as np

from highd_mean import local_weights
from scb_core import LocalLinear, WindowSums


class CoordinateMeanFit:
    """Full N-by-p response fit with common observation times and bandwidth."""

    def __init__(self, sid, t, y, grid, h, multipliers, observed_chunk_size=2048):
        _,self.sid=np.unique(sid,return_inverse=True)
        self.t=np.asarray(t,dtype=float)
        self.y=np.asarray(y,dtype=float)
        self.grid=np.asarray(grid,dtype=float)
        self.h=float(h)
        self.xi=np.asarray(multipliers,dtype=float)
        self.n=int(self.sid.max()+1)
        self.counts=np.bincount(self.sid,minlength=self.n)
        if self.y.ndim!=2 or self.y.shape[0]!=len(self.t) or len(self.sid)!=len(self.t):
            raise ValueError('Require one response row and subject index per observation.')
        if self.xi.ndim!=2 or self.xi.shape[1]!=self.n or self.h<=0:
            raise ValueError('Require positive h and B-by-n subject multipliers.')
        if not all(np.isfinite(z).all() for z in (self.t,self.y,self.grid,self.xi)):
            raise ValueError('Nonfinite input.')
        self.p=self.y.shape[1]
        started=time.perf_counter()
        self.weights,grid_eigenvalue=local_weights(self.t,self.sid,self.grid,self.h)
        scalar=LocalLinear(self.sid,self.t,np.ones((len(self.t),1)),np.zeros(len(self.t)))
        observed=scalar.fit(self.t,self.h)
        self.min_eigenvalue=min(grid_eigenvalue,observed.min_eigenvalue)
        # The same full-sample local-linear operator is applied to every column.
        prefix=WindowSums(self.t,self.y*scalar.w[:,None],3)
        fitted=np.empty_like(self.y)
        for start in range(0,len(self.t),observed_chunk_size):
            stop=min(start+observed_chunk_size,len(self.t));points=self.t[start:stop]
            fitted[start:stop]=(observed.inverse[start:stop,0,0,None]*prefix.moment(points,self.h,0)
                               +observed.inverse[start:stop,0,1,None]*prefix.moment(points,self.h,1))
        self.residual=self.y-fitted
        del prefix,fitted
        self.estimate=self.weights@self.y
        # Aggregate exact estimator-weighted residuals by subject. Batched GEMM
        # keeps every coordinate and all observations; equal counts are merely
        # an efficient storage layout, not an independence assumption.
        order=np.argsort(self.sid,kind='stable')
        if np.all(self.counts==self.counts[0]):
            m=int(self.counts[0])
            w=self.weights[:,order].reshape(len(self.grid),self.n,m).transpose(1,0,2)
            r=self.residual[order].reshape(self.n,m,self.p)
            self.scores=np.ascontiguousarray(w@r)
            self.score_aggregation='Exact batched n-by-(grid,m) times n-by-(m,p) GEMM'
        else:
            self.scores=np.empty((self.n,len(self.grid),self.p))
            for i in range(self.n):
                mask=self.sid==i
                self.scores[i]=self.weights[:,mask]@self.residual[mask]
            self.score_aggregation='Exact subject-block GEMM for unequal counts'
        self.se=np.sqrt(np.sum(self.scores**2,axis=0))
        if np.any(self.se<=1e-12) or not np.isfinite(self.se).all():
            raise FloatingPointError('Degenerate coordinate standard error.')
        self.fitting_seconds=time.perf_counter()-started

    def calibrate(self, truth, dimensions=(50,100), return_arrays=False):
        """Joint maxima for nested coordinate prefixes on the entire grid.

        Scores already contain the estimator weights; no additional sqrt(n)
        divisor is applied to either standard errors or multiplier processes.
        """
        started=time.perf_counter()
        truth=np.asarray(truth,dtype=float)
        dimensions=sorted(set(int(p) for p in dimensions))
        if truth.shape!=self.estimate.shape or not dimensions or min(dimensions)<1 or max(dimensions)>self.p:
            raise ValueError('Truth/dimension mismatch.')
        standardized=self.scores/self.se[None,:,:]
        process=self.xi@standardized.reshape(self.n,-1)
        coordinate_maxima=np.abs(process.reshape(len(self.xi),len(self.grid),self.p)).max(axis=1)
        del process,standardized
        prefix_maxima=np.maximum.accumulate(coordinate_maxima,axis=1)
        observed=np.abs(self.estimate-truth)/self.se
        rows=[];saved={}
        for p in dimensions:
            maxima=prefix_maxima[:,p-1]
            q=float(np.quantile(maxima,.95,method='inverted_cdf'))
            statistic=float(observed[:,:p].max())
            rows.append(dict(p=p,cp=int(statistic<=q),q=q,
                aw=float(2*q*self.se[:,:p].mean()),observed_maximum=statistic,
                min_se=float(self.se[:,:p].min()),max_se=float(self.se[:,:p].max()),
                min_eigenvalue=self.min_eigenvalue,fitting_seconds=self.fitting_seconds,
                calibration_seconds=time.perf_counter()-started,failure=''))
            if return_arrays:saved['bootstrap_maxima_p'+str(p)]=maxima.copy()
        if not return_arrays:return rows
        arrays=dict(grid=self.grid.copy(),truth=truth.copy(),estimate=self.estimate.copy(),
                    se=self.se.copy(),residual=self.residual.copy(),
                    subject_scores=self.scores.copy(),multipliers=self.xi.copy(),
                    coordinate_bootstrap_maxima=coordinate_maxima,**saved)
        for row in rows:
            p=row['p'];q=row['q']
            arrays['lower_p'+str(p)]=self.estimate[:,:p]-q*self.se[:,:p]
            arrays['upper_p'+str(p)]=self.estimate[:,:p]+q*self.se[:,:p]
        return rows,arrays
