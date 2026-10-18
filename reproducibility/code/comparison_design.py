"""Truth functions and published Cai--Hu spline implementation."""
import numpy as np
from scipy.interpolate import BSpline
GRID=np.linspace(.15,.85,201)

def truth(t,p):
    t=np.asarray(t)
    curves=[.5+.5*t,.5+.4*np.sin(2*np.pi*(t-.5)),.3*np.cos(2*np.pi*t),.5*t*t,.4*np.exp(t-1)]
    if p==1: return (np.sin(np.pi*t)+t+.25*(np.cos(2*np.pi*t)+np.sin(2*np.pi*t)))[:,None]
    return np.column_stack(curves[:p])

def spline_basis(t,J):
    knots=np.r_[np.zeros(4),np.linspace(0,1,J+2)[1:-1],np.ones(4)]
    return BSpline.design_matrix(t,knots,3,extrapolate=False).toarray()

def cai_hu(model,grid,rng,B):
    N=len(model.t); n=model.n
    # Cai--Hu arXiv version, Eq. (5.1); range depends on TOTAL N.
    low=max(1,int(np.ceil(.5*N**(1/6)))); high=max(low,int(np.floor(2*N**.25)))
    best=None
    for J in range(low,high+1):
        A=spline_basis(model.t,J); gram=A.T@A
        theta=np.linalg.solve(gram,A.T@model.y); res=model.y-A@theta
        bic=np.log(np.mean(res**2))+J*np.log(N)/N
        if best is None or bic<best[0]: best=(bic,J,A,gram,theta,res)
    _,J,A,gram,theta,res=best
    sums=np.zeros((n,A.shape[1]));np.add.at(sums,model.sid,A*res[:,None])
    influence=sums@np.linalg.inv(gram)
    C=influence.T@influence
    Ag=spline_basis(grid,J); se=np.sqrt(np.einsum('gi,ij,gj->g',Ag,C,Ag))
    val,vec=np.linalg.eigh(C)
    if val.min() < -1e-9: raise ValueError('Non-PSD spline covariance')
    root=vec*np.sqrt(np.maximum(val,0))
    draws=rng.normal(size=(B,len(val)))@root.T@Ag.T/se
    q=float(np.quantile(np.max(np.abs(draws),axis=1),.95,method='inverted_cdf'))
    return Ag@theta,se,q,J

