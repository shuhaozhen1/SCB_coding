"""Cao, Liu and Zhou (2018), Section 3; exact finite-N calibration.
Uniform design in the experiments. N denotes the total observation count.
"""
from pathlib import Path
import json
import numpy as np
from scb_core import WindowSums, bandwidth


def cao_se(model, grid, h, residual):
    N=len(model.t)
    f=WindowSums(model.t,np.ones(N),2).moment(grid,h,0)/N
    omega=WindowSums(model.t,model.x[:,:,None]*model.x[:,None,:],2).moment(grid,h,0)/N
    lam=WindowSums(model.t,model.x[:,:,None]*model.x[:,None,:]*residual[:,None,None]**2,2).moment(grid,h,0)/N
    inv=np.linalg.inv(omega)
    return np.sqrt(np.diagonal(inv@lam@inv,axis1=1,axis2=2)*.6/(N*h*f[:,None]))


def kernel_draw(t, eta, grid, h):
    # t is sorted. Polynomial prefix sums are exact, with no binning.
    z=t-.5; c=grid-.5
    pref=[np.r_[0.,np.cumsum(eta*z**j)] for j in range(3)]
    lo=np.searchsorted(t,grid-h);hi=np.searchsorted(t,grid+h,side='right')
    s=[p[hi]-p[lo] for p in pref]
    return .75*(s[0]-(s[2]-2*c*s[1]+c*c*s[0])/h**2)/np.sqrt(.6*len(t)*h)


def calibrate(job):
    n,m,B,out=job;N=n*m;h=bandwidth(np.full(n,m));grid=np.linspace(.15,.85,201)
    rng=np.random.default_rng(np.random.SeedSequence([713209,n,m]))
    maxima=np.empty(B)
    for k in range(B):
        spacings=np.cumsum(rng.exponential(size=N+1))
        t=spacings[:-1]/spacings[-1]
        eta=rng.normal(size=N)
        maxima[k]=np.max(np.abs(kernel_draw(t,eta,grid,h)))
    np.save(Path(out)/f'cao_maxima_n{n}_m{m}.npy',maxima)
    return dict(n=n,m=m,N=N,h=h,B=B,q95=float(np.quantile(maxima,.95,method='inverted_cdf')),q99=float(np.quantile(maxima,.99,method='inverted_cdf')))

if __name__=='__main__':
    from concurrent.futures import ProcessPoolExecutor
    out=Path(__file__).resolve().parents[1]/'results/regular';out.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=6) as pool:
        rows=list(pool.map(calibrate,[(n,m,20000,str(out)) for n in (200,400) for m in (5,20,80)]))
    (out/'cao_calibration.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps(rows,indent=2))
