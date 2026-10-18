"""Portable entry to the current paper's frozen numerical methods and seeds.

The engines copied alongside this file are unchanged. Historical production
runners are deliberately excluded: their machine-specific checkpoint protocols
are replaced by the portable driver at ../reproduce.py.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scb_core import LocalLinear, bandwidth
from optimized_vcm_engine import multiplier_band
from optimized_clz_fit import clz_fit
from optimized_clz_loso import loso_select
from optimized_cai_hu import cai_hu
from coordinate_mean_engine import CoordinateMeanFit
from comparison_design import truth
from nonrepeating_mean import mean_truth
from error_laws import LAWS, draw_innovations

ROOT=Path(__file__).resolve().parents[1]
GRID=np.linspace(.15,.85,51)
H=np.linspace(.0601,.0899,7)
SCENARIOS=('single','joint','scalar','highd')
SCALES={2:{5:1.25,20:.8,80:.65},5:{5:1.5,20:1.25,80:.65},1:{5:1.,20:1.,80:1.}}
_CRITICAL=None

def critical_values():
    global _CRITICAL
    if _CRITICAL is None:
        rows=json.loads((ROOT/'reference/clz-critical-values.json').read_text())
        _CRITICAL={(r['n'],r['m'],r['h_index'],r['level']):r['q'] for r in rows}
    return _CRITICAL

def generate_vcm(n,m,p,law,rep,stage=2):
    children=np.random.SeedSequence([20261018,771211,stage,n,m,p,LAWS.index(law),rep]).spawn(3)
    rng=np.random.default_rng(children[0])
    sid=np.repeat(np.arange(n),m);t=rng.uniform(size=n*m)
    basis=np.column_stack([np.ones(len(t)),np.sin(2*np.pi*t),np.cos(2*np.pi*t)])
    x=np.ones((len(t),p))
    if p>1:
        a=rng.normal(size=(n,p-1,3));x[:,1:]=np.einsum('ijk,ik->ij',a[sid],basis)/np.sqrt(2)
    z=draw_innovations(rng,(n,3),law)
    epsilon=np.sum(z[sid]*basis*np.array([1,.5,.5]),axis=1)
    y=np.sum(x*truth(t,p),axis=1)+epsilon
    model=LocalLinear(sid,t,x,y)
    xi=np.random.default_rng(children[1]).normal(size=(2000,n))
    return model,xi,np.random.default_rng(children[2])

def generate_mean(n,m,law,rep):
    children=np.random.SeedSequence([20261018,184700,n,m,LAWS.index(law),rep]).spawn(4)
    t=np.random.default_rng(children[0]).uniform(size=n*m);sid=np.repeat(np.arange(n),m)
    common=draw_innovations(np.random.default_rng(children[1]),(n,3),law)
    specific=draw_innovations(np.random.default_rng(children[2]),(100,n,3),law)
    basis=np.column_stack([np.ones(len(t)),.5*np.sin(2*np.pi*t),.5*np.cos(2*np.pi*t)])
    marks=(specific+common[None,:,:]).transpose(1,0,2)
    errors=(np.einsum('nmb,npb->nmp',basis.reshape(n,m,3),marks)/np.sqrt(2)).reshape(n*m,100)
    y=mean_truth(t,np.arange(100))+errors
    xi=np.random.default_rng(children[3]).normal(size=(1000,n))
    return sid,t,y,xi

def summarize_band(result,p,grid):
    columns=[1] if p==2 else list(range(p));se=result['se'][:,columns]
    statistic=float(np.max(abs(result['beta'][:,columns]-truth(grid,p)[:,columns])/se))
    return dict(p=p,cp=int(statistic<=result['q']),aw=float(2*result['q']*se.mean()),
                q=result['q'],observed_maximum=statistic,failure='')

def failed(p,error):
    return dict(p=p,cp=0,aw=None,q=None,failure=type(error).__name__+': '+str(error))

def one(job):
    scenario,n,m,law,rep=job
    if scenario not in SCENARIOS or law not in LAWS:raise ValueError('Unknown study or law')
    row=dict(scenario=scenario,n=n,m=m,law=law,rep=rep,B=1000 if scenario=='highd' else 2000)
    if scenario=='highd':
        sid,t,y,xi=generate_mean(n,m,law,rep);h=bandwidth(np.full(n,m))
        try:
            fit=CoordinateMeanFit(sid,t,y,GRID,h,xi)
            values=fit.calibrate(mean_truth(GRID,np.arange(100)),(50,100))
            row['proposed']=[dict(z,h=h,scale=1.) for z in values]
        except (np.linalg.LinAlgError,FloatingPointError) as err:
            row['proposed']=[dict(failed(p,err),h=h,scale=1.) for p in (50,100)]
        row['comparator']=None
        return row
    p={'single':2,'joint':5,'scalar':1}[scenario]
    model,xi,ref_rng=generate_vcm(n,m,p,law,rep)
    scale=SCALES[p][m];h=bandwidth(model.counts)*scale
    try:
        fit=multiplier_band(model,GRID,h,B=2000,components=[1] if p==2 else list(range(p)),multipliers=xi)
        proposed=summarize_band(fit,p,GRID)
    except (np.linalg.LinAlgError,FloatingPointError) as err:proposed=failed(p,err)
    row['proposed']=[dict(proposed,h=h,scale=scale)]
    if p==1:
        try:
            beta,se,q,J=cai_hu(model,GRID,ref_rng,2000)
            statistic=float(np.max(abs(beta-truth(GRID,1)[:,0])/se))
            row['comparator']=dict(p=1,cp=int(statistic<=q),aw=float(np.mean(2*q*se)),q=q,J=J,observed_maximum=statistic,failure='')
        except (np.linalg.LinAlgError,FloatingPointError) as err:row['comparator']=failed(1,err)
    else:
        try:
            hc,losses=loso_select(model,H);index=int(np.argmin(losses));level=.95 if p==2 else .99
            q=critical_values()[n,m,index,level];beta,se,_,b=clz_fit(model,GRID,hc)
            columns=[1] if p==2 else list(range(p));statistic=float(np.max(abs(beta[:,columns]-truth(GRID,p)[:,columns])/se[:,columns]))
            row['comparator']=dict(p=p,cp=int(statistic<=q),aw=float(np.mean(2*q*se[:,columns])),q=q,h=hc,h_index=index,b=b,c=hc,d=hc,critical_level=level,observed_maximum=statistic,cv_losses=[float(x) if np.isfinite(x) else None for x in losses],failure='')
        except (np.linalg.LinAlgError,FloatingPointError) as err:row['comparator']=failed(p,err)
    return row

def calibration_chunk(job):
    """Exactly the original 1000-draw chunk and shared seven-bandwidth seeds."""
    n,m,chunk=job;N=n*m
    rng=np.random.default_rng(np.random.SeedSequence([20261018,952407,n,m,chunk]))
    values=np.empty((1000,len(H)));hs=H[:,None];center=GRID[None,:]-.5
    for b in range(1000):
        spacings=np.cumsum(rng.exponential(size=N+1));t=spacings[:-1]/spacings[-1]
        eta=rng.normal(size=N);z=t-.5
        prefix=[np.r_[0.,np.cumsum(eta*z**k)] for k in range(3)]
        lo=np.searchsorted(t,GRID[None,:]-hs);hi=np.searchsorted(t,GRID[None,:]+hs,side='right')
        s0,s1,s2=[a[hi]-a[lo] for a in prefix]
        process=.75*(s0-(s2-2*center*s1+center*center*s0)/hs**2)/np.sqrt(.6*N*hs)
        values[b]=np.max(np.abs(process),axis=1)
    return n,m,chunk,values

def illustration(scenario,rep=0):
    grid=np.linspace(0,1,51);visual=np.linspace(0,1,1001);n,m=400,20
    if scenario=='joint':
        model,xi,_=generate_vcm(n,m,5,'normal',rep);h=1.25*bandwidth(model.counts)
        fit=multiplier_band(model,grid,h,B=2000,components=list(range(5)),multipliers=xi)
        arrays=dict(grid=grid,truth=truth(grid,5),estimate=fit['beta'],se=fit['se'],lower=fit['lower'],upper=fit['upper'],visual_grid=visual,visual_truth=truth(visual,5),bootstrap_maxima=fit['maxima'])
        return arrays,dict(model='five-coefficient VCM',p=5,critical_value=fit['q'])
    if scenario!='highd':raise ValueError('Unknown illustration')
    sid,t,y,xi=generate_mean(n,m,'normal',rep);h=bandwidth(np.full(n,m))
    fit=CoordinateMeanFit(sid,t,y,grid,h,xi);rows,arrays=fit.calibrate(mean_truth(grid,np.arange(100)),(50,100),return_arrays=True)
    chosen=next(x for x in rows if x['p']==100)
    arrays.update(lower=arrays['lower_p100'],upper=arrays['upper_p100'],visual_grid=visual,visual_truth=mean_truth(visual,np.arange(100)),bootstrap_maxima=arrays['bootstrap_maxima_p100'])
    return arrays,dict(model='100 distinct means with correlated coordinate-specific functional errors',p=100,critical_value=chosen['q'])
