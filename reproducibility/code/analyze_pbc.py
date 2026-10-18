"""Exploratory observed-visit analysis; no population or causal coverage claim."""
from pathlib import Path
import json,hashlib,os
os.environ.setdefault("MPLCONFIGDIR",str(Path(__file__).resolve().parents[1]/".mplconfig"))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits
from scb_core import LocalLinear,bandwidth,multiplier_band
plt.rcParams.update({'font.family':'serif','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/realdata';OUT.mkdir(exist_ok=True)
LABELS=['Intercept','D-penicillamine','Edema (standardized)','Albumin (standardized)','Prothrombin time (standardized)','Log ALP (standardized)']

def intervals(grid,flag):
    ix=np.flatnonzero(flag)
    if not len(ix):return []
    groups=np.split(ix,np.flatnonzero(np.diff(ix)>1)+1)
    return [[float(grid[g[0]]),float(grid[g[-1]])] for g in groups]

def prepare(raw,horizon,p):
    keys=['id','day','bili']+([] if p==1 else ['trt','edema','albumin','protime','alk.phos'])
    d=raw.loc[(raw.day>=0)&(raw.day<=horizon)].dropna(subset=keys).copy()
    d=d.loc[d.bili>0]
    center={};scale={}
    if p==1:x=np.ones((len(d),1))
    else:
        d=d.loc[d['alk.phos']>0].copy();d['logalp']=np.log(d['alk.phos'])
        cov=d[['edema','albumin','protime','logalp']]
        center=cov.mean().to_dict();scale=cov.std(ddof=0).to_dict()
        x=np.column_stack([np.ones(len(d)),(d.trt==1).astype(float),(cov-cov.mean())/cov.std(ddof=0)])
    model=LocalLinear(d.id.to_numpy(),d.day.to_numpy()/horizon,x,np.log(d.bili.to_numpy()))
    return d,model,center,scale

def run():
    raw=pd.read_csv(ROOT/'data/pbcseq.csv');summary=[]
    for horizon in (3000,2000):
      for p in (1,6):
        d,model,center,scale=prepare(raw,horizon,p)
        grid=np.linspace(.15,.85,201);base=bandwidth(model.counts)
        for factor in ((.8,1.,1.2) if horizon==3000 else (1.,)):
            h=factor*base;rng=np.random.default_rng(527100+horizon+p)
            result=multiplier_band(model,grid,h,rng,B=9999)
            days=grid*horizon
            info=dict(horizon=horizon,p=p,factor=factor,n=int(model.n),N=len(d),h=h,h_days=h*horizon,q=result['q'],mH=float(1/np.mean(1/model.counts)),mean_count=float(model.counts.mean()),median_count=float(np.median(model.counts)),min_count=int(model.counts.min()),max_count=int(model.counts.max()),center=center,scale=scale,min_eigenvalue=result['min_eigenvalue'])
            support=np.array([len(np.unique(model.sid[np.abs(model.t-t)<=h])) for t in grid]);info['min_local_subjects']=int(support.min())
            info['coefficients']=[]
            for k in range(p):
                info['coefficients'].append(dict(name=LABELS[k] if p==6 else 'Mean',aw=float(np.mean(2*result['q']*result['se'][:,k])),positive=intervals(days,result['lower'][:,k]>0),negative=intervals(days,result['upper'][:,k]<0)))
            summary.append(info)
            tag=f'h{horizon}_p{p}_scale{factor:.1f}'
            np.savez_compressed(OUT/(tag+'.npz'),grid=grid,days=days,beta=result['beta'],se=result['se'],lower=result['lower'],upper=result['upper'],q=result['q'],support=support,maxima=result['maxima'])
            if factor==1 and horizon==3000:
                fig,axs=plt.subplots(3,2,figsize=(7.1,7.4),squeeze=False) if p==6 else plt.subplots(1,1,figsize=(7.1,3.5),squeeze=False)
                for k,ax in enumerate(axs.flat):
                    ax.fill_between(days,result['lower'][:,k],result['upper'][:,k],color='.82',label='Nominal 95% band')
                    ax.plot(days,result['beta'][:,k],color='black',lw=1.4,label='Estimate')
                    if p==6:ax.axhline(0,color='.4',ls='--',lw=.8)
                    ax.set_title(LABELS[k].replace(' (standardized)', '\n(standardized)') if p==6 else 'Subject-balanced log-bilirubin trend')
                    ax.set_xlabel('Days since enrollment');ax.set_ylabel('Log-bilirubin scale')
                fig.tight_layout();fig.savefig(OUT/('pbc_vcm_bands.pdf' if p==6 else 'pbc_mean_bands.pdf'));plt.close(fig)
        # Save the exact retained rows, avoiding ambiguity about missing values.
        d.to_csv(OUT/f'analysis_h{horizon}_p{p}.csv',index=False)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    # Sampling pattern and attrition are shown rather than hidden by smoothing.
    d,model,_,_=prepare(raw,3000,1)
    plt.rcParams.update({'font.size':12,'axes.titlesize':12})
    fig,ax=plt.subplots(1,3,figsize=(10.5,3.3))
    rng=np.random.default_rng(710);ids=rng.choice(d.id.unique(),60,replace=False)
    for i in ids:
        di=d.loc[d.id==i].sort_values('day');ax[0].plot(di.day,np.log(di.bili),c='.65',lw=.5,alpha=.65)
    ax[0].set(xlabel='Days since enrollment',ylabel='Log bilirubin',title='(a) 60 sampled trajectories')
    ax[1].hist(model.counts,bins=np.arange(.5,model.counts.max()+1.5),color='.6',edgecolor='white');ax[1].set(xlabel='Retained visits',ylabel='Patients',title='(b) Counts per patient')
    ax[2].hist(d.day,bins=30,color='.6',edgecolor='white');ax[2].set(xlabel='Days since enrollment',ylabel='Visits',title='(c) Observation schedule')
    fig.tight_layout();fig.savefig(OUT/'pbc_data_illustration.pdf');plt.close(fig)
    source=ROOT/'data/pbcseq.csv'
    (ROOT/'data/PROVENANCE.txt').write_text('Dataset: survival::pbcseq (Mayo PBC sequential biomarkers).\nRetrieved 2026-10-15 from https://github.com/shuhaozhen1/SCB_coding/tree/main/data/pbcseq.csv\nPublic snapshot preserved locally; no claim that this matches the original manuscript commit.\nDocumentation: https://stat.ethz.ch/R-manual/R-devel/library/survival/html/pbcseq.html\nRaw CSV SHA-256: '+hashlib.sha256(source.read_bytes()).hexdigest()+'\nFull CSV: '+str(len(raw))+' rows; '+str(raw.id.nunique())+' patients.\nR package documentation warns that extra visits and missing laboratory measurements may be informative.\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':
    with threadpool_limits(limits=1):run()
