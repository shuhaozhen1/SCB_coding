"""Reproducible full-domain PBC comparisons; no publication-source edits.

Run from the repository root with single-threaded BLAS:
  revision/.venv/bin/python revision2/code/analyze_pbc_comparisons.py run
  revision/.venv/bin/python revision2/code/analyze_pbc_comparisons.py redraw

All proposed fits and observed-time residual fits are local linear. The main
CLZ comparison matches the proposed regression bandwidth; the original pooled
LOSO CV selector remains available with --clz-bandwidth-rule cv. Mean fits use
a fixed 1.25-fold bandwidth increase for smoother trajectories. Neither rule
is selected using relative width or statistical significance. The literal CLZ
variance and a separate density-normalization sensitivity are retained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir())/"pbc-comparison-mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from analyze_pbc import prepare, intervals, LABELS
from scb_core import LocalLinear, WindowSums, bandwidth
from optimized_vcm_engine import multiplier_band
from optimized_cai_hu import fit_cai_hu, spline_basis
from clz_source_audit import clz_fit, LeaveSubjectMoments

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"results/realdata"
GRID = np.linspace(0, 1, 201)
B = 10000
HORIZON = 3000
CLZ_BANDWIDTH_RULE = 'matched'
MEAN_BANDWIDTH_SCALE = 1.25
CLZ_LEVEL = 1-.05/6
CAIHU_JOINT_LEVEL = 1-.05/4
assert np.isclose(CLZ_LEVEL, .9916666666666667, atol=0, rtol=1e-15)
assert CAIHU_JOINT_LEVEL == .9875
MEAN_LABELS = ["Log bilirubin", "Albumin", "Prothrombin time", "Log alkaline phosphatase"]
CAUTIONS = [
    "Observed visits and complete cases are analyzed descriptively; informative visits/missingness are not corrected.",
    "CLZ is a scalar sparse-sampling procedure; the six-coefficient construction uses Bonferroni.",
    "CLZ full-domain curves are a numerical extension: the paper does not supply a dedicated local-linear equivalent-kernel calibration at the endpoints of observed support.",
    "The literal CLZ raw-moment variance has an additional density factor under nonuniform times. A formula sensitivity is saved separately, not used in the main plot.",
    "Pooled comparator and equal-subject proposed smoothers can have different observed-visit targets when visit counts are informative.",
    "No confidence-coverage or causal-effect claim is established from this single real dataset.",
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonable(value):
    if isinstance(value, dict):
        return {str(k):jsonable(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value):
    path.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False)+"\n")


def support(model, h):
    count=[];subjects=[];condition=[];minimum=[]
    s0,s1,s2=[model.xx.moment(GRID,h,k) for k in range(3)]
    gram=np.concatenate([np.concatenate([s0,s1],2),np.concatenate([s1,s2],2)],1)
    eigenvalues=np.linalg.eigvalsh(gram)
    for t in GRID:
        mask=np.abs(model.t-t)<=h
        count.append(int(mask.sum()));subjects.append(int(np.unique(model.sid[mask]).size))
    condition=eigenvalues[:,-1]/eigenvalues[:,0]
    return dict(local_observations=np.array(count),local_subjects=np.array(subjects),
                gram_min_eigenvalue=eigenvalues[:,0],gram_condition=condition)


def support_summary(values):
    return dict(min_local_observations=int(values['local_observations'].min()),
                min_local_subjects=int(values['local_subjects'].min()),
                endpoint_local_subjects=values['local_subjects'][[0,-1]].tolist(),
                min_gram_eigenvalue=float(values['gram_min_eigenvalue'].min()),
                max_gram_condition=float(values['gram_condition'].max()),
                endpoint_gram_condition=values['gram_condition'][[0,-1]].tolist())


def describe_curves(beta,se,q,labels):
    qs=np.broadcast_to(np.asarray(q), (len(labels),))
    lower=beta-se*qs[None,:];upper=beta+se*qs[None,:]
    return [dict(label=label,average_width=float((upper[:,k]-lower[:,k]).mean()),
                 lower_min=float(lower[:,k].min()),upper_max=float(upper[:,k].max()),
                 positive_t=intervals(GRID,lower[:,k]>0),
                 negative_t=intervals(GRID,upper[:,k]<0),
                 positive_days=intervals(GRID*HORIZON,lower[:,k]>0),
                 negative_days=intervals(GRID*HORIZON,upper[:,k]<0))
            for k,label in enumerate(labels)]


def source_metadata():
    names=("analyze_pbc_comparisons.py","analyze_pbc.py","scb_core.py",
           "optimized_vcm_engine.py","optimized_cai_hu.py","clz_source_audit.py")
    return dict(source_sha256={name:sha(ROOT/"code"/name) for name in names},
                raw_csv_sha256=sha(ROOT/"data/pbcseq.csv"),
                grid=GRID.tolist(),grid_size=len(GRID),observation_domain=[0,1],
                horizon_days=HORIZON,B=B,proposed_degree=1,
                clz_bandwidth_rule=CLZ_BANDWIDTH_RULE,
                mean_bandwidth_scale=MEAN_BANDWIDTH_SCALE,
                kernel="Epanechnikov",quantile_method="inverted_cdf",cautions=CAUTIONS)


def save(prefix, arrays, metadata, retained):
    np.savez_compressed(OUT/(prefix+".npz"),**arrays)
    retained.to_csv(OUT/(prefix+"-retained-visits.csv"),index=False)
    rows=[]
    for j,label in enumerate(metadata['labels']):
        for g,t in enumerate(GRID):
            row=dict(coordinate_one_based=j+1,label=label,t=float(t),day=float(t*HORIZON))
            for key in ('proposed_estimate','proposed_se','proposed_lower','proposed_upper',
                        'comparator_estimate','comparator_se','comparator_lower','comparator_upper'):
                row[key]=float(arrays[key][g,j])
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/(prefix+"-curves.csv"),index=False)
    metadata.update(source_metadata())
    metadata['saved_output_sha256']={suffix:sha(OUT/(prefix+suffix)) for suffix in
                                     ('.npz','-retained-visits.csv','-curves.csv')}
    write_json(OUT/(prefix+".json"),metadata)


def proposed(model, seed, h=None, xi=None):
    if h is None:h=bandwidth(model.counts)
    if xi is None:xi=np.random.default_rng(seed).normal(size=(B,model.n))
    result=multiplier_band(model,GRID,h,B=B,multipliers=xi,components=list(range(model.p)))
    assert result['q']==float(np.quantile(result['maxima'],.95,method='inverted_cdf'))
    return result,h,xi


def clz_reference(N,h):
    """Literal Section3 reference: fresh uniforms, independent observation marks."""
    seed=[20261018,180345,6,N]
    rng=np.random.default_rng(np.random.SeedSequence(seed))
    maxima=np.empty(B);first_t=None;first_eta=None;direct_error=0.
    center=GRID-.5
    for b in range(B):
        spacings=np.cumsum(rng.exponential(size=N+1));t=spacings[:-1]/spacings[-1]
        eta=rng.normal(size=N);z=t-.5
        pref=[np.r_[0.,np.cumsum(eta*z**k)] for k in range(3)]
        lo=np.searchsorted(t,GRID-h);hi=np.searchsorted(t,GRID+h,side='right')
        s0,s1,s2=[v[hi]-v[lo] for v in pref]
        process=.75*(s0-(s2-2*center*s1+center*center*s0)/h**2)/np.sqrt(.6*N*h)
        maxima[b]=np.abs(process).max()
        if b==0:
            first_t,first_eta=t.copy(),eta.copy()
            direct=(.75*np.maximum(1-((t[None,:]-GRID[:,None])/h)**2,0))@eta/np.sqrt(.6*N*h)
            direct_error=float(np.max(np.abs(process-direct)))
            np.testing.assert_allclose(process,direct,rtol=3e-9,atol=3e-9)
    q=float(np.quantile(maxima,CLZ_LEVEL,method='inverted_cdf'))
    assert q==float(np.sort(maxima)[int(np.ceil(CLZ_LEVEL*B))-1])
    return maxima,q,dict(seed=seed,B=B,level=CLZ_LEVEL,
                         method="Fresh iid U(0,1) points and iid observation N(0,1) marks, every draw; kernel sum/sqrt((3/5)Nh)",
                         first_draw_direct_max_error=direct_error,
                         full_input_storage=f"First reference draw stored; all further inputs exactly determined by saved seed and algorithm. All{B} maxima stored."),first_t,first_eta


def real_loso_select(model,candidates):
    """Whole-subject pooled CV with a fixed numerical rank failure rule.

    As in Eq3.7 every actual observation is predicted without its own subject.
    No ridge, pseudoinverse, outcome-dependent rank tolerance or expanded
    bandwidth search is used. The fixed100*eps relative eigenvalue threshold
    rejects a whole candidate if any deleted-subject local design is deficient.
    """
    p=model.p;N=len(model.t)
    xx=model.x[:,:,None]*model.x[:,None,:]
    marks=np.concatenate([xx.reshape(N,-1),model.x*model.y[:,None]],axis=1)
    total=WindowSums(model.t,marks,4);own=LeaveSubjectMoments(model,marks,4)
    tolerance=100*np.finfo(float).eps
    losses=[];diagnostics=[]
    for h in candidates:
        moments=[total.moment(model.t,h,k)-own.moment(h,k) for k in range(3)]
        s0,s1,s2=[v[:,:p*p].reshape(N,p,p) for v in moments]
        gram=np.concatenate([np.concatenate([s0,s1],2),np.concatenate([s1,s2],2)],1)
        gram=(gram+gram.transpose(0,2,1))/2
        eigenvalues=np.linalg.eigvalsh(gram)
        valid=(eigenvalues[:,-1]>0)&(eigenvalues[:,0]>tolerance*eigenvalues[:,-1])
        diagnostic=dict(h=float(h),rank_relative_tolerance=tolerance,
                        all_deleted_designs_valid=bool(valid.all()),
                        invalid_deleted_designs=int((~valid).sum()),
                        invalid_observation_indices=np.flatnonzero(~valid).tolist(),
                        min_eigenvalue=float(eigenvalues[:,0].min()),
                        min_relative_eigenvalue=float(np.min(eigenvalues[:,0]/eigenvalues[:,-1])))
        loss=float('inf')
        if valid.all():
            rhs=np.concatenate([moments[0][:,p*p:],moments[1][:,p*p:]],axis=1)
            try:
                theta=np.linalg.solve(gram,rhs[:,:,None])[:,:,0]
                value=float(np.mean((model.y-np.sum(model.x*theta[:,:p],axis=1))**2))
                if np.isfinite(value):loss=value
            except np.linalg.LinAlgError:
                diagnostic['solver_failure']=True
        diagnostics.append(diagnostic);losses.append(loss)
    losses=np.asarray(losses)
    if not np.isfinite(losses).any():
        raise np.linalg.LinAlgError('No rank-valid CLZ realdata CV candidate')
    return float(candidates[np.argmin(losses)]),losses,diagnostics


def vcm(raw):
    retained,model,center,scale=prepare(raw,HORIZON,6)
    h0=bandwidth(model.counts);h_candidates=np.geomspace(.5*h0,2*h0,21)
    print('PBC VCM: '+CLZ_BANDWIDTH_RULE+' bandwidth; CV retained as a reference',flush=True)
    ours,h,xi=proposed(model,530106)
    cv_chosen,loss,rank_diagnostics=real_loso_select(model,h_candidates)
    assert cv_chosen==float(h_candidates[np.argmin(loss)])
    chosen=h if CLZ_BANDWIDTH_RULE=='matched' else cv_chosen
    if CLZ_BANDWIDTH_RULE=='matched':assert chosen==h
    beta,se,residual,b=clz_fit(model,GRID,chosen)
    ref,q,ref_meta,first_t,first_eta=clz_reference(len(model.t),chosen)
    density=WindowSums(model.t,np.ones(len(model.t)),2).moment(GRID,b,0)/len(model.t)
    sensitivity_se=se*np.sqrt(density[:,None])
    pooled=LocalLinear(model.sid,model.t,model.x,model.y,weighting='observation')
    ps=support(model,h);cs=support(pooled,chosen)
    arrays=dict(grid=GRID,days=GRID*HORIZON,proposed_estimate=ours['beta'],proposed_se=ours['se'],
                proposed_lower=ours['lower'],proposed_upper=ours['upper'],proposed_q=np.array(ours['q']),
                proposed_maxima=ours['maxima'],proposed_subject_influence=ours['psi'],
                proposed_residual=ours['residual'],proposed_multipliers=xi,
                comparator_estimate=beta,comparator_se=se,comparator_lower=beta-q*se,
                comparator_upper=beta+q*se,comparator_q=np.array(q),
                comparator_reference_maxima=ref,comparator_residual=residual,
                comparator_h_candidates=h_candidates,comparator_cv_losses=loss,density=density,
                comparator_cv_rank_valid=np.array([z['all_deleted_designs_valid'] for z in rank_diagnostics]),
                comparator_cv_invalid_design_count=np.array([z['invalid_deleted_designs'] for z in rank_diagnostics]),
                density_sensitivity_se=sensitivity_se,density_sensitivity_lower=beta-q*sensitivity_se,
                density_sensitivity_upper=beta+q*sensitivity_se,
                reference_first_uniform_times=first_t,reference_first_gaussian_marks=first_eta,
                observation_subject_id=model.sid,observation_time=model.t,
                observation_x=model.x,observation_y=model.y,
                **{'proposed_'+k:v for k,v in ps.items()},**{'comparator_'+k:v for k,v in cs.items()})
    labels=LABELS
    meta=dict(analysis='six-coefficient VCM',labels=labels,n=int(model.n),N=len(retained),
              mH=float(1/np.mean(1/model.counts)),min_count=int(model.counts.min()),max_count=int(model.counts.max()),
              center=center,scale=scale,proposed_h=h,proposed_h_days=h*HORIZON,
              proposed_q=ours['q'],proposed_seed=530106,
              proposed_calibration='All6 coefficients and201 full-domain points jointly at.95',
              comparator='CLZ literal printed scalar procedure + Bonferroni',
              comparator_level=CLZ_LEVEL,comparator_q=q,comparator_h=chosen,comparator_c=chosen,
              comparator_bandwidth_rule=CLZ_BANDWIDTH_RULE,
              comparator_bandwidth_explanation='Same h as proposed; all coefficient/residual/design/score-moment fits share h' if CLZ_BANDWIDTH_RULE=='matched' else 'Whole-subject prediction CV; common estimation bandwidth',
              comparator_d=chosen,comparator_density_b=b,comparator_cv_candidates=h_candidates,
              comparator_cv_bandwidth=cv_chosen,
              comparator_cv_used_for_main=bool(CLZ_BANDWIDTH_RULE=='cv'),
              comparator_cv_losses=[float(v) if np.isfinite(v) else None for v in loss],
              comparator_cv_index=int(np.argmin(loss)),
              comparator_cv_rank_diagnostics=rank_diagnostics,
              comparator_cv_rank_rule='Symmetrize every leave-subject Gram; require lambda_min>100*machine_epsilon*lambda_max for every deleted local design; otherwise candidate loss is infinity/null. No regularization.',
              comparator_cv_at_search_edge=bool(np.argmin(loss) in (0,len(loss)-1)),
              comparator_cv_rule='Predeclared21 log-spaced points[.5h0,2h0], h0 depends only on retained subject counts; exact LOSO prediction squared error over every observed visit; first minimum',
              comparator_reference=ref_meta,proposed_support=support_summary(ps),comparator_support=support_summary(cs),
              density_range=[float(density.min()),float(density.max())],
              density_sensitivity='Formula sensitivity only: replace raw moments by conditional moments; same estimate,h,q, but se multiplied by sqrt(fhat). Not original printed CLZ; excluded from main comparison figure.',
              proposed_curves=describe_curves(ours['beta'],ours['se'],ours['q'],labels),
              comparator_curves=describe_curves(beta,se,q,labels),
              density_sensitivity_curves=describe_curves(beta,sensitivity_se,q,labels))
    save('pbc_vcm_comparison',arrays,meta,retained)
    return meta


def spline_comparisons(models,level,seed):
    assert level in (.95,CAIHU_JOINT_LEVEL)
    streams=np.random.SeedSequence(seed).spawn(len(models));fits=[];qs=[];maxima=[];draws=[]
    for model,stream in zip(models,streams):
        fit=fit_cai_hu(model,GRID);z=np.random.default_rng(stream).normal(size=(B,fit['root'].shape[0]))
        process=z@fit['root'].T@fit['grid_basis'].T/fit['se']
        mx=np.abs(process).max(axis=1);q=float(np.quantile(mx,level,method='inverted_cdf'))
        assert q==float(np.sort(mx)[int(np.ceil(level*B))-1])
        fits.append(fit);qs.append(q);maxima.append(mx);draws.append(z)
    return fits,np.array(qs),np.column_stack(maxima),draws


def means(raw,joint):
    if joint:
        # Identical rows to the six-coefficient analysis in this dataset, checked below.
        retained=raw.loc[(raw.day>=0)&(raw.day<=HORIZON)].dropna(subset=['id','day','bili','albumin','protime','alk.phos']).copy()
        retained=retained.loc[(retained.bili>0)&(retained['alk.phos']>0)].copy()
        y=np.column_stack([np.log(retained.bili),retained.albumin,retained.protime,np.log(retained['alk.phos'])])
        labels=MEAN_LABELS;level=CAIHU_JOINT_LEVEL;seed=530107;prefix='pbc_joint_mean_comparison'
    else:
        retained,scalar,_,_=prepare(raw,HORIZON,1);y=scalar.y[:,None]
        labels=['Log bilirubin'];level=.95;seed=530101;prefix='pbc_scalar_mean_comparison'
    sid=retained.id.to_numpy();t=retained.day.to_numpy()/HORIZON
    models=[LocalLinear(sid,t,np.ones((len(t),1)),y[:,k]) for k in range(y.shape[1])]
    model=models[0];h0=bandwidth(model.counts);h=MEAN_BANDWIDTH_SCALE*h0
    xi=np.random.default_rng(seed).normal(size=(B,model.n))
    print('PBC mean',len(models),'coordinates: common subject multipliers and CaiHu',flush=True)
    proposed_fits=[proposed(m,seed,h,xi)[0] for m in models]
    maxima=np.column_stack([z['maxima'] for z in proposed_fits]);joint_maxima=maxima.max(axis=1)
    q=float(np.quantile(joint_maxima,.95,method='inverted_cdf'))
    estimate=np.column_stack([z['beta'][:,0] for z in proposed_fits]);se=np.column_stack([z['se'][:,0] for z in proposed_fits])
    fits,cq,cmx,gaussian=spline_comparisons(models,level,[20261018,180345,len(models)])
    beta=np.column_stack([f['beta'] for f in fits]);cse=np.column_stack([f['se'] for f in fits])
    s=support(model,h)
    arrays=dict(grid=GRID,days=GRID*HORIZON,proposed_estimate=estimate,proposed_se=se,
                proposed_lower=estimate-q*se,proposed_upper=estimate+q*se,proposed_q=np.array(q),
                proposed_maxima=joint_maxima,proposed_coordinate_maxima=maxima,proposed_multipliers=xi,
                proposed_residual=np.column_stack([f['residual'] for f in proposed_fits]),
                proposed_subject_influence=np.stack([f['psi'][:,:,0] for f in proposed_fits],axis=2),
                comparator_estimate=beta,comparator_se=cse,comparator_lower=beta-cq[None,:]*cse,
                comparator_upper=beta+cq[None,:]*cse,comparator_q=cq,comparator_maxima=cmx,
                observation_subject_id=model.sid,observation_time=t,observation_y=y,
                **{'proposed_'+k:v for k,v in s.items()})
    for k,(f,z) in enumerate(zip(fits,gaussian)):
        for name in ('theta','covariance','root','grid_basis','residual','subject_scores','influence'):
            arrays[f'caihu_{k}_{name}']=f[name]
        arrays[f'caihu_{k}_gaussian_draws']=z
    vcm_rows=prepare(raw,HORIZON,6)[0]
    vcm_difference=len(set(retained.rownames)-set(vcm_rows.rownames))
    meta=dict(analysis='four-function joint means' if joint else 'scalar mean',labels=labels,
              n=int(model.n),N=len(retained),mH=float(1/np.mean(1/model.counts)),
              min_count=int(model.counts.min()),max_count=int(model.counts.max()),
              original_units=True,response_standardization=False,
              proposed_h=h,proposed_h_days=h*HORIZON,proposed_q=q,proposed_seed=seed,
              proposed_h0=h0,proposed_bandwidth_scale=MEAN_BANDWIDTH_SCALE,
              proposed_bandwidth_rule='Fixed scale times count-based h0 for smoother mean curves, independent of comparator widths',
              proposed_calibration=f'All{len(models)} mean functions and201 full-domain points jointly at.95; one common multiplier per subject',
              proposed_support=support_summary(s),comparator='Cai--Hu cubic B-spline',
              comparator_level=level,comparator_q=cq,
              comparator_calibration='Each marginal simultaneous spline band at1-.05/p; Bonferroni for joint mean when p4',
              comparator_seed=[20261018,180345,len(models)],
              comparator_knots=[f['J'] for f in fits],
              comparator_candidate_bic=[f['candidate_bic'] for f in fits],
              comparator_spline_gram_conditions=[float(np.linalg.cond(
                  (spline_basis(t,f['J']).T@spline_basis(t,f['J'])).toarray())) for f in fits],
              row_difference_from_vcm=vcm_difference,
              same_retained_rows_as_vcm=bool(set(retained.rownames)==set(vcm_rows.rownames)),
              proposed_curves=describe_curves(estimate,se,q,labels),
              comparator_curves=describe_curves(beta,cse,cq,labels))
    save(prefix,arrays,meta,retained)
    return meta


def sensitivity(raw):
    entries=[]
    for horizon in (3000,2000):
        retained,model,center,scale=prepare(raw,horizon,6);base=bandwidth(model.counts)
        xi=np.random.default_rng(527100+horizon+6).normal(size=(B,model.n))
        for factor in ((.8,1.,1.2) if horizon==3000 else (1.,)):
            fit,h,_=proposed(model,0,base*factor,xi);s=support(model,h)
            info=dict(horizon=horizon,factor=factor,n=int(model.n),N=len(retained),h=h,q=fit['q'],
                      mH=float(1/np.mean(1/model.counts)),center=center,scale=scale,
                      support=support_summary(s),curves=describe_curves(fit['beta'],fit['se'],fit['q'],LABELS))
            # describe_curves defaults to3000days; explicitly rescale this sensitivity.
            for curve in info['curves']:
                for label in ('positive','negative'):
                    curve[label+'_days']=[[a*horizon,b*horizon] for a,b in curve[label+'_t']]
            entries.append(info)
            np.savez_compressed(OUT/f'pbc_vcm_full_sensitivity_h{horizon}_scale{factor:.1f}.npz',
                                grid=GRID,days=GRID*horizon,beta=fit['beta'],se=fit['se'],
                                lower=fit['lower'],upper=fit['upper'],q=fit['q'],maxima=fit['maxima'],
                                **s)
    write_json(OUT/'pbc_vcm_full_sensitivity.json',entries)
    return entries


def draw(prefix,filename):
    meta=json.loads((OUT/(prefix+'.json')).read_text())
    with np.load(OUT/(prefix+'.npz'),allow_pickle=False) as saved:a={k:saved[k] for k in saved.files}
    p=len(meta['labels']);rows,cols=((3,2) if p==6 else (2,2) if p==4 else (1,1))
    fig,axes=plt.subplots(rows,cols,figsize=((7.1,7.1) if p==6 else (7.1,5.2) if p==4 else (7.1,3.3)),squeeze=False)
    for k,ax in enumerate(axes.flat):
        if k>=p:ax.axis('off');continue
        ax.fill_between(a['grid'],a['proposed_lower'][:,k],a['proposed_upper'][:,k],color='#CFDFEA',lw=0)
        ax.plot(a['grid'],a['proposed_estimate'][:,k],color='#163F5B',lw=1.1)
        for key in ('comparator_lower','comparator_upper'):
            ax.plot(a['grid'],a[key][:,k],color='#8E483C',ls='--',lw=.9)
        ax.plot(a['grid'],a['comparator_estimate'][:,k],color='#8E483C',ls='-.',lw=.85)
        if p==6:ax.axhline(0,color='.4',ls=':',lw=.65)
        ax.set_xlim(0,1);ax.set_xticks([0,.5,1]);ax.set_xlabel(r'$t$ (days /3000)')
        ax.set_title(f'({chr(97+k)}) '+meta['labels'][k].replace(' (standardized)','\n(standardized)'),loc='left')
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',color='.90',lw=.4)
        ax.tick_params(length=2.5);ax.margins(y=.06)
    comparator='CLZ' if p==6 else 'Cai–Hu'
    legend=[Line2D([],[],color='#163F5B',lw=1.1,label='Proposed estimate'),
            Line2D([],[],color='#8E483C',ls='-.',lw=.9,label=comparator+' estimate'),
            Patch(facecolor='#CFDFEA',label='Proposed joint 95% band'),
            Line2D([],[],color='#8E483C',ls='--',lw=.9,label=comparator+(' Bonferroni band' if p>1 else ' 95% band'))]
    fig.legend(handles=legend,loc='lower center',ncol=2,frameon=False,fontsize=8)
    fig.tight_layout(rect=[0,.10 if p==6 else .15 if p==4 else .22,1,1])
    for ext in ('.pdf','.png'):
        fig.savefig(OUT/(filename+ext),dpi=300,metadata={'Title':meta['analysis'],
                    'CreationDate':None,'ModDate':None} if ext=='.pdf' else None)
    plt.close(fig)


def run():
    started=time.monotonic();OUT.mkdir(exist_ok=True)
    raw=pd.read_csv(ROOT/'data/pbcseq.csv')
    required=['bili','albumin','protime','alk.phos']
    data_audit=dict(raw_N=len(raw),raw_subjects=int(raw.id.nunique()),
                    duplicate_subject_day_rows=int(raw.duplicated(['id','day']).sum()),
                    treatment_levels=sorted(raw.trt.dropna().unique().tolist()),
                    treatment_changes_within_subject=int((raw.groupby('id').trt.nunique()>1).sum()),
                    required_biomarker_missing={key:int(raw[key].isna().sum()) for key in required},
                    baseline_visits=int((raw.day==0).sum()),
                    nonuniform_visit_times=True,raw_csv_sha256=sha(ROOT/'data/pbcseq.csv'))
    summaries=[vcm(raw),means(raw,False),means(raw,True)]
    sensitivity_result=sensitivity(raw)
    data_audit['analyses']=[{k:z[k] for k in ('analysis','n','N','mH','min_count','max_count')} for z in summaries]
    data_audit['joint_mean_vs_vcm_rows_equal']=summaries[2]['same_retained_rows_as_vcm']
    write_json(OUT/'comparison-data-audit.json',data_audit)
    write_json(OUT/'comparison-summary.json',dict(analyses=summaries,vcmsensitivity=sensitivity_result,
              seconds=time.monotonic()-started,source=source_metadata(),all_proposed_fits_local_linear=True))
    redraw()
    print(json.dumps(jsonable(dict(seconds=time.monotonic()-started,analyses=[dict(analysis=z['analysis'],n=z['n'],N=z['N'],proposed_q=z['proposed_q'],comparator_q=z['comparator_q']) for z in summaries])),indent=2),flush=True)


def redraw():
    plt.rcParams.update({'font.family':'serif','font.size':9,'axes.titlesize':9,
                         'axes.labelsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'pdf.fonttype':42})
    for prefix,filename in (('pbc_vcm_comparison','pbc_vcm_bands'),
                            ('pbc_scalar_mean_comparison','pbc_mean_bands'),
                            ('pbc_joint_mean_comparison','pbc_joint_mean_bands')):
        draw(prefix,filename)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('run','redraw'))
    parser.add_argument('--clz-bandwidth-rule',choices=('matched','cv'),default='matched')
    parser.add_argument('--mean-bandwidth-scale',type=float,default=1.25)
    args=parser.parse_args()
    if not np.isfinite(args.mean_bandwidth_scale) or args.mean_bandwidth_scale<=0:
        parser.error('--mean-bandwidth-scale must be positive and finite')
    CLZ_BANDWIDTH_RULE=args.clz_bandwidth_rule
    MEAN_BANDWIDTH_SCALE=args.mean_bandwidth_scale
    with threadpool_limits(limits=1):
        run() if args.mode=='run' else redraw()
