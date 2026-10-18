"""Redraw retained numerical arrays or regenerate exact illustrated datasets."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import study
ROOT=Path(__file__).resolve().parents[1]
(ROOT/'results/realdata').mkdir(parents=True,exist_ok=True)
DISPLAY_MEANS=np.array([0,49,99])
BLUE,FILL='#24618C','#C8DDEB'
def panel(ax,arrays,coordinate,title):
    t = arrays['grid']
    ax.fill_between(t,arrays['lower'][:,coordinate],arrays['upper'][:,coordinate],
                    color=FILL,linewidth=0,zorder=1)
    # Smooth analytic true curve; band and estimate still connect the51 fitted points.
    ax.plot(arrays['visual_grid'],arrays['visual_truth'][:,coordinate],
            color='black',linewidth=1.15,zorder=3)
    ax.plot(t,arrays['estimate'][:,coordinate],color=BLUE,linewidth=1.05,
            linestyle=(0,(4,2.3)),zorder=4)
    ax.set_title(title,loc='left',pad=5)
    ax.set_xlim(0,1)
    ax.set_xticks([0,.5,1],labels=['0','0.5','1'])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.set_xlabel(r'$t$',labelpad=2)
    ax.margins(y=.08)
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(length=2.5)
    ax.grid(axis='y',color='.90',linewidth=.45)
    ax.set_axisbelow(True)


def legend(fig):
    fig.legend(handles=[Line2D([],[],color='black',linewidth=1.15,label='True function'),
                        Line2D([],[],color=BLUE,linewidth=1.05,linestyle=(0,(4,2.3)),label='Local linear estimate'),
                        Patch(facecolor=FILL,edgecolor='none',label='Nominal 95% joint band')],
               loc='lower center',bbox_to_anchor=(.5,.005),ncol=3,frameon=False,
               handlelength=2.4,columnspacing=1.7)


def draw_simulation(stem,arrays,metadata):
    if metadata['p']==5:
        fig = plt.figure(figsize=(6.4,3.8))
        gs = fig.add_gridspec(2,6,left=.065,right=.97,top=.925,bottom=.19,wspace=1.5,hspace=.66)
        for j,(row,col) in enumerate([(0,0),(0,2),(0,4),(1,1),(1,3)]):
            panel(fig.add_subplot(gs[row,col:col+2]),arrays,j,rf'({chr(97+j)}) $\beta_{{{j+1}}}(t)$')
    else:
        fig,axes = plt.subplots(1,3,figsize=(6.4,2.45),sharey=True)
        fig.subplots_adjust(left=.065,right=.97,top=.86,bottom=.31,wspace=.20)
        for j,(ax,coordinate) in enumerate(zip(axes,DISPLAY_MEANS)):
            panel(ax,arrays,int(coordinate),rf'({chr(97+j)}) $\mu_{{{coordinate+1}}}(t)$')
    legend(fig)
    fig.savefig(stem.with_suffix('.pdf'),metadata={'Title':metadata['model'],'Author':'','CreationDate':None,'ModDate':None})
    fig.savefig(stem.with_suffix('.png'),dpi=320)
    plt.close(fig)


def pbc_data_illustration(path):
    import pandas as pd
    from analyze_pbc import prepare
    raw=pd.read_csv(ROOT/'data/pbcseq.csv');d,model,_,_=prepare(raw,3000,1)
    with plt.rc_context({'font.family':'serif','font.size':12,'axes.titlesize':12,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42}):
        fig,ax=plt.subplots(1,3,figsize=(10.5,3.3));rng=np.random.default_rng(710)
        for subject in rng.choice(d.id.unique(),60,replace=False):
            di=d.loc[d.id==subject].sort_values('day');ax[0].plot(di.day,np.log(di.bili),c='.65',lw=.5,alpha=.65)
        ax[0].set(xlabel='Days since enrollment',ylabel='Log bilirubin',title='(a) 60 sampled trajectories')
        ax[1].hist(model.counts,bins=np.arange(.5,model.counts.max()+1.5),color='.6',edgecolor='white');ax[1].set(xlabel='Retained visits',ylabel='Patients',title='(b) Counts per patient')
        ax[2].hist(d.day,bins=30,color='.6',edgecolor='white');ax[2].set(xlabel='Days since enrollment',ylabel='Visits',title='(c) Observation schedule')
        fig.tight_layout();fig.savefig(path,metadata={'CreationDate':None,'ModDate':None});plt.close(fig)

def figures(output,regenerate=False):
    output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'serif','font.size':9,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'pdf.fonttype':42})
    for stem,scenario in [('vcm-estimation-bands','joint'),('highd-mean-estimation-bands','highd')]:
        with np.load(ROOT/'reference/plotdata'/(stem+'.npz')) as f:arrays={k:f[k] for k in f.files}
        metadata=json.loads((ROOT/'reference/plotdata'/(stem+'.json')).read_text())
        if regenerate:
            new,meta=study.illustration(scenario,metadata['rep'])
            for key in ['grid','truth','estimate','se','lower','upper','bootstrap_maxima']:
                np.testing.assert_allclose(new[key],arrays[key],rtol=5e-7,atol=5e-8,err_msg=stem+': '+key)
            np.testing.assert_allclose(meta['critical_value'],metadata['critical_value'],rtol=5e-7,atol=5e-8)
            arrays={k:new[k] for k in arrays}
        draw_simulation(output/stem,arrays,metadata)
        np.savez_compressed(output/(stem+'.npz'),**arrays)
        (output/(stem+'.json')).write_text(json.dumps(metadata,indent=2)+'\n')
    # The current PBC draw function only needs the retained curves and labels.
    import analyze_pbc_comparisons as pbc
    pbc.OUT=output
    for stem in ['pbc_vcm_comparison','pbc_scalar_mean_comparison','pbc_joint_mean_comparison']:
        for ext in ['.npz','.json']:
            (output/(stem+ext)).write_bytes((ROOT/'reference/plotdata'/(stem+ext)).read_bytes())
    pbc.redraw();pbc_data_illustration(output/'pbc_data_illustration.pdf')
    print('Redrew six manuscript figures from retained arrays and the public PBC data.',flush=True)

def realdata(output):
    output.mkdir(parents=True,exist_ok=True)
    import analyze_pbc_comparisons as pbc
    pbc.OUT=output;pbc.CLZ_BANDWIDTH_RULE='matched';pbc.MEAN_BANDWIDTH_SCALE=1.25
    pbc.run()
    for stem in ['pbc_vcm_comparison','pbc_scalar_mean_comparison','pbc_joint_mean_comparison']:
        with np.load(output/(stem+'.npz')) as actual,np.load(ROOT/'reference/plotdata'/(stem+'.npz')) as expected:
            for key in ['grid','proposed_estimate','proposed_se','proposed_q','comparator_estimate','comparator_se','comparator_q']:
                np.testing.assert_allclose(actual[key],expected[key],rtol=5e-7,atol=5e-8,err_msg=stem+': '+key)
    pbc_data_illustration(output/'pbc_data_illustration.pdf')
    (output/'reproduction-verification.json').write_text(json.dumps(dict(passed=True,analyses=3,grid_points=201,B=10000,CLZ_bandwidth='matched',mean_bandwidth_scale=1.25,rtol=5e-7,atol=5e-8),indent=2)+'\n')
    print('All three real-data analyses reproduced the retained numerical curves and critical values.',flush=True)
