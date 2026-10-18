#!/usr/bin/env python3
"""Reproduce the current manuscript from retained records or original seeds."""
from __future__ import annotations
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv,gzip,hashlib,itertools,json,os,platform,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(name,'1')
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'generated/.mplconfig'))
sys.path.insert(0,str(ROOT/'code'))
import numpy as np
from threadpoolctl import threadpool_limits
import study
from simulation_table_format import tables

def jsonable(value):
    if isinstance(value,dict):return {str(k):jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [jsonable(v) for v in value]
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    return value

def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(jsonable(value),indent=2,allow_nan=False)+'\n')

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def reference_records():
    with gzip.open(ROOT/'reference/replications.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:yield json.loads(line)

def key(row):return tuple(row[k] for k in ('scenario','n','m','law','rep'))

def summarize(records,expected_reps=None):
    groups=defaultdict(list);seen=set()
    for r in records:
        k=key(r)
        if k in seen:raise ValueError('Duplicate dataset: '+str(k))
        seen.add(k)
        if r['B']!=(1000 if r['scenario']=='highd' else 2000):raise ValueError('Unexpected multiplier count')
        for z in r['proposed']:
            if z['cp'] not in (0,1) or (z.get('failure') and z['cp']):raise ValueError('Invalid failure/coverage convention')
            if not z.get('failure'):
                if z['cp']!=int(z['observed_maximum']<=z['q']):raise ValueError('Inconsistent coverage statistic')
            groups[r['scenario'],r['n'],r['m'],r['law'],z['p']].append((r,z))
    rows=[]
    for (scenario,n,m,law,p),values in sorted(groups.items()):
        R=len(values)
        if expected_reps and {r['rep'] for r,_ in values}!=set(range(expected_reps)):
            raise ValueError('Incomplete replication set: '+str((scenario,n,m,law,p)))
        valid=[z for _,z in values if not z.get('failure')]
        if not valid:raise ValueError('No valid fits in a cell')
        cp=float(np.mean([z['cp'] for _,z in values]));aw=float(np.mean([z['aw'] for z in valid]))
        row=dict(target_p=100 if scenario=='highd' else p,p=p,n=n,m=m,law=law,R=R,B=1000 if scenario=='highd' else 2000,degree=1,grid_start=.15,grid_end=.85,grid_size=51,paper_rule='paper',paper_CP=cp,paper_AW=aw,paper_CP_MCSE=float(np.sqrt(cp*(1-cp)/R)),paper_failures=R-len(valid))
        if scenario!='highd':
            comp=[r['comparator'] for r,_ in values];okay=[z for z in comp if not z.get('failure')]
            if not okay:raise ValueError('No valid comparator fit')
            row.update(comparator_CP=float(np.mean([z['cp'] for z in comp])),comparator_AW=float(np.mean([z['aw'] for z in okay])),comparator_failures=R-len(okay))
        else:row.update(comparator_CP='',comparator_AW='',comparator_failures=0)
        rows.append(row)
    return rows

def write_summary(rows,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def rebuild_tables(out):
    rows=summarize(reference_records(),1000)
    if len(rows)!=90:raise ValueError('Expected all 90 cells')
    with (ROOT/'reference/paper-summary.csv').open(newline='') as f:
        expected={(int(r['target_p']),int(r['p']),int(r['n']),int(r['m']),r['law']):r for r in csv.DictReader(f)}
    for r in rows:
        old=expected[r['target_p'],r['p'],r['n'],r['m'],r['law']]
        names=['paper_CP','paper_AW']+(['comparator_CP','comparator_AW'] if r['target_p']!=100 else [])
        for name in names:
            expected_value=old[name] or old[old['paper_rule']+'_'+name.removeprefix('paper_')]
            np.testing.assert_allclose(r[name],float(expected_value),rtol=2e-12,atol=2e-14,err_msg=str((name,r)))
    out.mkdir(parents=True,exist_ok=True)
    for name,content in tables(rows).items():
        expected=(ROOT/'reference/tables'/name).read_text()
        if content!=expected:raise ValueError('Rendered manuscript table differs: '+name)
        (out/name).write_text(content)
    write_summary(rows,out/'rebuilt-summary.csv')
    write_json(out/'verification.json',dict(passed=True,independent_datasets=72000,cells=90,R=1000,tables_identical=4))
    print('All 72000 datasets -> 90 summary cells -> four byte-identical manuscript tables.',flush=True)
    return rows

def check_hashes():
    manifest=json.loads((ROOT/'MANIFEST.json').read_text())
    for name,digest in manifest['sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Changed or missing release input: '+name)
    frozen=json.loads((ROOT/'reference/engine-source-hashes.json').read_text())
    for name,digest in frozen.items():
        if sha(ROOT/'code'/name)!=digest:raise ValueError('Numerical engine changed: '+name)
    print('Release input hashes and all 15 unchanged numerical engines verified.',flush=True)

def compare(actual,expected):
    for name in ('scenario','n','m','law','rep','B'):
        if actual[name]!=expected[name]:raise ValueError('Metadata mismatch: '+name)
    def match(a,b):
        if bool(a.get('failure'))!=bool(b.get('failure')):raise ValueError('Failure status changed')
        if a['cp']!=b['cp']:raise ValueError('Coverage indicator changed')
        if not a.get('failure'):
            for name in ('aw','q','observed_maximum','h'):
                if name in a and name in b:np.testing.assert_allclose(a[name],b[name],rtol=5e-7,atol=5e-8,err_msg=name)
    for a,b in zip(actual['proposed'],expected['proposed'],strict=True):
        if a['p']!=b['p']:raise ValueError('Mean dimension changed')
        match(a,b)
    if expected['comparator'] is not None:match(actual['comparator'],expected['comparator'])

_THREAD_GUARD=None
def init_worker():
    global _THREAD_GUARD
    _THREAD_GUARD=threadpool_limits(limits=1)

def replay(workers,all_cells=False):
    selected={}
    for row in reference_records():
        if row['rep']==0 and (all_cells or (row['n']==200 and row['m']==5)):
            selected[key(row)]=row
    jobs=sorted(selected)
    with ProcessPoolExecutor(max_workers=workers,initializer=init_worker) as pool:
        for actual in pool.map(study.one,jobs,chunksize=1):
            compare(actual,selected[key(actual)]);print('Replayed',key(actual),flush=True)
    report=dict(passed=True,datasets_replayed=len(jobs),all_design_cells=all_cells,rtol=5e-7,atol=5e-8,coverage_indicators_exact=True)
    write_json(ROOT/'generated/replay-verification.json',report)
    return report

def simulate(args):
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    config=dict(version=1,driver_sha256=sha(Path(__file__)),critical_bank_sha256=sha(ROOT/'reference/clz-critical-values.json'),scenarios=args.scenarios,n=args.n,m=args.m,laws=args.laws,reps=args.reps,seed_namespaces=dict(coefficient_scalar=[20261018,771211,2],highd=[20261018,184700]),engines={p.name:sha(p) for p in sorted((ROOT/'code').glob('*.py'))})
    config_path=out/'run-config.json'
    if config_path.exists() and json.loads(config_path.read_text())!=config:raise ValueError('Resume config/source mismatch; choose a new output folder')
    write_json(config_path,config)
    write_json(out/'environment.json',dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,workers=args.workers))
    path=out/'replications.jsonl';done={}
    if path.exists():
        for line in path.open():
            r=json.loads(line)
            if key(r) in done:raise ValueError('Duplicate existing record')
            done[key(r)]=r
    expected=set(itertools.product(args.scenarios,args.n,args.m,args.laws,range(args.reps)))
    if not set(done)<=expected:raise ValueError('Unexpected dataset in resume file')
    jobs=sorted(expected-set(done));start=time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers,initializer=init_worker) as pool,path.open('a') as stream:
        for i,row in enumerate(pool.map(study.one,jobs,chunksize=1),1):
            stream.write(json.dumps(jsonable(row),allow_nan=False)+'\n');stream.flush();done[key(row)]=row
            if i%50==0 or i==len(jobs):print(i,'/',len(jobs),'new datasets;',round(time.monotonic()-start,1),'seconds',flush=True)
    rows=summarize(done.values(),args.reps);write_summary(rows,out/'summary.csv')
    if args.reps==1000 and len(rows)==90:
        for name,text in tables(rows).items():(out/name).write_text(text)
    print('Saved',len(done),'independent datasets and',len(rows),'cells to',out,flush=True)

def calibrate(args):
    if args.draws<1000 or args.draws%1000:raise ValueError('--draws must be a positive multiple of 1000')
    out=args.output;out.mkdir(parents=True,exist_ok=True);chunks=out/'chunks';chunks.mkdir(exist_ok=True)
    config=dict(generator_sha256=sha(ROOT/'code/study.py'),grid=study.GRID.tolist(),bandwidths=study.H.tolist(),chunk_size=1000,seed_namespace=[20261018,952407])
    if (out/'configuration.json').exists() and json.loads((out/'configuration.json').read_text())!=config:
        raise ValueError('Calibration source/configuration changed; choose a new output folder')
    write_json(out/'configuration.json',config)
    jobs=[(n,m,k) for n in (200,400) for m in (5,20,80) for k in range(args.draws//1000) if not (chunks/f'n{n}-m{m}-k{k:03d}.npz').exists()]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=init_worker) as pool:
        for n,m,k,values in pool.map(study.calibration_chunk,jobs,chunksize=1):
            np.savez_compressed(chunks/f'n{n}-m{m}-k{k:03d}.npz',maxima=values)
    rows=[]
    for n in (200,400):
        for m in (5,20,80):
            parts=[]
            for k in range(args.draws//1000):
                with np.load(chunks/f'n{n}-m{m}-k{k:03d}.npz') as a:parts.append(a['maxima'])
            values=np.concatenate(parts)
            for j,h in enumerate(study.H):
                for level in (.95,.99):rows.append(dict(n=n,m=m,h_index=j,h=float(h),level=level,q=float(np.quantile(values[:,j],level,method='inverted_cdf')),B=args.draws))
    write_json(out/'critical-values.json',rows)
    if args.draws==200000:
        old=study.critical_values()
        for r in rows:np.testing.assert_allclose(r['q'],old[r['n'],r['m'],r['h_index'],r['level']],rtol=2e-10,atol=2e-10)
        print('All 84 retained reference quantiles reproduced.',flush=True)
    else:print('Exploratory reference only; the manuscript uses 200000 draws.',flush=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='command',required=True)
    c=sub.add_parser('check',help='Hashes, all records, all four manuscript tables, optional seed replays');c.add_argument('--replay',action='store_true');c.add_argument('--all-cells',action='store_true');c.add_argument('--workers',type=int,default=2)
    t=sub.add_parser('tables');t.add_argument('--output',type=Path,default=ROOT/'generated/tables')
    r=sub.add_parser('replay');r.add_argument('--all-cells',action='store_true');r.add_argument('--workers',type=int,default=2)
    s=sub.add_parser('simulate');s.add_argument('--scenarios',nargs='+',choices=study.SCENARIOS,default=list(study.SCENARIOS));s.add_argument('--n',nargs='+',type=int,choices=[200,400],default=[200,400]);s.add_argument('--m',nargs='+',type=int,choices=[5,20,80],default=[5,20,80]);s.add_argument('--laws',nargs='+',choices=study.LAWS,default=list(study.LAWS));s.add_argument('--reps',type=int,default=1000);s.add_argument('--workers',type=int,default=4);s.add_argument('--output',type=Path,default=ROOT/'generated/simulations')
    f=sub.add_parser('figures');f.add_argument('--regenerate-simulations',action='store_true');f.add_argument('--output',type=Path,default=ROOT/'generated/figures')
    p=sub.add_parser('realdata');p.add_argument('--output',type=Path,default=ROOT/'generated/realdata')
    k=sub.add_parser('calibrate-clz');k.add_argument('--draws',type=int,default=200000);k.add_argument('--workers',type=int,default=4);k.add_argument('--output',type=Path,default=ROOT/'generated/clz-reference')
    args=ap.parse_args()
    if hasattr(args,'workers') and args.workers<1:ap.error('--workers must be positive')
    if hasattr(args,'reps') and args.reps<1:ap.error('--reps must be positive')
    with threadpool_limits(limits=1):
        if args.command=='check':
            check_hashes();rebuild_tables(ROOT/'generated/tables')
            if args.replay or args.all_cells:replay(args.workers,args.all_cells)
        elif args.command=='tables':rebuild_tables(args.output)
        elif args.command=='replay':replay(args.workers,args.all_cells)
        elif args.command=='simulate':simulate(args)
        elif args.command=='calibrate-clz':calibrate(args)
        else:
            import plotting
            if args.command=='figures':plotting.figures(args.output,args.regenerate_simulations)
            elif args.command=='realdata':plotting.realdata(args.output)
if __name__=='__main__':main()
