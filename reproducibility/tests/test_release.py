import json
from pathlib import Path
import numpy as np
from scb_core import LocalLinear,multiplier_band
from coordinate_mean_engine import CoordinateMeanFit
from optimized_clz_loso import loso_select
from optimized_clz_fit import clz_fit
from clz_source_audit import clz_fit as direct_clz_fit,loso_select as direct_loso
from reproduce import rebuild_tables,check_hashes
import study
ROOT=Path(__file__).resolve().parents[1]

def test_saved_records_rebuild_all_manuscript_tables(tmp_path):
    rows=rebuild_tables(tmp_path)
    assert len(rows)==90
    assert all(r['R']==1000 for r in rows)

def test_release_inputs_and_unchanged_engines():
    check_hashes()

def test_coordinate_specific_engine_matches_independent_scalar_fits():
    rng=np.random.default_rng(1182);counts=np.array([4,6,7,5]*15)
    sid=np.repeat(np.arange(len(counts)),counts);t=rng.uniform(size=len(sid))
    y=np.column_stack([np.sin(t)+rng.normal(size=len(sid)),np.cos(t)+rng.normal(size=len(sid)),rng.normal(size=len(sid))])
    xi=rng.normal(size=(79,len(counts)));grid=np.linspace(0,1,9);h=.4
    fit=CoordinateMeanFit(sid,t,y,grid,h,xi)
    rows,arrays=fit.calibrate(np.zeros((len(grid),3)),(1,3),return_arrays=True)
    for k in range(3):
        independent=multiplier_band(LocalLinear(sid,t,np.ones((len(t),1)),y[:,k]),grid,h,rng=None,B=79,multipliers=xi)
        np.testing.assert_allclose(arrays['estimate'][:,k],independent['beta'][:,0],rtol=1e-8,atol=1e-9)
        np.testing.assert_allclose(arrays['se'][:,k],independent['se'][:,0],rtol=1e-8,atol=1e-9)
        np.testing.assert_allclose(arrays['coordinate_bootstrap_maxima'][:,k],independent['maxima'],rtol=1e-7,atol=1e-8)

def test_accelerated_clz_matches_original_formulas_and_subject_deletion():
    model,_,_=study.generate_vcm(200,5,2,'normal',0)
    candidates=np.array([.065,.08])
    actual,loss=loso_select(model,candidates);expected,direct_loss=direct_loso(model,candidates)
    assert actual==expected
    np.testing.assert_allclose(loss,direct_loss,rtol=5e-7,atol=5e-8)
    a=clz_fit(model,study.GRID,actual);b=direct_clz_fit(model,study.GRID,actual)
    for x,y in zip(a,b):np.testing.assert_allclose(x,y,rtol=5e-7,atol=5e-8)

def test_reference_process_chunk_replays_original_seeds_for_all_bandwidths():
    n,m,k,values=study.calibration_chunk((200,5,0))
    with np.load(ROOT/'reference/clz-reference-chunk0.npz') as reference:
        np.testing.assert_allclose(values,reference['maxima'],rtol=2e-10,atol=2e-10)
    assert (n,m,k)==(200,5,0)
