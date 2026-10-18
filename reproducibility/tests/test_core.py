import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import quad
from scb_core import LocalLinear, WindowSums, epan
from validation_helpers import kernel_correlation


def data():
    r=np.random.default_rng(213)
    m=np.arange(4,24)
    sid=np.repeat(np.arange(len(m)),m)
    t=r.uniform(size=len(sid))
    x=np.column_stack([np.ones(len(t)),r.normal(size=len(t))])
    y=1+t+x[:,1]*(.4+.2*t)+r.normal(size=len(t))
    return sid,t,x,y


def test_prefix_local_fit_and_contributions():
    sid,t,x,y=data(); h=.29; g=np.array([0,.06,.37,.81,1.])
    model=LocalLinear(sid,t,x,y)
    fit=model.fit(g,h); psi,res=model.contributions(g,h,fit)
    obs=[]
    for u in t:
        z=(t-u)/h; design=np.column_stack([x,x*z[:,None]])
        w=epan(z)/h/model.counts[sid]
        coeff=np.linalg.solve(design.T@(w[:,None]*design),design.T@(w*y))
        obs.append(coeff[:2])
    directres=y-np.sum(x*np.array(obs),axis=1)
    assert_allclose(res,directres,rtol=1e-8,atol=1e-8)
    for k,u in enumerate(g):
        z=(t-u)/h; design=np.column_stack([x,x*z[:,None]])
        w=epan(z)/h/model.counts[sid]
        gram=design.T@(w[:,None]*design)
        coeff=np.linalg.solve(gram,design.T@(w*y))
        assert_allclose(fit.beta[k],coeff[:2],rtol=1e-9,atol=1e-9)
        for i in range(model.n):
            mask=sid==i
            score=design[mask].T@(w[mask]*directres[mask])
            direct=model.n*np.linalg.solve(gram,score)[:2]
            assert_allclose(psi[i,k],direct,rtol=1e-8,atol=1e-8)


def test_kernel_covariance_integral():
    g=np.array([0,.1,.2,.5,1.]); h=.2
    got=kernel_correlation(g,h)
    for j,u in enumerate(g):
        for k,v in enumerate(g):
            d=(u-v)/h
            expected=quad(lambda z:epan(z)*epan(z+d),-1,1,points=[np.clip(-1-d,-1,1),np.clip(1-d,-1,1)])[0]/.6
            assert_allclose(got[j,k],expected,atol=1e-10)


def test_equal_counts_weights_agree():
    r=np.random.default_rng(1); sid=np.repeat(np.arange(50),5)
    t=r.uniform(size=250); x=np.ones((250,1)); y=r.normal(size=250)
    g=np.linspace(0,1,21)
    a=LocalLinear(sid,t,x,y); b=LocalLinear(sid,t,x,y,'observation')
    assert_allclose(a.fit(g,.2).beta,b.fit(g,.2).beta,atol=1e-10)


def test_spline_covariance_and_simulation_scale():
    from comparison_design import cai_hu, spline_basis
    sid,t,x,y=data(); model=LocalLinear(sid,t,np.ones((len(t),1)),y)
    grid=np.linspace(.15,.85,31)
    beta,se,q,J=cai_hu(model,grid,np.random.default_rng(10),199)
    A=spline_basis(t,J);inv=np.linalg.inv(A.T@A)
    theta=inv@A.T@y;res=y-A@theta
    C=np.zeros_like(inv)
    for i in range(model.n):
        ix=sid==i;score=A[ix].T@res[ix]
        C+=np.outer(score,score)
    C=inv@C@inv
    Ag=spline_basis(grid,J)
    assert_allclose(beta,Ag@theta,rtol=1e-9,atol=1e-9)
    assert_allclose(se**2,np.einsum('gi,ij,gj->g',Ag,C,Ag),rtol=1e-9,atol=1e-9)
    assert 1<q<5


def test_cao_finite_calibration_direct():
    from comparators import kernel_draw, cao_se
    rng=np.random.default_rng(77);t=np.sort(rng.uniform(size=503));eta=rng.normal(size=len(t));g=np.linspace(.15,.85,21);h=.13
    direct=epan((t[:,None]-g[None,:])/h).T@eta/np.sqrt(.6*len(t)*h)
    assert_allclose(kernel_draw(t,eta,g,h),direct,rtol=1e-10,atol=1e-10)
    sid,t,x,y=data();model=LocalLinear(sid,t,x,y,'observation');res=y-np.sum(x*model.fit(t,.29).beta,axis=1)
    got=cao_se(model,g,.29,res)
    for k,u in enumerate(g):
        w=epan((t-u)/.29)/.29;omega=x.T@(w[:,None]*x)/len(t);lam=x.T@((w*res**2)[:,None]*x)/len(t);inv=np.linalg.inv(omega)
        expected=np.sqrt(np.diag(inv@lam@inv)*.6/(len(t)*.29*(w.sum()/len(t))))
        assert_allclose(got[k],expected,rtol=1e-8)

def test_functional_error_without_added_scalar_noise():
    from study import generate_vcm
    from comparison_design import truth
    # All observations from one subject must lie on that subject's one
    # three-basis error curve. Independent observation noise violates this.
    for p in (1,2,5):
        for law in ('normal','laplace','chisq5'):
            model,_,_=generate_vcm(30,9,p,law,213)
            error=model.y-np.sum(model.x*truth(model.t,p),axis=1)
            basis=np.column_stack([np.ones(len(model.t)),np.sin(2*np.pi*model.t),np.cos(2*np.pi*model.t)])
            for i in range(model.n):
                ix=model.sid==i
                coeff=np.linalg.lstsq(basis[ix],error[ix],rcond=None)[0]
                assert_allclose(basis[ix]@coeff,error[ix],atol=1e-12,rtol=1e-12)


