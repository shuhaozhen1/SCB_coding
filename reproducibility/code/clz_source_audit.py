"""CLZ (Bernoulli 2018), equations (2.1), (3.1)--(3.7), Section 4.

This source-faithful tuning implementation is separate from the archived,
common-bandwidth comparison in comparators.py. No subject sandwich, ridge,
residual degrees-of-freedom adjustment, or bias correction is added to CLZ.
"""
from math import comb
import numpy as np
from scb_core import WindowSums, LocalLinear
from comparators import kernel_draw

# Section 4 specifies (0.06,0.09), without specifying a numerical search mesh.
# A fixed mesh inside that range is specified before any coverage is inspected.
H_CANDIDATES = np.linspace(.0601, .0899, 7)
VERSION = 'clz-paper-section4-loso-v1'


class LeaveSubjectMoments:
    """Exact own-subject moments at each observed time, without binning.

    Shift times into disjoint subject intervals only to locate observations.
    Polynomial marks and kernel powers retain the original observation times.
    """
    def __init__(self, model, marks, degree):
        self.t = model.t
        self.shifted = model.t + 3 * model.sid
        ix = np.argsort(self.shifted, kind='stable')
        self.sorted = self.shifted[ix]
        self.shape = marks.shape[1:]
        marks = np.asarray(marks)[ix].reshape(len(ix), -1)
        z = self.t[ix] - .5
        self.prefix = [np.vstack([np.zeros((1, marks.shape[1])),
                                  np.cumsum(marks * z[:, None]**k, axis=0)])
                       for k in range(degree + 1)]

    def moment(self, h, degree):
        lo = np.searchsorted(self.sorted, self.shifted - h, side='left')
        hi = np.searchsorted(self.sorted, self.shifted + h, side='right')
        sums = [v[hi] - v[lo] for v in self.prefix[:degree + 3]]
        c = (self.t - .5)[:, None]
        def power(a):
            return sum(comb(a, k) * (-c)**(a-k) * sums[k]
                       for k in range(a+1)) / h**a
        out = .75 / h * (power(degree) - power(degree+2))
        return out.reshape((len(self.t),) + self.shape)


def loso_select(model, candidates=H_CANDIDATES):
    """Equation (3.7): delete every measurement of the test subject.

    All actual observation times, including near-boundary times, enter CV.
    Solving the subtracted normal equations is algebraically the same as
    refitting each training set. A singular candidate receives infinite loss.
    """
    p = model.p
    xx = model.x[:, :, None] * model.x[:, None, :]
    marks = np.concatenate([xx.reshape(len(xx), -1),
                            model.x * model.y[:, None]], axis=1)
    total = WindowSums(model.t, marks, 4)
    own = LeaveSubjectMoments(model, marks, 4)
    scores = []
    for h in candidates:
        moments = [total.moment(model.t, h, k) - own.moment(h, k)
                   for k in range(3)]
        s0, s1, s2 = [v[:, :p*p].reshape(-1, p, p) for v in moments]
        gram = np.concatenate([np.concatenate([s0, s1], axis=2),
                               np.concatenate([s1, s2], axis=2)], axis=1)
        rhs = np.concatenate([moments[0][:, p*p:],
                              moments[1][:, p*p:]], axis=1)
        try:
            theta = np.linalg.solve(gram, rhs[:, :, None])[:, :, 0]
            pred = np.sum(model.x * theta[:, :p], axis=1)
            loss = float(np.mean((model.y - pred)**2))
            if not np.isfinite(loss):
                loss = float('inf')
        except np.linalg.LinAlgError:
            loss = float('inf')
        scores.append(loss)
    if not np.isfinite(scores).any():
        raise np.linalg.LinAlgError('All CLZ LOSO bandwidth candidates failed')
    index = int(np.argmin(scores))
    return float(candidates[index]), np.asarray(scores)


def density_bandwidth(t):
    """Normal-reference KDE rule for the Epanechnikov kernel, Section 3.

    h = {R(K)/(mu2(K)^2 R(phi''))}^{1/5} sd(T) N^{-1/5}.
    For this kernel R(K)=3/5 and mu2(K)=1/5; the factor is 2.345.
    """
    factor = (.6 / (.2**2 * (3 / (8*np.sqrt(np.pi)))))**.2
    return float(factor * np.std(t, ddof=1) * len(t)**(-.2))


def clz_fit(model, grid, h, density_h=None):
    """Literal published raw-moment normalization; uniform-time experiments.

    Section 4 uses the CV bandwidth for beta, residuals, Omega and Lambda.
    The density uses a conventional independent-data rule as allowed in
    Section 3. Passing density_h=h permits a prespecified sensitivity check.
    """
    N = len(model.t)
    pooled = LocalLinear(model.sid, model.t, model.x, model.y,
                         weighting='observation')
    fit = pooled.fit(grid, h)
    observed = pooled.fit(model.t, h)
    residual = model.y - np.sum(model.x * observed.beta, axis=1)
    if density_h is None:
        density_h = density_bandwidth(model.t)
    f = WindowSums(model.t, np.ones(N), 2).moment(grid, density_h, 0) / N
    xx = model.x[:, :, None] * model.x[:, None, :]
    omega = WindowSums(model.t, xx, 2).moment(grid, h, 0) / N
    lam = WindowSums(model.t, xx * residual[:, None, None]**2, 2).moment(grid, h, 0) / N
    inv = np.linalg.inv(omega)
    var = np.diagonal(inv @ lam @ inv, axis1=1, axis2=2) * .6 / (N*h*f[:, None])
    if np.any(var <= 0) or not np.isfinite(var).all():
        raise FloatingPointError('Invalid CLZ variance')
    return fit.beta, np.sqrt(var), residual, float(density_h)


def calibrate_mesh(n, m, grid, B=20000, candidates=H_CANDIDATES):
    """Fresh uniforms and independent Gaussian marks on every Monte Carlo draw.

    Common random numbers across h reduce simulation noise in comparisons;
    for each h the marginal experiment is exactly the Section 3 experiment.
    """
    N = n*m
    rng = np.random.default_rng(np.random.SeedSequence([20261017, 952, n, m]))
    values = np.empty((B, len(candidates)))
    for b in range(B):
        spacings = np.cumsum(rng.exponential(size=N+1))
        t = spacings[:-1] / spacings[-1]
        eta = rng.normal(size=N)
        # Reuse the exact polynomial prefix sums for all bandwidths.
        z = t-.5
        pref = [np.r_[0., np.cumsum(eta*z**j)] for j in range(3)]
        h = np.asarray(candidates)[:, None]
        c = np.asarray(grid)[None, :] - .5
        lo = np.searchsorted(t, np.asarray(grid)[None, :]-h)
        hi = np.searchsorted(t, np.asarray(grid)[None, :]+h, side='right')
        s0, s1, s2 = [v[hi]-v[lo] for v in pref]
        process = .75*(s0-(s2-2*c*s1+c*c*s0)/h**2)/np.sqrt(.6*N*h)
        values[b] = np.max(np.abs(process), axis=1)
    return values


def paper_model(n, seed):
    """Section 4 (4.1), sine coefficient, four observations per subject.

    X has variance exp(t), correlation exp(-|s-t|/4); epsilon_i=zeta_i X_i.
    This is an external implementation check, not a replacement study DGP.
    """
    rng = np.random.default_rng(seed)
    m = 4
    t = rng.uniform(size=(n, m))
    covariance = np.exp((t[:, :, None]+t[:, None, :])/2
                        - np.abs(t[:, :, None]-t[:, None, :])/4)
    covariate = np.einsum('nij,nj->ni', np.linalg.cholesky(covariance),
                         rng.normal(size=(n, m)))
    error = rng.normal(size=(n, 1))*covariate
    beta0 = 5*(t-.6)**2
    beta1 = .5+.4*np.sin(2*np.pi*(t-.5))
    y = beta0+beta1*covariate+error
    return LocalLinear(np.repeat(np.arange(n), m), t.ravel(),
                       np.column_stack([np.ones(n*m), covariate.ravel()]), y.ravel())
