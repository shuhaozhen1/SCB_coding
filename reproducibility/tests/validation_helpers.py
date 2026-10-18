import numpy as np

def kernel_correlation(grid,h):
    """Exact integral of two Epanechnikov kernels, divided by integral K^2."""
    delta = np.abs(np.subtract.outer(grid,grid))/h
    a = -np.ones_like(delta)
    b = 1-delta
    # (1-u^2)(1-(u+d)^2) coefficients in u.
    coefficients = [1-delta**2,-2*delta,delta**2-2,2*delta,np.ones_like(delta)]
    integral = sum(c*(b**(k+1)-a**(k+1))/(k+1) for k,c in enumerate(coefficients))*.75**2/.6
    integral[delta>=2] = 0
    return np.clip(integral,-1,1)

