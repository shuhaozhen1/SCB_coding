"""Zero-mean, unit-variance innovation laws used in all current simulations."""
import numpy as np

LAWS = ('normal', 'laplace', 'chisq5')
LABELS = {'normal': 'Gaussian', 'laplace': 'Laplace', 'chisq5': 'Adjusted chi-square (5 df)'}


def draw_innovations(rng, size, law):
    if law == 'normal':
        return rng.normal(size=size)
    if law == 'laplace':
        return rng.laplace(loc=0.0, scale=1/np.sqrt(2), size=size)
    if law == 'chisq5':
        return (rng.chisquare(df=5, size=size)-5)/np.sqrt(10)
    raise ValueError('Unknown innovation law: '+str(law))
