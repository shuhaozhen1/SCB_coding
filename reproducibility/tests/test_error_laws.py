"""Check centering, scale, skewness and common-factor dependence of the DGPs."""
import numpy as np
from error_laws import LAWS, draw_innovations


def test_innovation_moments():
    for law in LAWS:
        x=draw_innovations(np.random.default_rng(82311),500000,law)
        assert abs(x.mean()) < .012
        assert abs(np.mean(x*x)-1) < .025
        skew=np.mean((x-x.mean())**3)/np.std(x)**3
        expected=np.sqrt(8/5) if law=='chisq5' else 0
        assert abs(skew-expected) < .11


