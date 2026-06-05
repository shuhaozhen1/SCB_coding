"""scb — local-linear simultaneous confidence bands for sparse functional VCM."""

from .bandwidth import BandwidthCVResult, make_bandwidth_grid, select_bandwidth_cv
from .config import BandwidthCVConfig, BootstrapConfig, LocalLinearConfig
from .estimation import LocalLinearFit, local_linear_vcm_fit
from .inference import SCBResult, band_covers_truth, local_linear_vcm_scb
from .kernels import epanechnikov_kernel, gaussian_kernel

__all__ = [
    "BandwidthCVConfig",
    "BandwidthCVResult",
    "BootstrapConfig",
    "LocalLinearConfig",
    "make_bandwidth_grid",
    "select_bandwidth_cv",
    "LocalLinearFit",
    "local_linear_vcm_fit",
    "SCBResult",
    "band_covers_truth",
    "local_linear_vcm_scb",
    "epanechnikov_kernel",
    "gaussian_kernel",
]
