from __future__ import annotations

import numpy as np


def epanechnikov_kernel(u: np.ndarray) -> np.ndarray:
    """Epanechnikov kernel K(u) = 0.75 * (1 - u^2) * 1{|u| <= 1}."""
    u = np.asarray(u, dtype=float)
    weights = 0.75 * (1.0 - u**2)
    return np.where(np.abs(u) <= 1.0, weights, 0.0)


def gaussian_kernel(u: np.ndarray) -> np.ndarray:
    """Standard Gaussian kernel."""
    u = np.asarray(u, dtype=float)
    return np.exp(-0.5 * u**2) / np.sqrt(2.0 * np.pi)


def kernel_h(u_raw: np.ndarray, bandwidth: float, kernel=epanechnikov_kernel) -> np.ndarray:
    """Return K_h(u_raw) = K(u_raw / h) / h."""
    return kernel(u_raw / bandwidth) / bandwidth
