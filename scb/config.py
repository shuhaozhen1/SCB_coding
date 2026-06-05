from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LocalLinearConfig:
    """Configuration for pooled local-linear VCM estimation and inference."""

    bandwidth: float = 0.15
    ridge: float = 1e-8
    weighting: str = "subject"  # "subject" or "observation"
    scaled_design: bool = True

    def __post_init__(self) -> None:
        if self.bandwidth <= 0:
            raise ValueError("bandwidth must be positive.")
        if self.ridge < 0:
            raise ValueError("ridge must be nonnegative.")
        if self.weighting not in {"subject", "observation"}:
            raise ValueError("weighting must be either 'subject' or 'observation'.")


@dataclass(frozen=True)
class BootstrapConfig:
    """Configuration for multiplier-bootstrap simultaneous confidence bands.

    residual_method controls how residuals are estimated for the multiplier
    process. The default "observation" first estimates beta at the original
    observation times T_ij and uses

        r_ij = Y_ij - X_ij^T beta_hat(T_ij),

    then reuses these centered residuals for all target grid points. This is
    the preferred implementation for the SCB bootstrap. The alternative
    "local" uses the target-specific local-linear residual

        r_ij(t) = Y_ij - U_ij(t)^T theta_hat(t),

    mainly for comparison/backward compatibility.

    "loo" uses leave-one-subject-out residuals: beta is re-estimated with
    subject i removed and evaluated at subject i's observation times,

        r_ij = Y_ij - X_ij^T beta_hat^{(-i)}(T_ij).

    Because the in-sample local-linear smoother absorbs subject i's own noise,
    the plug-in "observation" residual has E[r^2] < sigma^2 in finite samples
    (worse for small bandwidth / large q), which biases the influence-function
    variance downward. The leave-one-subject-out residual removes that
    finite-sample bias. It is asymptotically equivalent to "observation"
    (subject leverage -> 0), so the influence/sandwich/bootstrap theory is
    unchanged; only the nuisance residual estimate becomes consistent. The cost
    is one extra local-linear fit per subject.
    """

    n_boot: int = 1000
    alpha: float = 0.05
    seed: int = 123
    center_influence: bool = True
    residual_method: str = "observation"  # "observation" or "local"
    multiplier_dist: str = "gaussian"  # "gaussian", "laplace", "rademacher", "mammen", "t3"

    def __post_init__(self) -> None:
        if self.n_boot <= 0:
            raise ValueError("n_boot must be positive.")
        if not (0 < self.alpha < 1):
            raise ValueError("alpha must lie in (0, 1).")
        if self.residual_method not in {"observation", "local", "loo"}:
            raise ValueError("residual_method must be one of 'observation', 'local', 'loo'.")
        if self.multiplier_dist not in {"gaussian", "laplace", "rademacher", "mammen", "t3"}:
            raise ValueError("multiplier_dist must be one of 'gaussian','laplace','rademacher','mammen','t3'.")


@dataclass(frozen=True)
class BandwidthCVConfig:
    """Configuration for bandwidth selection by subject-level cross-validation."""

    method: str = "kfold"  # "loo" or "kfold"
    n_folds: int = 5
    seed: int = 123
    loss_weighting: str = "subject"  # "subject" or "observation"

    def __post_init__(self) -> None:
        if self.method not in {"loo", "kfold"}:
            raise ValueError("method must be either 'loo' or 'kfold'.")
        if self.n_folds <= 1:
            raise ValueError("n_folds must be greater than 1.")
        if self.loss_weighting not in {"subject", "observation"}:
            raise ValueError("loss_weighting must be either 'subject' or 'observation'.")
