"""Shared sequencing-depth and missingness calibration for CFTK power tools."""

from __future__ import annotations

import numpy as np
from scipy.stats import nbinom


def get_nbinom_params(mean_depth: float) -> tuple[float, float, float]:
    """Return empirical negative-binomial parameters and baseline missingness."""
    mean_depth = float(mean_depth)
    if mean_depth <= 0:
        raise ValueError("mean_depth must be positive.")

    k = 1.3409309326892476 * mean_depth + 3.526927718332715
    r = (mean_depth / k - 0.6291928395185754) / 0.02390549241316919
    p = r / (r + mean_depth)
    missing_percent = (
        mean_depth * r * -0.0007582750922856895
        + 4.56905302776041
    )
    return float(r), float(p), float(missing_percent / 100.0)


def missing_rate_at_depth(mean_depth: float) -> float:
    """Return the calibrated unusable-observation rate at a mean depth."""
    mean_depth = float(mean_depth)

    if mean_depth >= 15:
        _, _, missing_rate = get_nbinom_params(mean_depth)
    else:
        r, p, missing_rate = get_nbinom_params(15)
        missing_rate += float(
            nbinom.cdf(15 - mean_depth, r, p)
            - nbinom.cdf(0, r, p)
        )

    return float(np.clip(missing_rate, 0.0, 1.0))
