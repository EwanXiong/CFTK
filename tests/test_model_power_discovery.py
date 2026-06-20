from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analysis.model_power import prepare_template_ensemble
from analysis.model_power_discovery import run_power_sample_size_grid


def _reference_data(n_cpgs=120):
    rng = np.random.default_rng(1)
    index = [f"chr1_{index + 1}" for index in range(n_cpgs)]
    mean = pd.Series(rng.uniform(0.15, 0.85, n_cpgs), index=index)
    std = pd.DataFrame(
        {
            "10_mean": rng.uniform(0.04, 0.08, n_cpgs),
            "30_mean": rng.uniform(0.03, 0.06, n_cpgs),
        },
        index=index,
    )
    return std, mean


def test_cv_discovery_power_grid_returns_bounded_power():
    std, mean = _reference_data()
    templates = prepare_template_ensemble(
        std,
        mean,
        n_templates=2,
        template_kwargs={
            "depth": [10, 30],
            "n_features": 60,
            "n_signal_cpgs": 8,
            "meth_diff": 0.05,
            "effect_sd": 0.01,
        },
        random_state=4,
    )
    result = run_power_sample_size_grid(
        templates,
        sample_sizes=[30],
        simulations_per_template=2,
        power_kwargs={
            "models": ("logreg",),
            "cv_folds": 3,
            "top_k": 5,
            "target_auc": 0.70,
        },
        ci_method="none",
        n_jobs=1,
        random_state=5,
    )
    curve = result["power_curve"]
    assert len(curve) == 2
    assert curve["power"].between(0, 1).all()
    assert curve["n_templates"].eq(2).all()
