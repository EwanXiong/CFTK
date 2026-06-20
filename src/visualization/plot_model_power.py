"""Plots for simulation-based model-development power analyses."""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_power_by_sample_size(
    power_curve: pd.DataFrame,
    *,
    model: str = "logreg",
    power_metric: str = "power",
    show_ci: bool = False,
    target_power: float | None = 0.80,
    title: str | None = None,
    ax=None,
):
    """Plot CV discovery power against total study sample size by depth."""
    required = {"model", "sample_size", "mean_depth", power_metric}
    missing = required - set(power_curve.columns)
    if missing:
        raise KeyError(
            "power_curve is missing required columns: "
            + ", ".join(sorted(missing))
        )

    data = power_curve.loc[power_curve["model"] == model].copy()
    data = data.sort_values(["mean_depth", "sample_size"])
    if data.empty:
        raise ValueError(f"No rows were found for model={model!r}.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    ci_drawn = False
    for mean_depth, group in data.groupby("mean_depth", sort=True):
        x = group["sample_size"].to_numpy(dtype=float)
        y = group[power_metric].to_numpy(dtype=float)
        line = ax.plot(x, y, marker="o", label=f"{mean_depth:g}×")[0]

        if show_ci and {"power_ci_low", "power_ci_high"}.issubset(group.columns):
            lower = group["power_ci_low"].to_numpy(dtype=float)
            upper = group["power_ci_high"].to_numpy(dtype=float)
            valid = np.isfinite(lower) & np.isfinite(upper)
            if valid.any():
                ax.fill_between(
                    x[valid],
                    lower[valid],
                    upper[valid],
                    alpha=0.15,
                    color=line.get_color(),
                )
                ci_drawn = True

    if show_ci and not ci_drawn:
        warnings.warn(
            "show_ci=True, but no finite confidence intervals are available.",
            RuntimeWarning,
        )
    if target_power is not None:
        ax.axhline(
            float(target_power),
            linestyle="--",
            linewidth=1,
            label=f"Target = {target_power:.0%}",
        )

    ax.set_xlabel("Total study sample size")
    ax.set_ylabel("Model-development power")
    ax.set_ylim(0, 1.02)
    ax.set_title(title or f"CV biomarker-discovery power: {model}")
    ax.legend(title="Mean depth", frameon=False)
    return fig, ax
