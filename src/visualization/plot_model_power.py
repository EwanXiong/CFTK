"""Plots for simulation-based model-development power analyses."""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _model_power_plot_data(
    power_curve: pd.DataFrame,
    *,
    model: str,
    power_metric: str,
) -> pd.DataFrame:
    """Validate and subset a model-power summary table for plotting."""
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
    return data


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
    """Plot CV discovery power against total study sample size with Matplotlib."""
    data = _model_power_plot_data(
        power_curve,
        model=model,
        power_metric=power_metric,
    )

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


def plot_power_by_sample_size_plotly(
    power_curve: pd.DataFrame,
    *,
    model: str = "logreg",
    power_metric: str = "power",
    show_ci: bool = False,
    target_power: float | None = 0.80,
    title: str | None = None,
    height: int = 560,
):
    """Create an interactive Plotly power curve with responsive width.

    The chart supports hover inspection, zooming, panning, fullscreen display,
    legend toggling, and image export through Plotly's modebar.
    """
    try:
        import plotly.graph_objects as go
        from plotly.colors import qualitative
    except ImportError as exc:  # pragma: no cover - optional web dependency
        raise ImportError(
            "Plotly is required for the interactive model-power chart. "
            "Install CFTK with the web extra or install plotly directly."
        ) from exc

    data = _model_power_plot_data(
        power_curve,
        model=model,
        power_metric=power_metric,
    )
    figure = go.Figure()
    palette = qualitative.Plotly
    ci_drawn = False

    hover_columns = [
        ("mean_cv_auc", "Mean CV AUC"),
        ("mean_sensitivity_at_specificity", "Sensitivity at specificity"),
        ("mean_feature_recall", "Feature recall"),
        ("mean_feature_precision", "Feature precision"),
        ("mean_selection_jaccard", "Selection Jaccard"),
        ("total_simulations", "Simulations"),
    ]

    for depth_index, (mean_depth, group) in enumerate(
        data.groupby("mean_depth", sort=True)
    ):
        group = group.sort_values("sample_size")
        x = group["sample_size"].to_numpy(dtype=float)
        y = group[power_metric].to_numpy(dtype=float)
        color = palette[depth_index % len(palette)]
        legend_group = f"depth-{mean_depth:g}"

        if show_ci and {"power_ci_low", "power_ci_high"}.issubset(group.columns):
            lower = group["power_ci_low"].to_numpy(dtype=float)
            upper = group["power_ci_high"].to_numpy(dtype=float)
            valid = np.isfinite(lower) & np.isfinite(upper)
            if valid.any():
                x_valid = x[valid]
                lower_valid = lower[valid]
                upper_valid = upper[valid]
                figure.add_trace(
                    go.Scatter(
                        x=x_valid,
                        y=upper_valid,
                        mode="lines",
                        line={"width": 0, "color": color},
                        hoverinfo="skip",
                        showlegend=False,
                        legendgroup=legend_group,
                    )
                )
                figure.add_trace(
                    go.Scatter(
                        x=x_valid,
                        y=lower_valid,
                        mode="lines",
                        line={"width": 0, "color": color},
                        fill="tonexty",
                        fillcolor=color.replace("rgb", "rgba").replace(")", ",0.16)"),
                        hoverinfo="skip",
                        showlegend=False,
                        legendgroup=legend_group,
                    )
                )
                ci_drawn = True

        available_hover = [
            (column, label)
            for column, label in hover_columns
            if column in group.columns
        ]
        if available_hover:
            customdata = np.column_stack(
                [group[column].to_numpy() for column, _ in available_hover]
            )
        else:
            customdata = None

        hovertemplate = (
            "Total N: %{x:.0f}<br>"
            "Power: %{y:.3f}<br>"
            f"Mean depth: {mean_depth:g}×"
        )
        for index, (column, label) in enumerate(available_hover):
            if column == "total_simulations":
                hovertemplate += f"<br>{label}: %{{customdata[{index}]:.0f}}"
            else:
                hovertemplate += f"<br>{label}: %{{customdata[{index}]:.3f}}"
        hovertemplate += "<extra></extra>"

        figure.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                name=f"{mean_depth:g}×",
                legendgroup=legend_group,
                line={"color": color},
                marker={"size": 9},
                customdata=customdata,
                hovertemplate=hovertemplate,
            )
        )

    if show_ci and not ci_drawn:
        warnings.warn(
            "show_ci=True, but no finite confidence intervals are available.",
            RuntimeWarning,
        )

    if target_power is not None:
        figure.add_hline(
            y=float(target_power),
            line_dash="dash",
            annotation_text=f"Target {target_power:.0%}",
            annotation_position="top left",
        )

    figure.update_layout(
        title=title or f"CV biomarker-discovery power: {model}",
        xaxis_title="Total study sample size",
        yaxis_title="Probability of reaching target CV AUC",
        yaxis={"range": [0, 1.02], "tickformat": ".0%"},
        hovermode="x unified",
        legend={"title": {"text": "Mean depth"}},
        height=int(height),
        margin={"l": 60, "r": 25, "t": 70, "b": 60},
    )
    return figure
