"""Null-calibrated operating characteristics for CV biomarker discovery.

The base discovery engine estimates target-attainment probability from
out-of-fold cross-validated AUC. This module adds a full-pipeline null
calibration so three distinct study-design operating characteristics can be
reported:

- detection_power: probability of rejecting the no-discrimination null;
- target_attainment_probability: probability of reaching target_auc;
- probability_of_success: probability of satisfying both criteria.

Null studies use the same CpGs, baseline distributions, depth/missingness,
correlation structure, preprocessing, feature selection, CV splitter, and
classifier as signal studies, but case and control methylation distributions
are identical.
"""

from __future__ import annotations

from dataclasses import replace
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

from analysis.model_power import ModelPowerTemplate
from analysis.model_power_discovery import (
    run_cv_discovery_power_analysis,
    run_power_sample_size_grid as _run_target_attainment_grid,
)


def make_null_model_power_template(
    template: ModelPowerTemplate,
) -> ModelPowerTemplate:
    """Return a no-signal copy of a model-power template."""
    return replace(
        template,
        raw_effect=np.zeros_like(template.raw_effect, dtype=np.float64),
        is_signal=np.zeros_like(template.is_signal, dtype=bool),
        alpha_case={
            depth: np.asarray(values).copy()
            for depth, values in template.alpha_control.items()
        },
        beta_case={
            depth: np.asarray(values).copy()
            for depth, values in template.beta_control.items()
        },
        standardized_effect={
            depth: np.zeros_like(values, dtype=np.float64)
            for depth, values in template.standardized_effect.items()
        },
        metadata={
            **dict(template.metadata),
            "n_signal_cpgs": 0,
            "requested_mean_abs_effect": 0.0,
            "requested_effect_sd": 0.0,
            "realized_mean_abs_effect": 0.0,
            "realized_effect_sd": 0.0,
            "effect_direction": "null",
            "null_calibration": True,
        },
    )


def _empirical_upper_tail_pvalues(
    observed: np.ndarray,
    null_values: np.ndarray,
) -> np.ndarray:
    """Return add-one empirical upper-tail p-values for observed AUCs."""
    observed = np.asarray(observed, dtype=np.float64)
    null_values = np.asarray(null_values, dtype=np.float64)
    null_values = null_values[np.isfinite(null_values)]
    if len(null_values) == 0:
        raise ValueError("Null AUC distribution is empty.")
    return (
        1.0
        + np.sum(null_values[None, :] >= observed[:, None], axis=1)
    ) / (len(null_values) + 1.0)


def _upper_quantile(values: np.ndarray, probability: float) -> float:
    """Return a conservative empirical upper quantile across NumPy versions."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    try:
        return float(np.quantile(values, probability, method="higher"))
    except TypeError:  # NumPy < 1.22
        return float(np.quantile(values, probability, interpolation="higher"))


def _binary_ci_for_column(
    group: pd.DataFrame,
    *,
    success_column: str,
    ci_method: str,
    confidence: float,
    n_bootstrap: int,
    random_state: int,
) -> tuple[float, float]:
    if ci_method == "none":
        return np.nan, np.nan

    values = group[success_column].to_numpy(dtype=bool)
    if ci_method == "pooled_wilson":
        low, high = proportion_confint(
            int(values.sum()),
            len(values),
            alpha=1.0 - confidence,
            method="wilson",
        )
        return float(low), float(high)

    values_by_template = {
        template_id: subgroup[success_column].to_numpy(dtype=float)
        for template_id, subgroup in group.groupby("template_id")
    }
    template_ids = np.asarray(list(values_by_template))
    if len(template_ids) < 2:
        return np.nan, np.nan

    rng = np.random.default_rng(random_state)
    bootstrap = np.empty(n_bootstrap, dtype=np.float64)
    for index in range(n_bootstrap):
        sampled_ids = rng.choice(template_ids, size=len(template_ids), replace=True)
        template_means = []
        for template_id in sampled_ids:
            template_values = values_by_template[template_id]
            resampled = rng.choice(
                template_values,
                size=len(template_values),
                replace=True,
            )
            template_means.append(float(resampled.mean()))
        bootstrap[index] = float(np.mean(template_means))

    tail = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(bootstrap, tail)),
        float(np.quantile(bootstrap, 1.0 - tail)),
    )


def _run_null_calibration(
    templates: Sequence[Mapping[str, Any]],
    sample_sizes: Sequence[int],
    *,
    null_simulations_per_template: int,
    power_kwargs: Mapping[str, Any],
    n_jobs: int,
    random_state: int,
) -> pd.DataFrame:
    """Run no-signal studies for every template and sample-size design."""
    null_rows: list[pd.DataFrame] = []
    kwargs = dict(power_kwargs)
    kwargs.pop("target_auc", None)

    for sample_index, sample_size in enumerate(sample_sizes):
        for record in templates:
            template_id = int(record["template_id"])
            null_seed = int(
                np.random.SeedSequence(
                    [random_state, 99173, sample_index, template_id]
                ).generate_state(1, dtype=np.uint32)[0]
            )
            result = run_cv_discovery_power_analysis(
                make_null_model_power_template(record["template"]),
                total_sample_size=int(sample_size),
                target_auc=1.0,
                n_simulations=int(null_simulations_per_template),
                n_jobs=n_jobs,
                random_state=null_seed,
                **kwargs,
            )
            frame = result["replicate_results"].copy()
            frame.insert(0, "template_id", template_id)
            frame.insert(
                1,
                "template_seed",
                int(record.get("template_seed", template_id)),
            )
            frame["null_calibration"] = True
            null_rows.append(frame)

    return pd.concat(null_rows, ignore_index=True)


def run_power_sample_size_grid(
    templates: Sequence[Mapping[str, Any]],
    sample_sizes: Sequence[int],
    *,
    simulations_per_template: int = 10,
    null_simulations_per_template: int = 20,
    power_kwargs: Mapping[str, Any] | None = None,
    alpha: float = 0.05,
    ci_method: str = "none",
    confidence: float = 0.95,
    n_bootstrap: int = 500,
    n_jobs: int = 1,
    random_state: int = 0,
) -> dict[str, Any]:
    """Estimate three CV discovery operating characteristics.

    Detection uses an add-one empirical p-value from template-matched,
    full-pipeline no-signal simulations. Target attainment requires CV AUC to
    reach ``target_auc``. Probability of success requires both conditions.
    Equal weighting across CpG templates is preserved.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")
    if null_simulations_per_template < int(np.ceil(1.0 / alpha)) - 1:
        minimum = int(np.ceil(1.0 / alpha)) - 1
        raise ValueError(
            "null_simulations_per_template must be at least "
            f"{minimum} for alpha={alpha:g} so empirical significance is attainable."
        )

    kwargs = dict(power_kwargs or {})
    target_auc = float(kwargs.get("target_auc", 0.75))
    start = time.perf_counter()

    signal_result = _run_target_attainment_grid(
        templates,
        sample_sizes,
        simulations_per_template=simulations_per_template,
        power_kwargs=kwargs,
        ci_method="none",
        confidence=confidence,
        n_bootstrap=n_bootstrap,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    signal = signal_result["replicate_results"].copy()

    null = _run_null_calibration(
        templates,
        sample_sizes,
        null_simulations_per_template=null_simulations_per_template,
        power_kwargs=kwargs,
        n_jobs=n_jobs,
        random_state=random_state,
    )

    signal["target_attainment_success"] = signal["cv_auc"] >= target_auc
    signal["target_success"] = signal["target_attainment_success"]
    signal["detection_pvalue"] = np.nan
    signal["null_auc_threshold"] = np.nan

    keys = ["template_id", "sample_size", "model", "mean_depth"]
    for key, signal_index in signal.groupby(keys, sort=False).groups.items():
        template_id, sample_size, model_name, mean_depth = key
        null_mask = (
            (null["template_id"] == template_id)
            & (null["sample_size"] == sample_size)
            & (null["model"] == model_name)
            & (null["mean_depth"] == mean_depth)
        )
        null_aucs = null.loc[null_mask, "cv_auc"].to_numpy(dtype=np.float64)
        observed = signal.loc[signal_index, "cv_auc"].to_numpy(dtype=np.float64)
        signal.loc[signal_index, "detection_pvalue"] = (
            _empirical_upper_tail_pvalues(observed, null_aucs)
        )
        signal.loc[signal_index, "null_auc_threshold"] = _upper_quantile(
            null_aucs,
            1.0 - alpha,
        )

    signal["detection_success"] = signal["detection_pvalue"] <= alpha
    signal["probability_of_success_success"] = (
        signal["detection_success"] & signal["target_attainment_success"]
    )
    signal["power_success"] = signal["probability_of_success_success"]

    template_summary = (
        signal.groupby(keys, as_index=False)
        .agg(
            n_simulations=("cv_auc", "size"),
            mean_cv_auc=("cv_auc", "mean"),
            detection_power=("detection_success", "mean"),
            target_attainment_probability=("target_attainment_success", "mean"),
            probability_of_success=("probability_of_success_success", "mean"),
            mean_detection_pvalue=("detection_pvalue", "mean"),
            null_auc_threshold=("null_auc_threshold", "first"),
            mean_sensitivity_at_specificity=("sensitivity_at_specificity", "mean"),
            mean_feature_recall=("mean_feature_recall", "mean"),
            mean_feature_precision=("mean_feature_precision", "mean"),
            mean_selection_jaccard=("selection_jaccard", "mean"),
        )
    )

    rows: list[dict[str, Any]] = []
    for (sample_size, model_name, mean_depth), group in signal.groupby(
        ["sample_size", "model", "mean_depth"],
        sort=True,
    ):
        template_group = template_summary.loc[
            (template_summary["sample_size"] == sample_size)
            & (template_summary["model"] == model_name)
            & (template_summary["mean_depth"] == mean_depth)
        ]
        row: dict[str, Any] = {
            "sample_size": int(sample_size),
            "model": model_name,
            "depth": mean_depth,
            "mean_depth": mean_depth,
            "n_templates": int(template_group["template_id"].nunique()),
            "simulations_per_template": int(simulations_per_template),
            "null_simulations_per_template": int(null_simulations_per_template),
            "total_simulations": int(len(group)),
            "total_null_simulations": int(
                len(template_group) * null_simulations_per_template
            ),
            "alpha": float(alpha),
            "target_auc": target_auc,
            "mean_cv_auc": float(template_group["mean_cv_auc"].mean()),
            "detection_power": float(template_group["detection_power"].mean()),
            "target_attainment_probability": float(
                template_group["target_attainment_probability"].mean()
            ),
            "probability_of_success": float(
                template_group["probability_of_success"].mean()
            ),
            "power": float(template_group["probability_of_success"].mean()),
            "mean_null_auc_threshold": float(
                template_group["null_auc_threshold"].mean()
            ),
            "min_null_auc_threshold": float(
                template_group["null_auc_threshold"].min()
            ),
            "max_null_auc_threshold": float(
                template_group["null_auc_threshold"].max()
            ),
            "mean_sensitivity_at_specificity": float(
                template_group["mean_sensitivity_at_specificity"].mean()
            ),
            "mean_feature_recall": float(
                template_group["mean_feature_recall"].mean()
            ),
            "mean_feature_precision": float(
                template_group["mean_feature_precision"].mean()
            ),
            "mean_selection_jaccard": float(
                template_group["mean_selection_jaccard"].mean()
            ),
            "ci_method": ci_method,
        }

        metric_columns = {
            "detection_power": "detection_success",
            "target_attainment_probability": "target_attainment_success",
            "probability_of_success": "probability_of_success_success",
        }
        for metric, success_column in metric_columns.items():
            ci_seed = int(
                np.random.SeedSequence(
                    [
                        random_state,
                        int(sample_size),
                        int(float(mean_depth) * 100),
                        sum(ord(character) for character in metric),
                    ]
                ).generate_state(1, dtype=np.uint32)[0]
            )
            low, high = _binary_ci_for_column(
                group,
                success_column=success_column,
                ci_method=ci_method,
                confidence=confidence,
                n_bootstrap=n_bootstrap,
                random_state=ci_seed,
            )
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high

        row["power_ci_low"] = row["probability_of_success_ci_low"]
        row["power_ci_high"] = row["probability_of_success_ci_high"]
        rows.append(row)

    power_curve = pd.DataFrame(rows).sort_values(
        ["model", "mean_depth", "sample_size"]
    ).reset_index(drop=True)

    return {
        "replicate_results": signal,
        "null_replicate_results": null,
        "template_summary": template_summary,
        "power_curve": power_curve,
        "run_metadata": {
            "analysis_mode": "cv_discovery_null_calibrated",
            "sample_sizes": [int(value) for value in sample_sizes],
            "n_templates": len(templates),
            "simulations_per_template": int(simulations_per_template),
            "null_simulations_per_template": int(null_simulations_per_template),
            "alpha": float(alpha),
            "target_auc": target_auc,
            "ci_method": ci_method,
            "confidence": confidence if ci_method != "none" else None,
            "n_bootstrap": n_bootstrap if ci_method == "hierarchical_bootstrap" else None,
            "elapsed_seconds": time.perf_counter() - start,
        },
    }
