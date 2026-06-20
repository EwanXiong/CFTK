"""Cross-validated model-development power for biomarker discovery.

This module estimates the probability that a prespecified classification
pipeline reaches a target out-of-fold cross-validated AUC for a total study
sample size. It does not estimate external generalizability.
"""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.proportion import proportion_confint

try:
    from analysis.model_power import ModelPowerTemplate, simulate_from_model_power_template
except ImportError:  # pragma: no cover
    from .model_power import ModelPowerTemplate, simulate_from_model_power_template


def _prepare_fold_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    is_signal: np.ndarray,
    *,
    min_observed_fraction: float,
    top_k: int | None,
) -> dict[str, Any]:
    """Fit filtering, imputation, and feature selection on one training fold."""
    feature_indices = np.arange(X_train.shape[1])
    observed_control = np.mean(~np.isnan(X_train[y_train == 0]), axis=0)
    observed_case = np.mean(~np.isnan(X_train[y_train == 1]), axis=0)
    keep = (
        (observed_control >= min_observed_fraction)
        & (observed_case >= min_observed_fraction)
    )
    X_train = X_train[:, keep]
    X_test = X_test[:, keep]
    feature_indices = feature_indices[keep]
    if X_train.shape[1] == 0:
        raise RuntimeError("No CpGs pass the fold-specific observation filter.")

    medians = np.nanmedian(X_train, axis=0)
    keep = np.isfinite(medians)
    X_train = X_train[:, keep]
    X_test = X_test[:, keep]
    feature_indices = feature_indices[keep]
    medians = medians[keep]
    if X_train.shape[1] == 0:
        raise RuntimeError("No CpGs remain after fold-specific imputation checks.")

    X_train = np.where(np.isnan(X_train), medians[None, :], X_train)
    X_test = np.where(np.isnan(X_test), medians[None, :], X_test)

    variance = np.var(X_train, axis=0)
    keep = np.isfinite(variance) & (variance > 0)
    X_train = X_train[:, keep]
    X_test = X_test[:, keep]
    feature_indices = feature_indices[keep]
    if X_train.shape[1] == 0:
        raise RuntimeError("No variable CpGs remain in the training fold.")

    if top_k is not None and top_k < X_train.shape[1]:
        scores, _ = f_classif(X_train, y_train)
        # f_classif may preserve float32 input. Cast before replacing +inf with
        # float64 max to avoid NumPy's "overflow encountered in cast" warning.
        scores = np.asarray(scores, dtype=np.float64)
        scores = np.nan_to_num(
            scores,
            nan=-np.inf,
            posinf=np.finfo(np.float64).max,
            neginf=-np.inf,
        )
        selected_local = np.argsort(scores)[::-1][:top_k]
        X_train = X_train[:, selected_local]
        X_test = X_test[:, selected_local]
        feature_indices = feature_indices[selected_local]

    selected_signal = is_signal[feature_indices]
    total_signal = int(is_signal.sum())
    n_selected_signal = int(selected_signal.sum())
    return {
        "X_train": X_train,
        "X_test": X_test,
        "selected_feature_indices": feature_indices,
        "feature_recall": n_selected_signal / total_signal if total_signal else np.nan,
        "feature_precision": (
            n_selected_signal / len(feature_indices) if len(feature_indices) else np.nan
        ),
    }


def _make_estimator(
    model_name: str,
    *,
    model_seed: int,
    logreg_params: Mapping[str, Any],
    rf_params: Mapping[str, Any],
):
    if model_name == "logreg":
        params = dict(logreg_params)
        params.setdefault("random_state", model_seed)
        return LogisticRegression(**params), True
    if model_name == "rf":
        params = dict(rf_params)
        params.setdefault("random_state", model_seed)
        return RandomForestClassifier(**params), False
    raise ValueError(f"Unknown model: {model_name}")


def _sensitivity_at_specificity(
    y_true: np.ndarray,
    score: np.ndarray,
    target_specificity: float,
) -> float:
    fpr, tpr, _ = roc_curve(y_true, score)
    allowed = fpr <= (1.0 - target_specificity + 1e-12)
    return float(np.max(tpr[allowed])) if np.any(allowed) else 0.0


def _mean_pairwise_jaccard(feature_sets: Sequence[set[int]]) -> float:
    if len(feature_sets) < 2:
        return np.nan
    values = []
    for left_index in range(len(feature_sets) - 1):
        for right_index in range(left_index + 1, len(feature_sets)):
            left = feature_sets[left_index]
            right = feature_sets[right_index]
            union = left | right
            values.append(len(left & right) / len(union) if union else 1.0)
    return float(np.mean(values))


def _run_cv_for_depth_model(
    X: np.ndarray,
    y: np.ndarray,
    template: ModelPowerTemplate,
    *,
    model_name: str,
    cv_folds: int,
    cv_repeats: int,
    top_k: int | None,
    min_observed_fraction: float,
    specificity_target: float,
    logreg_params: Mapping[str, Any],
    rf_params: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    repeat_aucs = []
    repeat_sensitivities = []
    fold_feature_sets = []
    fold_recall = []
    fold_precision = []

    for repeat in range(cv_repeats):
        split_seed = int(
            np.random.SeedSequence([seed, repeat]).generate_state(1, dtype=np.uint32)[0]
        )
        splitter = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=split_seed,
        )
        oof_score = np.full(len(y), np.nan, dtype=np.float64)

        for fold, (train_index, test_index) in enumerate(splitter.split(X, y)):
            prepared = _prepare_fold_features(
                X[train_index],
                y[train_index],
                X[test_index],
                template.is_signal,
                min_observed_fraction=min_observed_fraction,
                top_k=top_k,
            )
            estimator, needs_scaling = _make_estimator(
                model_name,
                model_seed=int(
                    np.random.SeedSequence([seed, repeat, fold]).generate_state(
                        1, dtype=np.uint32
                    )[0]
                ),
                logreg_params=logreg_params,
                rf_params=rf_params,
            )
            X_train = prepared["X_train"]
            X_test = prepared["X_test"]
            if needs_scaling:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
            estimator.fit(X_train, y[train_index])
            oof_score[test_index] = estimator.predict_proba(X_test)[:, 1]

            fold_feature_sets.append(
                set(prepared["selected_feature_indices"].astype(int).tolist())
            )
            fold_recall.append(float(prepared["feature_recall"]))
            fold_precision.append(float(prepared["feature_precision"]))

        if not np.isfinite(oof_score).all():
            raise RuntimeError("Cross-validation did not produce all OOF scores.")
        repeat_aucs.append(float(roc_auc_score(y, oof_score)))
        repeat_sensitivities.append(
            _sensitivity_at_specificity(y, oof_score, specificity_target)
        )

    return {
        "cv_auc": float(np.mean(repeat_aucs)),
        "cv_auc_sd_across_repeats": (
            float(np.std(repeat_aucs, ddof=1)) if len(repeat_aucs) > 1 else 0.0
        ),
        "sensitivity_at_specificity": float(np.mean(repeat_sensitivities)),
        "mean_feature_recall": float(np.nanmean(fold_recall)),
        "mean_feature_precision": float(np.nanmean(fold_precision)),
        "selection_jaccard": _mean_pairwise_jaccard(fold_feature_sets),
    }


def _run_one_replicate(
    simulation: int,
    simulation_seed: int,
    template: ModelPowerTemplate,
    *,
    total_sample_size: int,
    ratio: float,
    models: tuple[str, ...],
    cv_folds: int,
    cv_repeats: int,
    top_k: int | None,
    min_observed_fraction: float,
    target_auc: float,
    specificity_target: float,
    paired_depths: bool,
    logreg_params: Mapping[str, Any],
    rf_params: Mapping[str, Any],
) -> list[dict[str, Any]]:
    X_by_depth, y = simulate_from_model_power_template(
        template,
        total_sample_size=total_sample_size,
        ratio=ratio,
        paired_depths=paired_depths,
        random_state=simulation_seed,
    )
    rows = []
    for depth_index, mean_depth in enumerate(template.depths):
        for model_index, model_name in enumerate(models):
            model_seed = int(
                np.random.SeedSequence(
                    [simulation_seed, depth_index, model_index]
                ).generate_state(1, dtype=np.uint32)[0]
            )
            metrics = _run_cv_for_depth_model(
                X_by_depth[mean_depth],
                y,
                template,
                model_name=model_name,
                cv_folds=cv_folds,
                cv_repeats=cv_repeats,
                top_k=top_k,
                min_observed_fraction=min_observed_fraction,
                specificity_target=specificity_target,
                logreg_params=logreg_params,
                rf_params=rf_params,
                seed=model_seed,
            )
            success = metrics["cv_auc"] >= target_auc
            rows.append(
                {
                    "simulation": simulation,
                    "simulation_seed": simulation_seed,
                    "sample_size": total_sample_size,
                    "n_cases": int(np.sum(y == 1)),
                    "n_controls": int(np.sum(y == 0)),
                    "depth": mean_depth,
                    "mean_depth": mean_depth,
                    "model": model_name,
                    "cv_folds": cv_folds,
                    "cv_repeats": cv_repeats,
                    "target_auc": target_auc,
                    "specificity_target": specificity_target,
                    "target_success": bool(success),
                    "power_success": bool(success),
                    **metrics,
                }
            )
    return rows


def run_cv_discovery_power_analysis(
    template: ModelPowerTemplate,
    *,
    total_sample_size: int,
    ratio: float = 1.0,
    models: Sequence[str] = ("logreg",),
    cv_folds: int = 5,
    cv_repeats: int = 1,
    top_k: int | None = 10,
    min_observed_fraction: float = 0.5,
    target_auc: float = 0.75,
    specificity_target: float = 0.90,
    n_simulations: int = 10,
    paired_depths: bool = True,
    logreg_params: Mapping[str, Any] | None = None,
    rf_params: Mapping[str, Any] | None = None,
    n_jobs: int = 1,
    parallel_prefer: str = "threads",
    random_state: int = 0,
) -> dict[str, Any]:
    """Estimate CV biomarker-discovery power for one fixed template."""
    models = tuple(models)
    if not models or set(models) - {"logreg", "rf"}:
        raise ValueError("models may contain only 'logreg' and 'rf'.")
    if cv_folds < 2 or cv_repeats < 1:
        raise ValueError("cv_folds must be >=2 and cv_repeats must be positive.")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be positive or None.")
    if not 0 < min_observed_fraction <= 1:
        raise ValueError("min_observed_fraction must be in (0, 1].")
    if not 0.5 <= target_auc <= 1:
        raise ValueError("target_auc must be between 0.5 and 1.")
    if not 0 < specificity_target < 1:
        raise ValueError("specificity_target must be between 0 and 1.")
    if n_simulations < 1:
        raise ValueError("n_simulations must be positive.")
    if parallel_prefer not in {"threads", "processes"}:
        raise ValueError("parallel_prefer must be 'threads' or 'processes'.")

    expected_cases = int(round(total_sample_size * ratio / (1.0 + ratio)))
    expected_controls = total_sample_size - expected_cases
    if min(expected_cases, expected_controls) < cv_folds:
        raise ValueError(
            "Each class must contain at least cv_folds subjects. Reduce "
            "cv_folds or increase total_sample_size."
        )

    logreg_defaults = {
        "C": 1.0,
        "solver": "liblinear",
        "max_iter": 2000,
        "class_weight": None,
    }
    logreg_defaults.update(dict(logreg_params or {}))
    rf_defaults = {
        "n_estimators": 100,
        "max_features": "sqrt",
        "min_samples_leaf": 5,
        "class_weight": None,
        "n_jobs": 1,
    }
    rf_defaults.update(dict(rf_params or {}))

    children = np.random.SeedSequence(random_state).spawn(n_simulations)
    seeds = [int(child.generate_state(1, dtype=np.uint32)[0]) for child in children]
    start = time.perf_counter()

    # The public app uses threads. Make that backend explicit rather than a
    # soft preference so joblib never starts loky worker/resource-tracker
    # processes on macOS or Streamlit Community Cloud.
    parallel_kwargs: dict[str, Any]
    if parallel_prefer == "threads":
        parallel_kwargs = {"backend": "threading"}
    else:
        parallel_kwargs = {"prefer": "processes"}

    nested = Parallel(n_jobs=n_jobs, **parallel_kwargs)(
        delayed(_run_one_replicate)(
            simulation=index,
            simulation_seed=seeds[index],
            template=template,
            total_sample_size=total_sample_size,
            ratio=ratio,
            models=models,
            cv_folds=cv_folds,
            cv_repeats=cv_repeats,
            top_k=top_k,
            min_observed_fraction=min_observed_fraction,
            target_auc=target_auc,
            specificity_target=specificity_target,
            paired_depths=paired_depths,
            logreg_params=logreg_defaults,
            rf_params=rf_defaults,
        )
        for index in range(n_simulations)
    )
    replicate_results = pd.DataFrame(
        [row for replicate_rows in nested for row in replicate_rows]
    )
    summary = (
        replicate_results.groupby(["model", "mean_depth"], as_index=False)
        .agg(
            n_simulations=("power_success", "size"),
            mean_cv_auc=("cv_auc", "mean"),
            sd_cv_auc=("cv_auc", "std"),
            median_cv_auc=("cv_auc", "median"),
            target_attainment_probability=("power_success", "mean"),
            mean_sensitivity_at_specificity=("sensitivity_at_specificity", "mean"),
            mean_feature_recall=("mean_feature_recall", "mean"),
            mean_feature_precision=("mean_feature_precision", "mean"),
            mean_selection_jaccard=("selection_jaccard", "mean"),
        )
    )
    summary["power"] = summary["target_attainment_probability"]
    return {
        "replicate_results": replicate_results,
        "power_summary": summary,
        "analysis_metadata": {
            "analysis_mode": "cv_discovery",
            "elapsed_seconds": time.perf_counter() - start,
            "n_simulations": n_simulations,
            "paired_depths": paired_depths,
            "parallel_backend": (
                "threading" if parallel_prefer == "threads" else "processes"
            ),
        },
    }


def _binary_ci(
    group: pd.DataFrame,
    *,
    ci_method: str,
    confidence: float,
    n_bootstrap: int,
    random_state: int,
) -> tuple[float, float]:
    if ci_method == "none":
        return np.nan, np.nan
    values = group["power_success"].to_numpy(dtype=bool)
    if ci_method == "pooled_wilson":
        low, high = proportion_confint(
            int(values.sum()), len(values), alpha=1.0 - confidence, method="wilson"
        )
        return float(low), float(high)

    values_by_template = {
        template_id: subgroup["power_success"].to_numpy(dtype=float)
        for template_id, subgroup in group.groupby("template_id")
    }
    template_ids = np.asarray(list(values_by_template))
    if len(template_ids) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(random_state)
    bootstrap = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        sampled_ids = rng.choice(template_ids, size=len(template_ids), replace=True)
        template_means = []
        for template_id in sampled_ids:
            template_values = values_by_template[template_id]
            template_means.append(
                float(
                    rng.choice(
                        template_values,
                        size=len(template_values),
                        replace=True,
                    ).mean()
                )
            )
        bootstrap[index] = float(np.mean(template_means))
    tail = (1.0 - confidence) / 2.0
    return float(np.quantile(bootstrap, tail)), float(np.quantile(bootstrap, 1.0 - tail))


def run_power_sample_size_grid(
    templates: Sequence[Mapping[str, Any]],
    sample_sizes: Sequence[int],
    *,
    simulations_per_template: int = 10,
    power_kwargs: Mapping[str, Any] | None = None,
    ci_method: str = "none",
    confidence: float = 0.95,
    n_bootstrap: int = 500,
    n_jobs: int = 1,
    random_state: int = 0,
) -> dict[str, Any]:
    """Estimate CV discovery power over total sample size and mean depth."""
    if not templates:
        raise ValueError("templates must contain at least one template.")
    sample_sizes = tuple(int(value) for value in sample_sizes)
    if not sample_sizes or any(value < 4 for value in sample_sizes):
        raise ValueError("sample_sizes must contain study sizes >=4.")
    if len(set(sample_sizes)) != len(sample_sizes):
        raise ValueError("sample_sizes contains duplicated values.")
    if simulations_per_template < 1:
        raise ValueError("simulations_per_template must be positive.")
    if ci_method not in {"none", "pooled_wilson", "hierarchical_bootstrap"}:
        raise ValueError(
            "ci_method must be 'none', 'pooled_wilson', or 'hierarchical_bootstrap'."
        )
    if ci_method == "hierarchical_bootstrap" and n_bootstrap < 100:
        raise ValueError("Use at least 100 hierarchical bootstrap replicates.")

    kwargs = dict(power_kwargs or {})
    forbidden = {"template", "total_sample_size", "n_simulations", "n_jobs", "random_state"}
    conflicts = forbidden.intersection(kwargs)
    if conflicts:
        raise ValueError(
            "Remove controlled power_kwargs: " + ", ".join(sorted(conflicts))
        )

    all_results = []
    start = time.perf_counter()
    for sample_index, sample_size in enumerate(sample_sizes):
        for record in templates:
            template_id = int(record["template_id"])
            run_seed = int(
                np.random.SeedSequence(
                    [random_state, sample_index, template_id]
                ).generate_state(1, dtype=np.uint32)[0]
            )
            result = run_cv_discovery_power_analysis(
                record["template"],
                total_sample_size=sample_size,
                n_simulations=simulations_per_template,
                n_jobs=n_jobs,
                random_state=run_seed,
                **kwargs,
            )
            replicate = result["replicate_results"].copy()
            replicate.insert(0, "template_id", template_id)
            replicate.insert(1, "template_seed", int(record.get("template_seed", template_id)))
            all_results.append(replicate)

    replicate_results = pd.concat(all_results, ignore_index=True)
    template_summary = (
        replicate_results.groupby(
            ["template_id", "sample_size", "model", "mean_depth"], as_index=False
        )
        .agg(
            n_simulations=("power_success", "size"),
            mean_cv_auc=("cv_auc", "mean"),
            target_attainment_probability=("power_success", "mean"),
            mean_sensitivity_at_specificity=("sensitivity_at_specificity", "mean"),
            mean_feature_recall=("mean_feature_recall", "mean"),
            mean_feature_precision=("mean_feature_precision", "mean"),
            mean_selection_jaccard=("selection_jaccard", "mean"),
        )
    )

    rows = []
    for (sample_size, model_name, mean_depth), group in replicate_results.groupby(
        ["sample_size", "model", "mean_depth"], sort=True
    ):
        template_group = template_summary.loc[
            (template_summary["sample_size"] == sample_size)
            & (template_summary["model"] == model_name)
            & (template_summary["mean_depth"] == mean_depth)
        ]
        point_power = float(template_group["target_attainment_probability"].mean())
        ci_seed = int(
            np.random.SeedSequence(
                [random_state, sample_size, int(mean_depth * 100)]
            ).generate_state(1, dtype=np.uint32)[0]
        )
        ci_low, ci_high = _binary_ci(
            group,
            ci_method=ci_method,
            confidence=confidence,
            n_bootstrap=n_bootstrap,
            random_state=ci_seed,
        )
        rows.append(
            {
                "sample_size": sample_size,
                "model": model_name,
                "depth": mean_depth,
                "mean_depth": mean_depth,
                "n_templates": int(template_group["template_id"].nunique()),
                "simulations_per_template": simulations_per_template,
                "total_simulations": len(group),
                "mean_cv_auc": float(template_group["mean_cv_auc"].mean()),
                "target_attainment_probability": point_power,
                "power": point_power,
                "power_ci_low": ci_low,
                "power_ci_high": ci_high,
                "between_template_power_sd": (
                    float(template_group["target_attainment_probability"].std(ddof=1))
                    if len(template_group) > 1 else np.nan
                ),
                "mean_sensitivity_at_specificity": float(
                    template_group["mean_sensitivity_at_specificity"].mean()
                ),
                "mean_feature_recall": float(template_group["mean_feature_recall"].mean()),
                "mean_feature_precision": float(template_group["mean_feature_precision"].mean()),
                "mean_selection_jaccard": float(template_group["mean_selection_jaccard"].mean()),
                "ci_method": ci_method,
            }
        )

    power_curve = pd.DataFrame(rows).sort_values(
        ["model", "mean_depth", "sample_size"]
    ).reset_index(drop=True)
    return {
        "replicate_results": replicate_results,
        "template_summary": template_summary,
        "power_curve": power_curve,
        "run_metadata": {
            "analysis_mode": "cv_discovery",
            "sample_sizes": list(sample_sizes),
            "n_templates": len(templates),
            "simulations_per_template": simulations_per_template,
            "total_study_simulations": (
                len(templates) * len(sample_sizes) * simulations_per_template
            ),
            "ci_method": ci_method,
            "confidence": confidence if ci_method != "none" else None,
            "n_bootstrap": n_bootstrap if ci_method == "hierarchical_bootstrap" else None,
            "elapsed_seconds": time.perf_counter() - start,
        },
    }
