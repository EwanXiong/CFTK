"""Modality performance evaluation via repeated stratified k-fold CV."""

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

np.random.seed(0)


def wilcoxon(X, y):
    """
    Wilcoxon rank-sum score function for GenericUnivariateSelect.
    Returns NEGATIVE p-value so larger = more significant (matches original mesa).
    """
    return -mannwhitneyu(X[y == 0], X[y == 1])[1]


def feature_preprocessing(X, max_missing_ratio=0.1, normalization=True):
    """Filter high-missing features, impute, optionally normalize."""
    keep = (X.isna().sum(axis=0) <= X.shape[0] * max_missing_ratio).values
    X_f  = X.iloc[:, keep]
    imp  = SimpleImputer(strategy="mean")
    arr  = imp.fit_transform(X_f)
    if normalization:
        arr = Normalizer().fit_transform(arr)
    return pd.DataFrame(arr, index=X_f.index, columns=X_f.columns)


def feature_sampler(X, subset="all"):
    """Subsample features by count, fraction, or 'all'."""
    if subset == "all":
        return X
    if isinstance(subset, int):
        idx = np.random.choice(X.shape[1], subset, replace=False)
    else:
        idx = np.random.choice(X.shape[1], int(X.shape[1] * subset), replace=False)
    return X.iloc[:, idx]


def modality_performance(
    modality_name,
    modality_matrix,
    y,
    classifiers,
    scoring="roc_auc",
    feature_size=100,
    selector=None,
    cv=None,
    subset=0.1,
    repeat=3,
    n_jobs=-1,
):
    """
    Evaluate each modality with each classifier via cross-validation.
    Returns summary DataFrame sorted by best AUC.
    """
    if selector is None:
        selector = GenericUnivariateSelect(score_func=wilcoxon, mode="k_best")
    if cv is None:
        cv = RepeatedStratifiedKFold(n_repeats=5, n_splits=5, random_state=0)

    selector.param = feature_size
    clf_names = [c.__class__.__name__ for c in classifiers]
    clf_abbr  = ["".join(ch for ch in n if ch.isupper()) for n in clf_names]

    cv_result = []
    for _ in range(repeat):
        preprocessed = [
            feature_preprocessing(feature_sampler(m, subset))
            for m in modality_matrix
        ]
        rep_scores = [
            [
                cross_val_score(
                    make_pipeline(selector, clf),
                    X, y, cv=cv, scoring=scoring, n_jobs=n_jobs,
                )
                for X in preprocessed
            ]
            for clf in classifiers
        ]
        cv_result.append(rep_scores)

    rows = []
    for m_idx in range(len(modality_matrix)):
        scores = np.hstack([
            np.array([rep[m_idx] for rep in r_scores])
            for r_scores in cv_result
        ])
        means = scores.mean(axis=1)
        stds  = scores.std(axis=1)
        best  = int(np.argmax(means))
        rows.append({
            f"clf_{scoring}_mean ({','.join(clf_abbr)})": means.round(4),
            f"clf_{scoring}_std ({','.join(clf_abbr)})":  stds.round(4),
            "best_classifier, idx": (clf_names[best], best),
            f"best_{scoring}_mean": round(means[best], 4),
        })

    summary = pd.DataFrame(rows, index=modality_name).sort_values(
        f"best_{scoring}_mean", ascending=False
    )
    return summary, modality_name, classifiers
