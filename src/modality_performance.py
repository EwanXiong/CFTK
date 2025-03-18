import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

np.random.seed(0)


def feature_sampler(subset, matrix):
    if subset == "all":
        return matrix
    elif isinstance(subset, int):
        return matrix.iloc[:, np.random.choice(matrix.shape[1], subset, replace=False)]
    elif isinstance(subset, float):
        return matrix.iloc[
            :,
            np.random.choice(matrix.shape[1], int(matrix.shape[1] * subset), replace=False),
        ]
    elif isinstance(subset, str):
        return matrix.iloc[:, matrix.columns.str.startswith(subset)]
    else:
        raise ValueError("subset must be int/float/str")


def feature_preprocessing(X, max_missing_ratio=0.1, normalization=True):
    """
    Preprocessing for feature matrix.

    Args:
        X: dataframe or array of shape (n_features, n_samples)
        y: array-like of shape (n_samples,)
        max_missing_ratio: float, default = 0.1
            The maximum missing ratio for features.
        normalization: bool, default = True
            If True, normalize the feature matrix.

    Returns:
        X: dataframe or array of shape (n_features, n_samples)
        y: array-like of shape (n_samples,)
    """
    X_filtered = X.iloc[
        :, (X.isna().sum(axis=0) <= (X.shape[0] * max_missing_ratio)).values
    ]
    imputer = SimpleImputer(strategy="mean")
    if normalization:
        X_cleaned = pd.DataFrame(
            Normalizer().fit_transform(imputer.fit_transform(X_filtered))
        )
    else:
        X_cleaned = pd.DataFrame(imputer.fit_transform(X_filtered))
    X_cleaned.index = X_filtered.index
    X_cleaned.columns = X_filtered.columns
    return X_cleaned


def wilcoxon(X, y):
    """
    Score function for feature selection using Wilcoxon rank-sum test.

    Args:
        X: dataframe or array of shape (n_features, n_samples)
        y: array-like of shape (n_samples,)

    Returns:
        p-values of Wilcoxon rank-sum test for each feature
    """
    return mannwhitneyu(X[y == 0], X[y == 1])[1]


def modality_performance(
    modality_name,
    modality_matrix,
    y,
    classifiers,
    scoring="roc_auc",
    performance_threshold=0.80,
    feature_size=100,
    selector=GenericUnivariateSelect(score_func=wilcoxon, mode="k_best"),
    cv=RepeatedStratifiedKFold(n_repeats=5, n_splits=5, random_state=0),
    subset="all",
    repeat=5,
    verbose=0,
    n_jobs=-1
):
    """
    Main function for automatic modality performance test.

    Parameters
    ----------
    modality_name: list/array of strings
        Names for moldality to input (order-sensitive).
    modality_matrix: list/array of dataframe(s)
        Matrixs for maldalities, dataframe of shape (n_features, n_samples).
    y: list/array of int
        Labels for samples.
    performance_threshold: float, default = 0.80
        Performance threshold for modality selection.
        Only modality has performance >= threshold kept for final model.
    feature_size: int, default = 100
        The number of top features to test and compare.
    subset: str/int/float, default = "all"
        Chromosome to test and compare.
        When integer, randomly pick n features from features.
        When float, randomly pick n*100% features from features.
        When "all", test all features.
        When "chrN"(N = 1-22 or X/Y), test features in chromosome N.
    repeat: int, default = 5
        if subset is not "all", repeat the sampling for n times.

    Return
    ---------
    test_summary:
    """
    clf_names = [c.__class__.__name__ for c in classifiers]
    clf_names_short = [
        "".join([char for char in text if char.isupper()]) for text in clf_names
    ]
    selector.param = feature_size
    cv_result = []

    for r in range(repeat):
        modality_matrix_subset = [
            feature_preprocessing(
                feature_sampler(subset, matrix), max_missing_ratio=0.1
            )
            for matrix in modality_matrix
        ]

        cv_result_temp = [
            [
                cross_val_score(
                    make_pipeline(selector, c),
                    X,
                    y,
                    cv=cv,
                    n_jobs=n_jobs,
                    scoring=scoring,
                    verbose=verbose,
                )
                for X in modality_matrix_subset
            ]
            for c in classifiers
        ]
        cv_result.append(cv_result_temp)

    cv_result_summary = []
    for m in range(len(modality_matrix)):
        temp = np.hstack([np.array([c[m] for c in rep]) for rep in cv_result])
        cv_result_summary.append(
            [
                np.around(temp.mean(axis=1), decimals=4),
                np.around(temp.std(axis=1), decimals=4),
                (clf_names[np.argmax(temp.mean(axis=1))], np.argmax(temp.mean(axis=1))),
                np.around(temp.mean(axis=1).max(), decimals=4),
            ]
        )

    cv_result_summary = pd.DataFrame(
        cv_result_summary,
        columns=[
            f"classifier_{scoring}_mean (%s)" % (",".join(clf_names_short)),
            f"classifier_{scoring}_std (%s)" % (",".join(clf_names_short)),
            "best_classifier, idx",
            f"best_{scoring}_mean",
        ],
        index=modality_name,
    ).sort_values(f"best_{scoring}_mean", ascending=False)
    # print(cv_result_summary)
    return cv_result_summary, modality_name, classifiers
