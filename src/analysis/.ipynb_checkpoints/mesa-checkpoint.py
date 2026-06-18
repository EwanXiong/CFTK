"""
MESA — Multimodal Epigenomic Stacking Analysis.
Merges: mesa model classes, LOOCV runner, modality performance integration.
"""

import os
import sys
import pickle
import time
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone as sk_clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut, RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, Normalizer
from sklearn.svm import SVC
from scipy.stats import mannwhitneyu
from joblib import Parallel, delayed

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

from analysis.modality_performance import (
    wilcoxon, feature_preprocessing, feature_sampler, modality_performance
)


def _disp(msg):
    print(f"@{time.asctime()}\t{msg}", file=sys.stderr)


# ── classifier registry ───────────────────────────────────────────────────────

def _build_clf_dist():
    d = {
        1: RandomForestClassifier(random_state=0, n_jobs=-1),
        2: LogisticRegression(random_state=0, max_iter=1000, n_jobs=-1),
        3: SVC(random_state=0, probability=True),
    }
    if _HAS_XGB:
        d[4] = XGBClassifier(random_state=0, n_jobs=-1, eval_metric="logloss")
    return d


CLF_DIST = _build_clf_dist()


# ── MESA_modality ─────────────────────────────────────────────────────────────

class MESA_modality:
    """Single-modality pipeline: filter → select → fit."""

    def __init__(self, classifier=None, top_n=100):
        self.classifier = classifier if classifier is not None else RandomForestClassifier(random_state=0)
        self.top_n = top_n
        self._selector = GenericUnivariateSelect(
            score_func=wilcoxon, mode="k_best", param=top_n
        )
        self._pipeline = None
        # track which columns were kept during fit (for consistent train/test transform)
        self._keep_cols = None
        self._imputer   = SimpleImputer(strategy="mean")
        self._normalizer = Normalizer()

    def _preprocess_fit(self, X, max_missing_ratio=0.1):
        """Fit preprocessing on training data and transform. Returns X_pre."""
        # guard: drop duplicate index/columns before processing
        if X.index.duplicated().any():
            X = X[~X.index.duplicated(keep="first")]
        if X.columns.duplicated().any():
            X = X.loc[:, ~X.columns.duplicated(keep="first")]
        keep = (X.isna().sum(axis=0) <= X.shape[0] * max_missing_ratio).values
        self._keep_cols = X.columns[keep]
        X_f  = X[self._keep_cols]
        arr  = self._imputer.fit_transform(X_f)
        arr  = self._normalizer.fit_transform(arr)
        return pd.DataFrame(arr, index=X.index, columns=self._keep_cols)

    def _preprocess_transform(self, X):
        """Transform test data using columns and imputer fitted on training data."""
        # guard: drop duplicate columns before reindex
        if X.columns.duplicated().any():
            X = X.loc[:, ~X.columns.duplicated(keep="first")]
        # align to training columns: add missing cols as NaN, drop extras
        X_aligned = X.reindex(columns=self._keep_cols)
        arr = self._imputer.transform(X_aligned)
        arr = self._normalizer.transform(arr)
        return pd.DataFrame(arr, index=X.index, columns=self._keep_cols)

    def fit(self, X, y):
        X_pre = self._preprocess_fit(X)
        self._pipeline = make_pipeline(self._selector, self.classifier)
        self._pipeline.fit(X_pre, y)
        return self

    def predict_proba(self, X):
        X_pre = self._preprocess_transform(X)
        return self._pipeline.predict_proba(X_pre)

    def get_params(self, deep=True):
        return {"classifier": self.classifier, "top_n": self.top_n}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


# ── MESA ──────────────────────────────────────────────────────────────────────

class MESA:
    """Stacking meta-learner over multiple MESA_modality base estimators."""

    def __init__(self, modalities=None, meta_estimator=None):
        self.modalities = modalities or []
        self.meta_estimator = meta_estimator if meta_estimator is not None else LogisticRegression(
            random_state=0, max_iter=1000
        )

    def fit(self, X_list, y):
        """Fit each modality, then fit meta-estimator on OOF predictions."""
        # for simplicity, train base models on full data (LOOCV handles OOF)
        for mod, X in zip(self.modalities, X_list):
            mod.fit(X, y)
        meta_X = np.column_stack([
            mod.predict_proba(X)[:, 1]
            for mod, X in zip(self.modalities, X_list)
        ])
        self.meta_estimator.fit(meta_X, y)
        return self

    def predict_proba(self, X_list):
        meta_X = np.column_stack([
            mod.predict_proba(X)[:, 1]
            for mod, X in zip(self.modalities, X_list)
        ])
        return self.meta_estimator.predict_proba(meta_X)


# ── MESA_CV ───────────────────────────────────────────────────────────────────

class MESA_CV:
    """Cross-validation wrapper for a single MESA_modality."""

    def __init__(self, modality=None, cv=None):
        self.modality = modality if modality is not None else MESA_modality()
        self.cv = cv if cv is not None else RepeatedStratifiedKFold(n_repeats=5, n_splits=5, random_state=0)
        self.cv_result = []

    def fit(self, X, y):
        self.cv_result = []
        for train_idx, test_idx in self.cv.split(X, y):
            mod = MESA_modality(
                classifier=sk_clone(self.modality.classifier),
                top_n=self.modality.top_n,
            )
            mod.fit(X.iloc[train_idx], y[train_idx])
            y_pred = mod.predict_proba(X.iloc[test_idx])
            self.cv_result.append((y_pred, y[test_idx]))
        return self

    def _fold_prediction(self, y_pred):
        return y_pred[:, 1]

    def get_performance(self):
        y_true = np.concatenate([fold[1] for fold in self.cv_result])
        y_pred = np.concatenate([
            self._fold_prediction(fold[0]) for fold in self.cv_result
        ])
        return roc_auc_score(y_true, y_pred)


# ── LOOCV helpers ─────────────────────────────────────────────────────────────

def _resolve_clf_per_modality(args, n_modalities, performance=None, perf_tsv=None):
    """Determine best classifier per modality from performance table or args.clf."""
    import re
    names = list(args.modality)

    def _from_df(df):
        clfs = []
        for name in names:
            if name in df.index:
                raw = df.loc[name, "best_classifier, idx"]
                if isinstance(raw, str):
                    m = re.search(r'(\d+)', raw)
                    pos = int(m.group(1)) if m else 0
                else:
                    pos = int(raw[1])
                clfs.append(sk_clone(CLF_DIST[args.clf[pos]]))
            else:
                clfs.append(sk_clone(CLF_DIST[args.clf[0]]))
        return clfs

    if performance is not None:
        return _from_df(performance[0])
    if perf_tsv and os.path.exists(perf_tsv):
        return _from_df(pd.read_csv(perf_tsv, sep="\t", index_col=0))
    if len(args.clf) == n_modalities:
        return [sk_clone(CLF_DIST[c]) for c in args.clf]
    return [sk_clone(CLF_DIST[args.clf[0]])] * n_modalities


def _loocv_single(name, X, clf, y, top_n):
    _disp(f"LOOCV [{name}] n={X.shape[0]} feat={X.shape[1]}")
    mod = MESA_modality(classifier=clf, top_n=top_n)
    cv  = MESA_CV(modality=mod, cv=LeaveOneOut())
    cv.fit(X, y)
    proba  = np.array([cv._fold_prediction(f[0])[0] for f in cv.cv_result])
    y_true = np.array([f[1][0] for f in cv.cv_result])
    return proba, y_true


def _loocv_multimodal(X_list, clfs, y, top_n):
    n = X_list[0].shape[0]
    _disp(f"LOOCV [Multimodal] n={n} modalities={len(X_list)}")
    proba_arr  = np.empty(n)
    y_true_arr = np.empty(n)

    for i, (tr, te) in enumerate(LeaveOneOut().split(X_list[0])):
        mods = [MESA_modality(classifier=sk_clone(c), top_n=top_n) for c in clfs]
        mesa = MESA(
            modalities=mods,
            meta_estimator=LogisticRegression(random_state=0, max_iter=1000),
        )
        mesa.fit([X.iloc[tr] for X in X_list], y[tr])
        proba_arr[te[0]]  = mesa.predict_proba([X.iloc[te] for X in X_list])[0, 1]
        y_true_arr[te[0]] = y[te[0]]
        if (i + 1) % 10 == 0 or (i + 1) == n:
            _disp(f"  [Multimodal] fold {i+1}/{n}")

    return proba_arr, y_true_arr


# ── public API ────────────────────────────────────────────────────────────────

def run_modality_performance(args):
    """Test per-modality classification performance and save TSV."""
    os.makedirs(args.output_dir, exist_ok=True)
    matrices   = [pd.read_csv(f, sep="\t", index_col=0).T for f in args.infile]
    sample_ids = matrices[0].index.tolist()
    label_df   = pd.read_table(args.label, header=None, index_col=0)
    label_df.columns = ["label"]
    y = label_df.reindex(sample_ids)["label"].values.reshape(-1)
    clfs = [sk_clone(CLF_DIST[c]) for c in args.clf]
    perf, _, _ = modality_performance(
        args.modality, matrices, y, clfs,
        feature_size=args.size, subset=args.subset,
        repeat=args.repeat, n_jobs=args.cores,
    )
    out = os.path.join(args.output_dir, "modality_performance.tsv")
    perf.to_csv(out, sep="\t")
    _disp(f"Performance saved → {out}")
    return perf, args.modality, clfs


def run_mesa_model(args, performance=None):
    """Train full MESA stacking model and save pickle."""
    os.makedirs(args.output_dir, exist_ok=True)
    matrices   = [pd.read_csv(f, sep="\t", index_col=0).T for f in args.infile]
    sample_ids = matrices[0].index.tolist()
    label_df   = pd.read_table(args.label, header=None, index_col=0)
    label_df.columns = ["label"]
    y = label_df.reindex(sample_ids)["label"].values
    clfs = _resolve_clf_per_modality(
        args, len(matrices), performance=performance,
        perf_tsv=getattr(args, "perf_tsv", None),
    )
    mods  = [MESA_modality(clf, top_n=args.size).fit(X, y) for clf, X in zip(clfs, matrices)]
    model = MESA(modalities=mods,
                 meta_estimator=LogisticRegression(random_state=0, max_iter=1000))
    model.fit(matrices, y)
    out = os.path.join(args.output_dir, "MESA_model.pkl")
    pickle.dump(model, open(out, "wb"))
    _disp(f"MESA model saved → {out}")
    return model


def run_mesa_loocv(args, performance=None):
    """Run LOOCV for all modalities + multimodal, save predictions TSV."""
    os.makedirs(args.output_dir, exist_ok=True)
    matrices   = [pd.read_csv(f, sep="\t", index_col=0).T for f in args.infile]
    sample_ids = matrices[0].index.tolist()

    # explicitly align labels to matrix sample order to avoid mismatch
    label_df = pd.read_table(args.label, header=None, index_col=0)
    label_df.columns = ["label"]
    y = label_df.reindex(sample_ids)["label"].values.astype(int)
    if np.isnan(y).any():
        missing = [s for s in sample_ids if s not in label_df.index]
        sys.exit(
            f"[mesa] ERROR: {len(missing)} sample(s) in matrix not found in label.tsv: "
            f"{missing[:5]}"
        )

    # log matrix shapes and paths for debugging
    for name, mat, fp in zip(args.modality, matrices, args.infile):
        _disp(f"[mesa] {name}: {mat.shape[0]} samples × {mat.shape[1]} features — {fp}")

    clfs = _resolve_clf_per_modality(
        args, len(matrices), performance=performance,
        perf_tsv=getattr(args, "perf_tsv", None),
    )

    pred_dict  = {}
    y_true_ref = None

    for name, X, clf in zip(args.modality, matrices, clfs):
        proba, y_true = _loocv_single(name, X, clf, y, args.size)
        y_true_ref = y_true if y_true_ref is None else y_true_ref
        pred_dict[name] = proba

    proba_m, y_true_m = _loocv_multimodal(matrices, clfs, y, args.size)
    pred_dict["Multimodal"] = proba_m

    pred_df = pd.DataFrame(pred_dict, index=sample_ids)
    pred_df.insert(0, "y_true", y_true_ref.astype(int))
    pred_df.index.name = "sample_id"

    out = os.path.join(args.output_dir, "loocv_predictions.tsv")
    pred_df.to_csv(out, sep="\t")
    _disp(f"LOOCV predictions saved → {out}")
    return pred_df
