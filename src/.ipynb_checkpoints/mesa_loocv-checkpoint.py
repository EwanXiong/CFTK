"""
mesa_loocv.py
=============
LOOCV (Leave-One-Out Cross-Validation) for MESA single-modality and
multimodal models.

Replaces the broken cv_mesa branch in util.py with a correct implementation:
  1. Runs LOOCV for each single modality using its best classifier.
  2. Runs LOOCV for the multimodal MESA model.
  3. Saves all predictions to a single TSV file.
  4. Calls the visualization script.

Integration into util.py
"""

import os
import sys
import time

# Ensure the directory containing this script is always on sys.path,
# so that mesa_cv_plot.py can be imported regardless of the working directory.
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.svm import SVC

from mesa import MESA, MESA_CV, MESA_modality

try:
    from xgboost import XGBClassifier
    _clf_dist = {
        1: RandomForestClassifier(random_state=0, n_jobs=-1),
        2: LogisticRegression(random_state=0, n_jobs=-1),
        3: SVC(random_state=0, probability=True),  # probability=True for predict_proba
        4: XGBClassifier(random_state=0, n_jobs=-1),
    }
except ImportError:
    _clf_dist = {
        1: RandomForestClassifier(random_state=0, n_jobs=-1),
        2: LogisticRegression(random_state=0, n_jobs=-1),
        3: SVC(random_state=0, probability=True),
    }


def disp(txt):
    print("@%s \t%s" % (time.asctime(), txt), file=sys.stderr)


# ══════════════════════════════════════════════════════════════════════════════
# Extract predictions from MESA_CV.cv_result
# ══════════════════════════════════════════════════════════════════════════════

def _get_predictions(mesa_cv):
    """
    Extract per-sample LOOCV predicted probabilities in original sample order.

    This works without any patch to MESA_CV because:
    - LeaveOneOut always yields test=[0], [1], ..., [n-1] in order.
    - joblib.Parallel preserves output order relative to input order.
    So cv_result[i] always corresponds to the i-th sample in the input.

    Parameters
    ----------
    mesa_cv : fitted MESA_CV instance (cv must be LeaveOneOut)

    Returns
    -------
    y_pred_proba : np.ndarray, shape (n_samples,)
        Predicted probability of the positive class for each sample.
    y_true : np.ndarray, shape (n_samples,)
        True labels in original sample order.
    """
    # cv_result is a list of (y_pred, y_test) tuples, one per fold.
    # For LeaveOneOut each fold has exactly 1 test sample:
    #   y_pred shape: (1, n_classes)
    #   y_test shape: (1,)
    # _fold_prediction extracts the positive-class probability column.
    y_pred_proba = np.array([
        mesa_cv._fold_prediction(fold[0])[0]  # scalar: P(Cancer) for sample i
        for fold in mesa_cv.cv_result
    ])
    y_true = np.array([
        fold[1][0]                            # true label for sample i
        for fold in mesa_cv.cv_result
    ])
    return y_pred_proba, y_true


# ══════════════════════════════════════════════════════════════════════════════
# Classifier resolution
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_clf_per_modality(args, modality_matrix, performance=None,
                               perf_tsv=None):
    """
    Determine the best classifier for each modality.

    The "best_classifier, idx" column in the performance table stores
    (clf_name, position_idx) where position_idx is the index into the
    classifiers list passed to modality_performance() — i.e. the index
    into args.clf. For example if args.clf=[1,2,3] and idx=2, the best
    clf is args.clf[2]=3 → SVC.

    Priority order:
      1. If performance tuple is provided (from --performance in same run)
         → use best_classifier position index per modality.
      2. Elif perf_tsv path is provided (pre-computed modality_performance.tsv)
         → load TSV and use best_classifier position index per modality.
      3. Elif len(args.clf) == len(modality_matrix) → one-to-one mapping.
      4. Else → use clf[0] for all modalities.

    Parameters
    ----------
    args             : argparse.Namespace
    modality_matrix  : list of pd.DataFrame, one per modality
    performance      : tuple returned by mesa_performance(), or None
    perf_tsv         : str, path to modality_performance.tsv, or None

    Returns
    -------
    modality_clfs  : list of sklearn estimators, one per modality
    modality_names : list of str, modality names in input order
    """
    import ast
    n              = len(modality_matrix)
    modality_names = list(args.modality)

    def _clf_from_perf_df(perf_df):
        """Extract one clf per modality from a performance DataFrame."""
        import re
        clfs = []
        for name in modality_names:
            if name in perf_df.index:
                raw = perf_df.loc[name, "best_classifier, idx"]
                # raw is a string like "('RandomForestClassifier', np.int64(0))"
                # or a tuple if already parsed (when called from same-session performance).
                # ast.literal_eval cannot handle np.int64(...) — use regex instead.
                if isinstance(raw, str):
                    # Extract the first integer found in the string, which is the
                    # position index. Handles both "0" and "np.int64(0)" formats.
                    clf_name_match = re.search(r"'([^']+)'", raw)
                    idx_match      = re.search(r'(\d+)', raw)
                    clf_name = clf_name_match.group(1) if clf_name_match else "unknown"
                    pos_idx  = int(idx_match.group(1)) if idx_match else 0
                else:
                    # Already a tuple: (clf_name, pos_idx)
                    clf_name = str(raw[0])
                    pos_idx  = int(raw[1])

                clf_key = args.clf[pos_idx]   # position index → _clf_dist key
                clfs.append(_clf_dist[clf_key])
                disp(f"  [{name}] best clf: {clf_name} "
                     f"(args.clf[{pos_idx}]={clf_key})")
            else:
                clfs.append(_clf_dist[args.clf[0]])
                disp(f"  [{name}] not in performance table, "
                     f"falling back to clf[0]={args.clf[0]}")
        return clfs

    if performance is not None:
        # Called from util.py with performance tuple from same run
        modality_clfs = _clf_from_perf_df(performance[0])

    elif perf_tsv is not None:
        # Called standalone: load pre-computed modality_performance.tsv
        disp(f"Loading performance table from: {perf_tsv}")
        perf_df = pd.read_csv(perf_tsv, sep="\t", index_col=0)
        modality_clfs = _clf_from_perf_df(perf_df)

    elif len(args.clf) == n:
        modality_clfs = [_clf_dist[c] for c in args.clf]
        disp("Using one-to-one clf mapping from --clf.")

    else:
        modality_clfs = [_clf_dist[args.clf[0]]] * n
        disp(f"len(--clf)={len(args.clf)} != n_modalities={n}. "
             f"Using clf[0]={args.clf[0]} for all.")

    return modality_clfs, modality_names


# ══════════════════════════════════════════════════════════════════════════════
# Single-modality LOOCV
# ══════════════════════════════════════════════════════════════════════════════

def _run_single_modality_loocv(name, X, clf, y, top_n):
    """
    Run LOOCV for one modality using the MESA_modality pipeline.

    The MESA_modality pipeline internally performs:
      missing-value filter → variance filter → Wilcoxon top-k selection
      → Boruta feature selection → final predictor (clf)

    Parameters
    ----------
    name  : str   modality name (for logging)
    X     : pd.DataFrame  shape (n_samples, n_features)
    clf   : sklearn estimator instance
    y     : np.ndarray    shape (n_samples,)
    top_n : int           number of features kept after Boruta

    Returns
    -------
    y_pred_proba : np.ndarray, shape (n_samples,)
    y_true       : np.ndarray, shape (n_samples,)
    """
    disp(f"  LOOCV [{name}] "
         f"n_samples={X.shape[0]}, n_features={X.shape[1]}, "
         f"clf={clf.__class__.__name__}, top_n={top_n}")

    modality_obj = MESA_modality(classifier=clf, top_n=top_n)

    mesa_cv = MESA_CV(modality=modality_obj, cv=LeaveOneOut())
    mesa_cv.fit(X, y)

    y_pred_proba, y_true = _get_predictions(mesa_cv)

    disp(f"  [{name}] done. "
         f"prob range [{y_pred_proba.min():.3f}, {y_pred_proba.max():.3f}]")
    return y_pred_proba, y_true


# ══════════════════════════════════════════════════════════════════════════════
# Multimodal MESA LOOCV
# ══════════════════════════════════════════════════════════════════════════════

def _run_multimodal_loocv(modality_matrix, modality_clfs, y, top_n):
    """
    Run LOOCV for the multimodal MESA stacking model using a manual loop.

    MESA does not implement get_params() so sklearn.base.clone() fails when
    MESA_CV tries to clone self.modality internally. We therefore bypass
    MESA_CV entirely and implement the LeaveOneOut loop manually:
      - For each fold: build a fresh MESA object, fit on train, predict on test.
      - Collect predictions in original sample order (LeaveOneOut is sequential).

    Parameters
    ----------
    modality_matrix : list of pd.DataFrame, one per modality
    modality_clfs   : list of sklearn estimators, one per modality
    y               : np.ndarray, shape (n_samples,)
    top_n           : int, Boruta top-n for each base modality

    Returns
    -------
    y_pred_proba : np.ndarray, shape (n_samples,)
    y_true       : np.ndarray, shape (n_samples,)
    """
    from sklearn.base import clone as sk_clone

    n_samples = modality_matrix[0].shape[0]
    disp(f"  LOOCV [Multimodal] "
         f"n_modalities={len(modality_matrix)}, n_samples={n_samples}")

    y_pred_proba = np.empty(n_samples, dtype=float)
    y_true_arr   = np.empty(n_samples, dtype=float)

    loo = LeaveOneOut()

    for fold_i, (train_idx, test_idx) in enumerate(loo.split(modality_matrix[0])):
        # Slice train / test for every modality
        # modality_matrix[m] is a DataFrame: rows=samples, cols=features
        X_train_list = [X.iloc[train_idx] for X in modality_matrix]
        X_test_list  = [X.iloc[test_idx]  for X in modality_matrix]
        y_train      = y[train_idx]
        y_test       = y[test_idx]

        # Build fresh MESA_modality objects for this fold (no clone needed)
        base_modalities = [
            MESA_modality(classifier=sk_clone(clf), top_n=top_n)
            for clf in modality_clfs
        ]

        # Build and fit a fresh MESA model on the training split
        mesa_model = MESA(
            modalities=base_modalities,
            meta_estimator=LogisticRegression(random_state=0, n_jobs=-1),
        )
        mesa_model.fit(X_train_list, y_train)

        # Predict probability of positive class for the single test sample
        proba = mesa_model.predict_proba(X_test_list)  # shape (1, 2)
        y_pred_proba[test_idx[0]] = proba[0, 1]
        y_true_arr[test_idx[0]]   = y_test[0]

        if (fold_i + 1) % 10 == 0 or (fold_i + 1) == n_samples:
            disp(f"  [Multimodal] fold {fold_i + 1}/{n_samples}")

    disp(f"  [Multimodal] done. "
         f"prob range [{y_pred_proba.min():.3f}, {y_pred_proba.max():.3f}]")
    return y_pred_proba, y_true_arr


# ══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════════════════

def run_mesa_loocv(args, performance=None, perf_tsv=None):
    """
    Run LOOCV for all single modalities and the multimodal MESA model,
    save predictions to TSV, and generate visualizations.

    Parameters
    ----------
    args        : argparse.Namespace
                  Required attributes:
                    args.infile      — list of feature matrix file paths
                    args.modality    — list of modality names (same order as infile)
                    args.label       — path to label TSV (index=sample_id, col=0/1)
                    args.clf         — list of classifier keys [int]
                    args.size        — top_n for Boruta feature selection
                    args.output_dir  — output directory
    performance : tuple or None
                  Return value of mesa_performance(args) when --performance
                  was also specified in the same run.
    perf_tsv    : str or None
                  Path to a pre-computed modality_performance.tsv file.
                  Used when calling this script standalone (not via util.py).
                  If both performance and perf_tsv are None, falls back to args.clf.

    Output
    ------
    <output_dir>/loocv_predictions.tsv
        Columns: sample_id | y_true | <modality1> | ... | Multimodal

    <output_dir>/mesa_roc_LOOCV.pdf
    <output_dir>/mesa_heatmap_LOOCV.pdf
    <output_dir>/mesa_spearman_LOOCV.pdf
    """
    # ── 1. Load modality matrices ─────────────────────────────────────────────
    disp("Loading modality matrices...")
    modality_matrix = [
        pd.read_csv(f, sep="\t", index_col=0).T   # rows=samples, cols=features
        for f in args.infile
    ]
    sample_ids = modality_matrix[0].index.tolist()
    n_samples  = len(sample_ids)
    disp(f"  {len(modality_matrix)} modalities loaded, {n_samples} samples")

    # ── 2. Load labels ────────────────────────────────────────────────────────
    # Expected format: index_col=0 (sample IDs), single column (0/1 integer)
    y = pd.read_table(args.label, header=None, index_col=0) \
          .values.reshape(-1).astype(int)
    disp(f"  Labels: {(y==1).sum()} positive (Cancer), "
         f"{(y==0).sum()} negative (Non-Cancer)")

    # ── 3. Resolve best clf per modality ──────────────────────────────────────
    modality_clfs, modality_names = _resolve_clf_per_modality(
        args, modality_matrix, performance=performance, perf_tsv=perf_tsv
    )

    # ── 4. Single-modality LOOCV ──────────────────────────────────────────────
    pred_dict  = {}
    y_true_ref = None

    for name, X, clf in zip(modality_names, modality_matrix, modality_clfs):
        y_pred_proba, y_true = _run_single_modality_loocv(
            name=name, X=X, clf=clf, y=y, top_n=args.size,
        )
        # All modalities must produce the same y_true ordering
        if y_true_ref is None:
            y_true_ref = y_true
        else:
            assert np.array_equal(y_true, y_true_ref), (
                f"y_true mismatch for modality '{name}'. "
                "Ensure all modality matrices have the same sample order."
            )
        pred_dict[name] = y_pred_proba

    # ── 5. Multimodal MESA LOOCV ──────────────────────────────────────────────
    y_pred_multi, y_true_multi = _run_multimodal_loocv(
        modality_matrix=modality_matrix,
        modality_clfs=modality_clfs,
        y=y,
        top_n=args.size,
    )
    assert np.array_equal(y_true_multi, y_true_ref), (
        "y_true mismatch between single-modality and multimodal LOOCV."
    )
    pred_dict["Multimodal"] = y_pred_multi

    # ── 6. Assemble output DataFrame ─────────────────────────────────────────
    pred_df = pd.DataFrame(pred_dict, index=sample_ids)
    pred_df.insert(0, "y_true", y_true_ref.astype(int))
    pred_df.index.name = "sample_id"

    disp(f"Prediction table: {pred_df.shape[0]} samples × "
         f"{pred_df.shape[1]-1} modalities + y_true")

    # ── 7. Save TSV ───────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "loocv_predictions.tsv")
    pred_df.to_csv(out_path, sep="\t")
    disp(f"Predictions saved to: {out_path}")

    # ── 8. Visualize ──────────────────────────────────────────────────────────
    try:
        from mesa_cv_plot import plot_mesa_loocv
        plot_mesa_loocv(pred_df, output_dir=args.output_dir)
    except ImportError:
        disp("Warning: mesa_cv_plot.py not found, skipping visualization.")
    except Exception as e:
        disp(f"Warning: visualization failed: {e}")

    return pred_df


# ══════════════════════════════════════════════════════════════════════════════
# Command-line entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    import ast
    import types

    parser = argparse.ArgumentParser(
        description=(
            "MESA LOOCV: run leave-one-out cross-validation for each single "
            "modality and the multimodal MESA model, then save predictions and "
            "generate visualizations.\n\n"
            "Example:\n"
            "  python mesa_loocv.py \\\n"
            "      DHS_meth.tsv CGI_meth.tsv Occupancy.tsv WPS.tsv \\\n"
            "      --modality 'DHS meth' 'CGI meth' 'Occupancy' 'WPS' \\\n"
            "      --label label_control_sALS.tsv \\\n"
            "      --perf  output/modality_performance.tsv \\\n"
            "      --clf   1 2 3 \\\n"
            "      --size  100 \\\n"
            "      -o      output/ \\\n"
            "      --cohort 'LOOCV on Cohort 1'"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "infile", nargs="+",
        help="Feature matrix TSV files, one per modality, in the same order "
             "as --modality names.",
    )
    parser.add_argument(
        "--modality", nargs="+", required=True,
        help="Modality names in the same order as infile. Must match the index "
             "of modality_performance.tsv exactly (including spaces).",
    )
    parser.add_argument(
        "--label", required=True,
        help="Label TSV file. index_col=0 (sample IDs), single column of 0/1.",
    )
    parser.add_argument(
        "--perf", default=None,
        help="Path to modality_performance.tsv from a previous --performance "
             "run. When provided, each modality uses its best classifier. "
             "When omitted, falls back to --clf.",
    )
    parser.add_argument(
        "--clf", nargs="+", type=int, default=[1, 2, 3],
        help="Classifier keys used in the original --performance run. "
             "Must be in the same order (1=RF, 2=LR, 3=SVC, 4=XGBoost). "
             "Default: 1 2 3",
    )
    parser.add_argument(
        "--size", type=int, default=100,
        help="Number of top features kept by Boruta. Default: 100.",
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="Output directory.",
    )
    parser.add_argument(
        "--cohort", default="LOOCV",
        help="Cohort label shown in figure titles. Default: 'LOOCV'.",
    )

    cli = parser.parse_args()

    # Validate
    for f in cli.infile:
        if not os.path.exists(f):
            sys.exit(f"ERROR: input file not found: {f}")
    if not os.path.exists(cli.label):
        sys.exit(f"ERROR: label file not found: {cli.label}")
    if cli.perf and not os.path.exists(cli.perf):
        sys.exit(f"ERROR: performance TSV not found: {cli.perf}")
    if len(cli.infile) != len(cli.modality):
        sys.exit(
            f"ERROR: {len(cli.infile)} input files but "
            f"{len(cli.modality)} modality names."
        )

    # Build args namespace matching what run_mesa_loocv() expects
    args = types.SimpleNamespace(
        infile     = cli.infile,
        modality   = cli.modality,
        label      = cli.label,
        clf        = cli.clf,
        size       = cli.size,
        output_dir = cli.output,
    )

    pred_df = run_mesa_loocv(args, performance=None, perf_tsv=cli.perf)

    # Re-run visualization with custom cohort label if specified
    if cli.cohort != "LOOCV":
        try:
            from mesa_cv_plot import plot_mesa_loocv
            plot_mesa_loocv(
                pred_df,
                output_dir   = cli.output,
                cohort_label = cli.cohort,
            )
        except Exception as e:
            disp(f"Visualization warning: {e}")