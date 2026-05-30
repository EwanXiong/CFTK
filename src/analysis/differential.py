"""Differential analysis — Mann-Whitney U test with BH correction."""

import os
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from joblib import Parallel, delayed


def _load_matrix(filepath):
    sep = "," if filepath.endswith(".csv") else "\t"
    return pd.read_csv(filepath, sep=sep, header=0, index_col=0)


def _resolve_groups(df, group_labels_raw):
    """
    Resolve group → list-of-column-names.
    Accepts:
      dict  {"GroupA": ["col1","col2"], "GroupB": ["col3"]}  ← exact lists
      dict  {"GroupA": "prefix_"}                             ← startswith prefix
      list  ["GroupA","prefA","GroupB","prefB"]               ← legacy pairs
    """
    if isinstance(group_labels_raw, dict):
        resolved = {}
        for name, spec in group_labels_raw.items():
            if isinstance(spec, (list, tuple)):
                # exact column name list — validate
                missing = [s for s in spec if s not in df.columns]
                if missing:
                    import sys
                    sys.exit(
                        f"[diff] ERROR: columns not found in matrix for group '{name}': "
                        f"{missing}. Available: {df.columns.tolist()[:10]}"
                    )
                resolved[name] = list(spec)
            elif isinstance(spec, str):
                # prefix matching (legacy/fallback)
                matched = [c for c in df.columns if c.startswith(spec)]
                if not matched:
                    import sys
                    sys.exit(
                        f"[diff] ERROR: no columns start with prefix '{spec}' "
                        f"for group '{name}'. Available: {df.columns.tolist()[:10]}"
                    )
                resolved[name] = matched
        return resolved
    # legacy list form: ['GroupA','prefA','GroupB','prefB']
    it = iter(group_labels_raw)
    pairs = {name: pref for name, pref in zip(it, it)}
    return _resolve_groups(df, pairs)


def _bh_correction(pvalues):
    """Benjamini-Hochberg FDR correction."""
    n = len(pvalues)
    order = np.argsort(pvalues)
    ranked_p = pvalues[order]
    bh = ranked_p * n / (np.arange(n) + 1)
    # enforce monotonicity
    for i in range(n - 2, -1, -1):
        bh[i] = min(bh[i], bh[i + 1])
    q = np.empty(n)
    q[order] = np.minimum(bh, 1.0)
    return q


def _mwu_chunk(A_chunk, B_chunk):
    return mannwhitneyu(A_chunk, B_chunk, nan_policy="omit",
                        alternative="two-sided")[1]


def run_differential(args, chunk_size=500):
    """Run per-feature MWU test and save differential_result.tsv."""
    import warnings
    # suppress small-sample and fastcluster warnings — not errors
    warnings.filterwarnings("ignore", message=".*too small.*")
    warnings.filterwarnings("ignore", category=UserWarning)
    try:
        from scipy.stats import SmallSampleWarning
        warnings.filterwarnings("ignore", category=SmallSampleWarning)
    except ImportError:
        pass

    modality = args.modality
    out_dir = os.path.join(args.output_dir, modality)
    os.makedirs(out_dir, exist_ok=True)

    df = _load_matrix(args.infile)
    groups = _resolve_groups(df, args.group_labels)
    names = list(groups.keys())

    # groups[name] is already a resolved list of column names
    cols_a = groups[names[0]]
    cols_b = groups[names[1]]
    A = df[cols_a].T
    B = df[cols_b].T

    n_feat = df.shape[0]
    n_chunks = int(np.ceil(n_feat / chunk_size))

    pvals = np.hstack(
        Parallel(n_jobs=getattr(args, "cores", -1), backend="multiprocessing")(
            delayed(_mwu_chunk)(
                A.iloc[:, i * chunk_size:(i + 1) * chunk_size].values,
                B.iloc[:, i * chunk_size:(i + 1) * chunk_size].values,
            )
            for i in range(n_chunks)
        )
    )

    qvals = _bh_correction(pvals)
    mean_a = df[cols_a].mean(axis=1).values
    mean_b = df[cols_b].mean(axis=1).values
    meandiff = mean_b - mean_a

    result = pd.DataFrame({
        "feature":  df.index,
        "MWU_pvalue": pvals,
        "qvalue":   qvals,
        "meandiff": meandiff,
        f"mean_{names[0]}": mean_a,
        f"mean_{names[1]}": mean_b,
    }).set_index("feature")

    out_path = os.path.join(out_dir, "differential_result.tsv")
    result.to_csv(out_path, sep="\t")
    print(f"[diff] saved to {out_path}")
    return result
