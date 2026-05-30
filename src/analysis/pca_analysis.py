"""PCA computation — saves coordinates and variance to txt files."""

import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


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


def _prepare_matrix(df, group_labels_raw, max_nan_frac=0.5):
    groups = _resolve_groups(df, group_labels_raw)
    group_names = list(groups.keys())
    cols, labels = [], []
    # groups[name] is a resolved list of exact column names
    for name, col_list in groups.items():
        cols.extend(col_list)
        labels.extend([name] * len(col_list))
    sub = df[cols].copy()
    sub = sub.loc[~sub.isna().all(axis=1)]
    sub = sub.loc[sub.isna().mean(axis=1) <= max_nan_frac]
    sub = sub.loc[sub.var(axis=1, skipna=True) > 0]
    return sub.T, labels, group_names


def run_pca(args):
    """Compute PCA and save pca_coordinates.txt + pca_variance.txt."""
    modality = args.modality
    out_dir = os.path.join(args.output_dir, modality)
    os.makedirs(out_dir, exist_ok=True)

    df = _load_matrix(args.infile)
    data, labels, group_names = _prepare_matrix(df, args.group_labels)

    X = SimpleImputer(strategy="mean").fit_transform(data.values)
    X = StandardScaler().fit_transform(X)

    pca = PCA(n_components=min(10, X.shape[1]))
    pcs = pca.fit_transform(X)
    var = pca.explained_variance_ratio_ * 100

    # save coordinates
    coord_df = pd.DataFrame(
        pcs,
        index=data.index,
        columns=[f"PC{i+1}" for i in range(pcs.shape[1])],
    )
    coord_df.insert(0, "group", labels)
    coord_df.to_csv(os.path.join(out_dir, "pca_coordinates.txt"), sep="\t")

    # save variance explained
    var_df = pd.DataFrame(
        {"PC": [f"PC{i+1}" for i in range(len(var))],
         "variance_explained_pct": var.round(4)}
    )
    var_df.to_csv(os.path.join(out_dir, "pca_variance.txt"), sep="\t", index=False)

    print(f"[pca] saved to {out_dir}")
    return coord_df, var_df, group_names
