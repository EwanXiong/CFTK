"""Compute average feature values per genomic region."""

import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed


def _mean_region(matrix_chrom, idx_chrom, start, end):
    mask = (idx_chrom["pos"] > start) & (idx_chrom["pos"] <= end)
    sub  = matrix_chrom.iloc[mask.values]
    return sub.mean(axis=0) if len(sub) else pd.Series(dtype=float)


def run_region_average(matrix_path, region_bed, output_path, cores=-1):
    """
    Average matrix values (features × samples) over BED regions.
    Feature index format: chr_position.
    """
    df      = pd.read_csv(matrix_path, sep="\t", header=0, index_col=0)
    regions = pd.read_csv(region_bed, sep="\t", header=None,
                          names=["chrom", "start", "end"])

    idx = pd.DataFrame(
        [r.split("_")[:2] for r in df.index], columns=["chr", "pos"]
    )
    keep = idx["pos"].str.isnumeric().values
    df, idx = df.iloc[keep], idx.iloc[keep]
    idx["pos"] = idx["pos"].astype(int)

    out = pd.DataFrame(columns=df.columns)
    for chrom in regions["chrom"].unique():
        mc  = df.iloc[(idx["chr"] == chrom).values]
        ic  = idx.iloc[(idx["chr"] == chrom).values]
        rc  = regions[regions["chrom"] == chrom]

        rows = Parallel(n_jobs=cores, backend="multiprocessing")(
            delayed(_mean_region)(mc, ic, row.start, row.end)
            for _, row in rc.iterrows()
        )
        idx_labels = rc.apply(
            lambda r: f"{chrom}:{r.start}-{r.end}", axis=1
        )
        chunk = pd.DataFrame(rows, index=idx_labels)
        out   = pd.concat([out, chunk])

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    out.to_csv(output_path, sep="\t")
    print(f"[region_avg] saved → {output_path}")
    return out
