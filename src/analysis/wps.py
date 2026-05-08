"""WPS (Windowed Protection Score) per genomic region. Auto-merges when >1 sample."""

import os
import sys
import time
import numpy as np
import pandas as pd
import pysam
from bx.intervals.intersection import Intersecter, Interval
from joblib import Parallel, delayed


def _disp(msg):
    import sys
    print(f"@{time.asctime()}\t{msg}", file=sys.stderr)


def _wps_chrom(bam_path, region_df, chrom, window_half, step, min_len, max_len):
    """Compute WPS for all regions on one chromosome."""
    tree = Intersecter()
    chrom_regions = region_df[region_df.iloc[:, 0] == chrom].iloc[:, 1:3].astype(int)
    bam = pysam.AlignmentFile(bam_path, "rb")
    for ra, rb in chrom_regions.values:
        for read in bam.fetch(chrom, ra, rb, multiple_iterators=True):
            tree.add_interval(Interval(read.reference_start, read.reference_end))
    bam.close()

    records = []
    for ra, rb in chrom_regions.values:
        wps_vals = []
        for pos in range(ra, rb + 1, step):
            wa, wb = pos - window_half, pos + window_half
            end_cnt = com_cnt = 0
            for rd in tree.find(wa, wb):
                span = rd.end - rd.start + 1
                if min_len <= span <= max_len:
                    if rd.start > wa or rd.end < wb:
                        end_cnt += 1
                    else:
                        com_cnt += 1
            wps_vals.append(com_cnt - end_cnt)
        arr = np.array(wps_vals)
        records.append((chrom, ra, rb, arr, float(arr.mean())))

    return pd.DataFrame(records, columns=["chr", "start", "end", "WPS", "mean_WPS"])


def run_wps(args):
    """
    Compute WPS for each BAM.
    Output dir  : 4_fragmentomics/wps/
    Per-sample  : {out_dir}/{sample}.wps.tsv
    Matrix      : {out_dir}/wps_matrix.tsv  (auto when >1 sample)
    """
    out_dir     = args.wps_out
    region_file = getattr(args, "region", None)
    if not region_file:
        sys.exit("[wps] ERROR: reference_data.tss_pas_bed is required.")

    os.makedirs(out_dir, exist_ok=True)
    regions    = pd.read_csv(region_file, sep="\t", header=None)
    chrom_list = [f"chr{i}" for i in list(range(1, 23)) + ["X", "Y"]]
    chrom_list = [c for c in chrom_list if c in regions.iloc[:, 0].unique()]

    window_half = getattr(args, "wps_window", 120) // 2
    step        = getattr(args, "wps_step",   10)
    min_len     = getattr(args, "min_frag",   100)
    max_len     = getattr(args, "max_frag",   220)
    cores       = getattr(args, "cores",      1)

    results = []   # list of (tsv_path, col_name)
    for bam in args.infile:
        if not os.path.exists(bam):
            _disp(f"[wps] WARNING: BAM not found: {bam}")
            continue

        stem     = os.path.splitext(os.path.basename(bam))[0]
        name     = stem.replace(".markdup", "")  # clean sample name
        out_tsv  = os.path.join(out_dir, f"{name}.wps.tsv")
        col_name = name

        _disp(f"WPS: {name}")
        chrom_results = Parallel(n_jobs=cores, backend="multiprocessing")(
            delayed(_wps_chrom)(bam, regions, chrom,
                                window_half, step, min_len, max_len)
            for chrom in chrom_list
        )
        pd.concat(chrom_results).to_csv(out_tsv, sep="\t", index=False)
        results.append((out_tsv, col_name))
        _disp(f"[wps] saved → {out_tsv}")

    # auto-merge
    if len(results) > 1:
        _merge_wps(results, out_dir)
    elif len(results) == 1:
        _disp("[wps] Single sample — skipping matrix merge.")
    return [r[0] for r in results]


def _merge_wps(wps_col_pairs, out_dir):
    """
    Merge per-sample WPS TSVs into wps_matrix.tsv.
    Index = chr:start-end (region identifier matching tss_pas_bed regions).
    Column names = {group}_{sample_name} for group prefix matching.
    """
    out_path = os.path.join(out_dir, "wps_matrix.tsv")
    _disp(f"[wps] merging {len(wps_col_pairs)} files → {out_path}")
    frames = {}
    for fp, col_name in wps_col_pairs:
        df = pd.read_csv(fp, sep="\t")
        df.index = (df["chr"] + ":" + df["start"].astype(str)
                    + "-" + df["end"].astype(str))
        frames[col_name] = df["mean_WPS"]
    matrix = pd.DataFrame(frames)
    matrix.to_csv(out_path, sep="\t")
    _disp(f"[wps] {matrix.shape[0]} regions × {matrix.shape[1]} samples → {out_path}")
    return out_path
