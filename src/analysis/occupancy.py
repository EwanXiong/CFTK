"""Nucleosome occupancy via DANPOS3. Auto-merges matrix when >1 sample."""

import os
import sys
import glob
import pandas as pd
from util import disp, run_command


def run_occupancy(args):
    """
    DANPOS3 dpos → wigToBigWig → bigWigAverageOverBed per sample.
    Output dir  : 4_fragmentomics/occupancy/
    Per-sample  : {out_dir}/{stem}.occupancy.tsv
    Matrix      : {out_dir}/occupancy_matrix.tsv  (auto when >1 sample)
    """
    out_dir    = args.occ_out
    chrom_sz   = args.chrom_sizes
    region_bed = args.region
    tool       = getattr(args, "danpos",       "danpos.py")
    extra      = getattr(args, "danpos_extra", "--paired 1 -u 0 -c 1000000")
    os.makedirs(out_dir, exist_ok=True)

    results = []   # list of (occ_path, col_name)
    for bam in args.infile:
        if not os.path.exists(bam):
            disp(f"[occupancy] WARNING: BAM not found: {bam}")
            continue

        stem      = os.path.splitext(os.path.basename(bam))[0]
        name      = stem.replace(".markdup", "")  # clean sample name
        danpos_tmp = os.path.join(out_dir, "danpos_intermediate")
        os.makedirs(danpos_tmp, exist_ok=True)
        wig  = os.path.join(danpos_tmp, "pooled", f"{stem}.Fnor.smooth.wig")
        bw   = os.path.join(out_dir,   f"{name}.bw")
        occ  = os.path.join(out_dir,   f"{name}.occupancy.tsv")

        cmd = (
            f"python {tool} dpos {bam} {extra} -o {danpos_tmp} && "
            f"wigToBigWig -clip {wig} {chrom_sz} {bw} && "
            f"bigWigAverageOverBed {bw} {region_bed} {occ} || exit 1"
        )
        run_command(cmd, label=f"occupancy [{name}]")
        col_name = name
        results.append((occ, col_name))
        disp(f"[occupancy] saved → {occ}")

    # auto-merge
    if len(results) > 1:
        _merge_occupancy(results, out_dir)
    elif len(results) == 1:
        disp("[occupancy] Single sample — skipping matrix merge.")
    return [r[0] for r in results]


def _merge_occupancy(occ_col_pairs, out_dir):
    """
    Merge per-sample occupancy TSVs into occupancy_matrix.tsv.
    bigWigAverageOverBed output (no header):
      col0=region_name  col1=size  col2=covered  col3=sum  col4=mean0  col5=mean
    """
    out_path = os.path.join(out_dir, "occupancy_matrix.tsv")
    disp(f"[occupancy] merging {len(occ_col_pairs)} files → {out_path}")
    frames = {}
    for fp, col_name in occ_col_pairs:
        # no header in bigWigAverageOverBed output
        df = pd.read_csv(fp, sep="\t", header=None,
                         usecols=[0, 5], names=["region", col_name])
        frames[col_name] = df.set_index("region")[col_name]
    matrix = pd.DataFrame(frames)
    matrix.to_csv(out_path, sep="\t")
    disp(f"[occupancy] {matrix.shape[0]} regions × {matrix.shape[1]} samples → {out_path}")
    return out_path
