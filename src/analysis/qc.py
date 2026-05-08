"""
QC analysis — three independent steps driven by cftk_init.json.
Step 1: methylation beta-value distribution  (requires cpg_matrix.tsv)
Step 2: fragment length distribution          (requires BAM files)
Step 3: dinucleotide frequency               (requires BAM files + reference)
"""

import os
import glob
import subprocess
import sys
import numpy as np
import pandas as pd
from util import disp


def run_qc(args):
    """
    Dispatcher: route to the correct QC computation based on args.step.
    Computation results are saved to args.output_dir; visualization is
    handled separately by visualization.visualization.plot_qc().
    """
    step = args.step
    os.makedirs(args.output_dir, exist_ok=True)

    if step == 1:
        return _run_meth_distribution(args)
    elif step == 2:
        return _run_fragment_length(args)
    elif step == 3:
        return _run_dinucleotide(args)
    else:
        sys.exit(f"[qc] Invalid step: {step}. Valid: 1, 2, 3.")


# ── Step 1: Methylation distribution ─────────────────────────────────────────

def _run_meth_distribution(args):
    """
    Load cpg_matrix.tsv and return path for visualization.
    No additional computation needed — plot_qc reads the matrix directly.
    """
    matrix = os.path.join(args.matrices_dir, "cpg_matrix.tsv")
    if not os.path.exists(matrix):
        sys.exit(
            f"[qc step1] cpg_matrix.tsv not found: {matrix}\n"
            "Run 'cftk merge --modality cpg' first."
        )
    disp(f"[qc step1] cpg matrix: {matrix}")
    return {"matrix": matrix}


# ── Step 2: Fragment length distribution ──────────────────────────────────────

def _run_fragment_length(args):
    """
    Run bamPEFragmentSize for each BAM and save raw fragment length CSVs.
    Returns dict with prefix path for visualization.
    """
    bams    = args.infile
    out_dir = args.output_dir
    cores   = getattr(args, "cores", 1)
    sub_dir = os.path.join(out_dir, "2_fragment_length")
    os.makedirs(sub_dir, exist_ok=True)
    prefix  = os.path.join(sub_dir, "fragment_length")

    if not bams:
        sys.exit("[qc step2] No BAM files provided.")

    for bam in bams:
        if not os.path.exists(bam):
            disp(f"[qc step2] WARNING: BAM not found: {bam}")
            continue
        stem = os.path.splitext(os.path.basename(bam))[0]
        raw_csv  = f"{prefix}.{stem}.raw.csv"
        hist_png = f"{prefix}.{stem}.hist.png"
        cmd = (
            f"bamPEFragmentSize "
            f"--outRawFragmentLengths {raw_csv} "
            f"-hist {hist_png} "
            f"-p {cores} -b {bam}"
        )
        disp(f"[qc step2] {stem}")
        ret = subprocess.run(cmd, shell=True)
        if ret.returncode != 0:
            disp(f"[qc step2] WARNING: bamPEFragmentSize failed for {bam}")

    raw_files = sorted(glob.glob(f"{prefix}.*.raw.csv"))
    disp(f"[qc step2] {len(raw_files)} raw fragment length files written.")
    return {"prefix": prefix, "raw_files": raw_files}


# ── Step 3: Dinucleotide frequency ───────────────────────────────────────────

def _run_dinucleotide(args):
    """
    Extract dinucleotide frequencies around fragment centres using bedtools nuc.
    Produces per-dinucleotide count files; visualization reads these directly.
    """
    bams      = args.infile
    ref_fa    = args.ref_fa
    frag_len  = getattr(args, "fragment",  167)
    clip_r1   = getattr(args, "clip_r1",   0)
    clip_r2   = getattr(args, "clip_r2",   0)
    cores     = getattr(args, "cores",     1)
    out_dir   = args.output_dir
    sub_dir   = os.path.join(out_dir, "3_dinucleotide_freq")
    os.makedirs(sub_dir, exist_ok=True)
    prefix    = os.path.join(sub_dir, "dinucleotide")

    if not ref_fa or not os.path.exists(ref_fa):
        sys.exit(
            "[qc step3] reference_data.genome_fa is required for dinucleotide analysis."
        )

    all_frag_file = f"{prefix}.all_fragment"
    if os.path.exists(all_frag_file):
        os.remove(all_frag_file)

    # build fragment BED per BAM
    for bam in bams:
        if not os.path.exists(bam):
            disp(f"[qc step3] WARNING: BAM not found: {bam}")
            continue
        sample = os.path.splitext(os.path.basename(bam))[0]
        cmd = (
            f"bedtools bamtobed -bedpe -mate1 -i {bam} 2>/dev/null | "
            f"awk -v OFS='\\t' -v sample={sample} "
            f"-v cr1={clip_r1} -v cr2={clip_r2} '{{"
            f"if ($9 == \"+\") {{"
            f"  start = ($2-cr1 < $5) ? $2-cr1 : $5;"
            f"  end   = ($3 > $6+cr2) ? $3 : $6+cr2;"
            f"  print $1, start, end, sample;"
            f"}} else {{"
            f"  start = ($2 < $5-cr1) ? $2 : $5-cr1;"
            f"  end   = ($3+cr2 > $6) ? $3+cr2 : $6;"
            f"  print $1, start, end, sample;"
            f"}}}}' | "
            f"awk -v OFS='\\t' '$3-$2=={frag_len} {{print}}' "
            f">> {all_frag_file} || exit 1"
        )
        disp(f"[qc step3] bamtobed: {sample}")
        subprocess.run(cmd, shell=True, check=True)

    if not os.path.exists(all_frag_file) or os.path.getsize(all_frag_file) == 0:
        sys.exit(f"[qc step3] No fragments extracted to {all_frag_file}.")

    # build 2bp sliding window BED
    window_file = f"{all_frag_file}.window2bp"
    if os.path.exists(window_file):
        os.remove(window_file)

    frags   = pd.read_table(all_frag_file, header=None)
    half_w  = (250 - frag_len) // 2
    with open(window_file, "ab") as fh:
        for _, row in frags.iterrows():
            # fix: always generate exactly 250 positions from start_w
            # avoids inhomogeneous shape when row[2]-row[1] != frag_len
            start_w = int(row[1]) - half_w
            pos     = np.arange(250)
            arr     = np.column_stack([
                [str(row[0])] * 250,
                (start_w + pos).astype(str),
                (start_w + pos + 2).astype(str),
                pos.astype(str),
                [str(row[3])] * 250,
            ])
            np.savetxt(fh, arr, delimiter="\t", fmt="%s")

    # run bedtools nuc per dinucleotide pattern
    dinu_list = ["AA", "AT", "TA", "TT", "GG", "GC", "CG", "CC"]

    def _nuc(pattern):
        out = f"{prefix}.all_fragment_{pattern}.txt"
        cmd = (
            f"bedtools nuc -pattern {pattern} -C -fi {ref_fa} "
            f"-bed {window_file} > {out} || exit 1"
        )
        subprocess.run(cmd, shell=True, check=True)

    from joblib import Parallel, delayed
    Parallel(n_jobs=cores)(delayed(_nuc)(p) for p in dinu_list)

    disp(f"[qc step3] dinucleotide files written to {out_dir}")
    return {"prefix": prefix, "dinu_list": dinu_list}
