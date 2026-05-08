"""Cleavage profile analysis — wraps finaletoolkit cleavage-profile CLI."""

import os
import subprocess
import sys


def _run(cmd):
    print(f"[cleavage] {cmd[:100]}", file=sys.stderr)
    ret = subprocess.run(cmd, shell=True)
    if ret.returncode != 0:
        sys.exit("[cleavage] ERROR: command failed.")


def run_cleavage(args):
    """Run finaletoolkit cleavage-profile for each input BAM."""
    out_dir = args.cleavage_out
    os.makedirs(out_dir, exist_ok=True)

    if not args.bed:
        sys.exit("[cleavage] ERROR: reference_data.ctcf_bed is required.")

    minf   = getattr(args, "min_frag", 100)
    maxf   = getattr(args, "max_frag", 220)
    mapq   = getattr(args, "cl_mapq",  getattr(args, "mapq", 30))
    window = getattr(args, "window",   20)
    extra  = getattr(args, "cl_extra", "")

    results = []
    for bam in args.infile:
        sample = os.path.splitext(os.path.basename(bam))[0].replace(".markdup", "")
        out_bw = os.path.join(out_dir, f"{sample}_cleavage.bw")

        cmd = (
            f"finaletoolkit cleavage-profile "
            f"-o {out_bw} -min {minf} -max {maxf} "
            f"-q {mapq} -w {window} -v {extra} "
            f"{bam} {args.bed} {args.chrom_sizes}"
        )
        _run(cmd)
        results.append(out_bw)
        print(f"[cleavage] saved → {out_bw}")

    return results
