"""End-motif analysis — wraps finaletoolkit end-motifs CLI."""

import os
import subprocess
import sys


def _run(cmd):
    print(f"[end_motif] {cmd[:100]}", file=sys.stderr)
    ret = subprocess.run(cmd, shell=True)
    if ret.returncode != 0:
        sys.exit("[end_motif] ERROR: command failed.")


def run_end_motif(args):
    """Run finaletoolkit end-motifs for each input BAM."""
    out_dir = args.end_motif_out
    os.makedirs(out_dir, exist_ok=True)

    kmer  = getattr(args, "kmer",     4)
    mapq  = getattr(args, "mapq",    30)
    minf  = getattr(args, "min_frag", 100)
    maxf  = getattr(args, "max_frag", 220)
    extra = getattr(args, "em_extra", "")

    results = []
    for bam in args.infile:
        sample  = os.path.splitext(os.path.basename(bam))[0].replace(".markdup", "")
        out_tsv = os.path.join(out_dir, f"{sample}_{kmer}mer.tsv")

        cmd = (
            f"finaletoolkit end-motifs "
            f"-k {kmer} -min {minf} -max {maxf} "
            f"-o {out_tsv} -q {mapq} -v {extra} "
            f"{bam} {args.genome2bit}"
        )
        _run(cmd)
        results.append(out_tsv)
        print(f"[end_motif] saved → {out_tsv}")

    return results
