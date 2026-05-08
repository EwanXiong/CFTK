"""DELFI analysis — wraps finaletoolkit delfi CLI."""

import os
import subprocess
import sys


def _run(cmd):
    print(f"[delfi] {cmd[:100]}", file=sys.stderr)
    ret = subprocess.run(cmd, shell=True)
    if ret.returncode != 0:
        sys.exit("[delfi] ERROR: command failed.")


def run_delfi(args):
    """Run finaletoolkit delfi for each input BAM."""
    out_dir  = args.delfi_out
    os.makedirs(out_dir, exist_ok=True)

    mapq   = getattr(args, "delfi_mapq",   getattr(args, "mapq", 30))
    window = getattr(args, "delfi_window", getattr(args, "window", 20))
    extra  = getattr(args, "delfi_extra",  "")

    results = []
    for bam in args.infile:
        sample  = os.path.splitext(os.path.basename(bam))[0].replace(".markdup", "")
        out_tsv = os.path.join(out_dir, f"{sample}_delfi.tsv")

        cmd = (
            f"finaletoolkit delfi "
            f"-b {args.blacklist} -g {args.gap} "
            f"-o {out_tsv} -R -q {mapq} -w {window} -M -v {extra} "
            f"{bam} {args.chrom_sizes} {args.genome2bit} {args.bins}"
        )
        _run(cmd)
        results.append(out_tsv)
        print(f"[delfi] saved → {out_tsv}")

    return results
