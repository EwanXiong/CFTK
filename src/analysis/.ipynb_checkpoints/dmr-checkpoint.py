"""DMR analysis — runs metilene then annotates via R script."""

import os
import subprocess
import sys


def _run(cmd, label=""):
    print(f"[dmr] {label or cmd[:80]}", file=sys.stderr)
    ret = subprocess.run(cmd, shell=True)
    if ret.returncode != 0:
        sys.exit(f"[dmr] ERROR: failed — {label}")


def run_dmr(args):
    """
    1. Run metilene to call DMRs.
    2. Annotate DMRs via dmr_annotation.r (annotatr hg38).
    Outputs: dmr_raw.bed, dmr_annotated.bed
    """
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    raw_bed = os.path.join(out_dir, "dmr_raw.bed")
    ann_bed = os.path.join(out_dir, "dmr_annotated.bed")

    # resolve tool and params from args (set by _cmd_dmr in cftk.py)
    tool    = getattr(args, "metilene_tool", "metilene")
    threads = getattr(args, "threads", 20)
    extra   = getattr(args, "dmr_extra_args", "")

    if tool == "metilene":
        cmd = (
            f"metilene -a {args.group_a} -b {args.group_b} "
            f"-t {threads} {extra} {args.metilene_input} > {raw_bed}"
        )
    else:
        sys.exit(f"[dmr] unsupported tool: {tool}")

    _run(cmd, "metilene")

    # R annotation
    r_script = os.path.join(os.path.dirname(__file__), "dmr_annotation.r")
    _run(f"Rscript {r_script} {raw_bed} {ann_bed}", "dmr_annotation.r")

    print(f"[dmr] raw   → {raw_bed}")
    print(f"[dmr] annot → {ann_bed}")
    return raw_bed, ann_bed
