"""
process.py — per-sample raw data processing (Part 1).
Steps: 1=trimming  2=alignment  3=markdup  4=methylation
All samples output to: results/1_process/{1_trimming, 2_alignment, 3_markdup, 4_methylation}/
"""

import os
import sys
from util import disp, run_command
from init import load_config, get_all_samples, get_work_paths, get_bam

STEPS = {
    1: "Adapter trimming",
    2: "Bisulfite alignment",
    3: "Mark duplicates",
    4: "CpG methylation calling",
}

# supported tool sets per step (for validation and future extension)
SUPPORTED_TOOLS = {
    1: {"trim_galore", "fastp"},
    2: {"bwameth", "bismark"},
    3: {"sambamba", "picard", "samblaster"},
    4: {"methyldackel", "bismark_extractor"},
}


# ── Step implementations ──────────────────────────────────────────────────────

def _step1_trim(sample, step_cfg, ref_data, paths):
    """Adapter trimming — FASTQ only; BAM input samples are skipped."""
    if sample["input_type"] != "fastq":
        disp(f"  [trim] skip BAM input: {sample['name']}")
        return None, None

    tool   = step_cfg["tool"]
    params = step_cfg["params"]
    cores  = params.get("cores", 20)
    extra  = params.get("extra_args", "")
    name   = sample["name"]
    out    = paths["trimming"]
    os.makedirs(out, exist_ok=True)

    if tool == "trim_galore":
        cmd = (
            f"trim_galore --paired --2colour 20 --cores {cores} "
            f"-o {out} --basename {name} {extra} "
            f"{sample['r1']} {sample['r2']} || exit 1"
        )
    elif tool == "fastp":
        ext    = "fq.gz" if sample["r1"].endswith(".gz") else "fq"
        r1_out = os.path.join(out, f"{name}_val_1.{ext}")
        r2_out = os.path.join(out, f"{name}_val_2.{ext}")
        cmd = (
            f"fastp -i {sample['r1']} -I {sample['r2']} "
            f"-o {r1_out} -O {r2_out} "
            f"-w {cores} {extra} || exit 1"
        )
    else:
        sys.exit(f"[step1] unsupported tool: {tool}. Supported: {SUPPORTED_TOOLS[1]}")

    run_command(cmd, label=f"trim [{name}]")

    ext    = "fq.gz" if sample["r1"].endswith(".gz") else "fq"
    r1_out = os.path.join(out, f"{name}_val_1.{ext}")
    r2_out = os.path.join(out, f"{name}_val_2.{ext}")
    return r1_out, r2_out


def _step2_align(sample, r1, r2, step_cfg, ref_data, paths):
    """Bisulfite alignment → sorted+filtered BAM."""
    tool   = step_cfg["tool"]
    params = step_cfg["params"]
    cores  = params.get("cores", 20)
    extra  = params.get("extra_args", "")
    ref    = ref_data["genome_fa"]
    name   = sample["name"]
    out    = paths["alignment"]
    os.makedirs(out, exist_ok=True)
    bam    = os.path.join(out, f"{name}.bam")

    if tool == "bwameth":
        cmd = (
            f"bwameth.py --reference {ref} -t {cores} {extra} {r1} {r2} | "
            f"sambamba view -t {cores} "
            f"-F 'not secondary_alignment and not failed_quality_control "
            f"and not supplementary and proper_pair and mapping_quality > 0' "
            f"-f bam -S -l 0 /dev/stdin | "
            f"sambamba sort -t {cores} -o {bam} /dev/stdin || exit 1; "
            f"samtools index -@ {cores} {bam} || exit 1"
        )
    elif tool == "bismark":
        cmd = (
            f"bismark --genome {os.path.dirname(ref)} -1 {r1} -2 {r2} "
            f"-p {cores} -o {out} {extra} || exit 1"
        )
    else:
        sys.exit(f"[step2] unsupported tool: {tool}. Supported: {SUPPORTED_TOOLS[2]}")

    run_command(cmd, label=f"align [{name}]")
    return bam


def _step3_markdup(sample, bam_in, step_cfg, ref_data, paths):
    """Duplicate marking."""
    tool   = step_cfg["tool"]
    params = step_cfg["params"]
    cores  = params.get("cores", 20)
    extra  = params.get("extra_args", "")
    ref    = ref_data["genome_fa"]
    name   = sample["name"]
    out    = paths["markdup"]
    os.makedirs(out, exist_ok=True)
    bam_out = os.path.join(out, f"{name}.markdup.bam")

    if tool == "sambamba":
        cmd = (
            f"sambamba markdup -t {cores} {extra} {bam_in} {bam_out} || exit 1; "
            f"samtools index -@ {cores} {bam_out} || exit 1"
        )
    elif tool == "picard":
        metrics = bam_out.replace(".bam", "_metrics.txt")
        cmd = (
            f"picard MarkDuplicates I={bam_in} O={bam_out} R={ref} "
            f"M={metrics} SORTING_COLLECTION_SIZE_RATIO=0.15 "
            f"ASSUME_SORT_ORDER=coordinate OPTICAL_DUPLICATE_PIXEL_DISTANCE=2500 "
            f"MAX_RECORDS_IN_RAM=1000 {extra} || exit 1; "
            f"samtools index -@ {cores} {bam_out} || exit 1"
        )
    elif tool == "samblaster":
        cmd = (
            f"samblaster {extra} --addMateTags "
            f"< {bam_in} > {bam_out} || exit 1; "
            f"samtools index -@ {cores} {bam_out} || exit 1"
        )
    else:
        sys.exit(f"[step3] unsupported tool: {tool}. Supported: {SUPPORTED_TOOLS[3]}")

    run_command(cmd, label=f"markdup [{name}]")
    return bam_out


def _step4_methylation(sample, bam_in, step_cfg, ref_data, paths):
    """CpG methylation calling."""
    tool   = step_cfg["tool"]
    params = step_cfg["params"]
    cores  = params.get("cores", 20)
    depth  = params.get("min_depth", 10)
    extra  = params.get("extra_args", "")
    ref    = ref_data["genome_fa"]
    name   = sample["name"]
    out    = paths["methylation"]
    os.makedirs(out, exist_ok=True)
    prefix = os.path.join(out, name)

    if tool == "methyldackel":
        mbias = f"{prefix}_mbias_OT_OB.temp"
        cmd = (
            f"MethylDackel mbias -@ {cores} {ref} {bam_in} {prefix} "
            f"&> {mbias} || exit 1; "
            f"MethylDackel extract --minDepth {depth} --maxVariantFrac 0.25 "
            f"-@ {cores} "
            f"--OT $(grep -oP '(?<=--OT )[^ ]+' {mbias}) "
            f"--OB $(grep -oP '(?<=--OB )[^ ]+' {mbias}) "
            f"-o {prefix} {extra} {ref} {bam_in} || exit 1"
        )
    elif tool == "bismark_extractor":
        cmd = (
            f"bismark_methylation_extractor --paired-end --gzip "
            f"--CpG_only --cytosine_report --genome_folder {os.path.dirname(ref)} "
            f"-o {out} {extra} {bam_in} || exit 1"
        )
    else:
        sys.exit(f"[step4] unsupported tool: {tool}. Supported: {SUPPORTED_TOOLS[4]}")

    run_command(cmd, label=f"methylation [{name}]")
    return f"{prefix}_CpG.bedGraph"


# ── Main ──────────────────────────────────────────────────────────────────────

def process(args, config_path="./cftk_init.json"):
    """
    Run processing pipeline steps 1-4 for all samples.
    BAM input samples skip step 1 (trimming) and step 2 (alignment) automatically.
    """
    cfg     = load_config(config_path)
    paths   = get_work_paths(cfg)
    samples = get_all_samples(cfg)
    steps   = sorted(set(args.step))
    ref     = cfg["reference_data"]
    proc    = cfg["process"]

    # validate requested steps
    invalid = [s for s in steps if s not in STEPS]
    if invalid:
        sys.exit(f"[process] Invalid steps: {invalid}. Valid: {list(STEPS.keys())}")

    disp(f"Samples : {len(samples)} total")
    disp(f"Steps   : {[STEPS[s] for s in steps]}")

    for sample in samples:
        name = sample["name"]
        grp  = sample["group"]
        itype = sample["input_type"]
        disp(f"\n── {name} ({grp}, {itype}) ──")

        current_bam = sample.get("bam") if itype == "bam" else None
        r1_trim = r2_trim = None

        if 1 in steps:
            r1_trim, r2_trim = _step1_trim(
                sample, proc["step1_trimming"], ref, paths
            )

        r1 = r1_trim or sample.get("r1")
        r2 = r2_trim or sample.get("r2")

        if 2 in steps:
            if itype == "bam":
                disp(f"  [align] skip — BAM input")
            else:
                current_bam = _step2_align(
                    sample, r1, r2, proc["step2_alignment"], ref, paths
                )

        # resolve BAM from previous steps if not yet set
        if current_bam is None:
            current_bam = get_bam(sample, paths)

        if 3 in steps:
            current_bam = _step3_markdup(
                sample, current_bam, proc["step3_markdup"], ref, paths
            )

        if 4 in steps:
            if not (current_bam and os.path.exists(current_bam)):
                sys.exit(f"[step4] BAM not found: {current_bam}")
            _step4_methylation(
                sample, current_bam, proc["step4_methylation"], ref, paths
            )

    disp("\nAll samples processed.")
