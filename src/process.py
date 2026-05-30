"""
process.py — Part 1 raw data processing.
Steps: 1=trimming  2=alignment  3=markdup  4=methylation
Each step creates its own output directory at runtime.
Step 4 auto-merges cpg_matrix.tsv when >1 sample succeeds.
Supports parallel_samples: run N samples simultaneously per step.
"""

import os
import sys
from joblib import Parallel, delayed
from util import disp, run_command
from init import load_config, get_all_samples, get_work_paths, get_bam

STEPS = {
    1: "Adapter trimming",
    2: "Bisulfite alignment",
    3: "Mark duplicates",
    4: "CpG methylation calling",
}

SUPPORTED_TOOLS = {
    1: {"trim_galore", "fastp"},
    2: {"bwameth", "bismark"},
    3: {"sambamba", "picard", "samblaster"},
    4: {"methyldackel", "bismark_extractor"},
}


# ── Step implementations (single sample) ─────────────────────────────────────

def _step1_trim(sample, step_cfg, ref_data, paths, per_cores):
    if sample["input_type"] != "fastq":
        disp(f"  [trim] skip BAM input: {sample['name']}")
        return None, None

    tool  = step_cfg["tool"]
    extra = step_cfg["params"].get("extra_args", "")
    name  = sample["name"]
    out   = paths["trimming"]
    os.makedirs(out, exist_ok=True)

    if tool == "trim_galore":
        cmd = (
            f"trim_galore --paired --2colour 20 --cores {per_cores} "
            f"--fastqc "  # generate FastQC reports for MultiQC
            f"-o {out} --basename {name} {extra} "
            f"{sample['r1']} {sample['r2']} || exit 1"
        )
    elif tool == "fastp":
        ext    = "fq.gz" if sample["r1"].endswith(".gz") else "fq"
        r1_out = os.path.join(out, f"{name}_val_1.{ext}")
        r2_out = os.path.join(out, f"{name}_val_2.{ext}")
        cmd = (
            f"fastp -i {sample['r1']} -I {sample['r2']} "
            f"-o {r1_out} -O {r2_out} -w {per_cores} {extra} || exit 1"
        )
    else:
        sys.exit(f"[step1] unsupported tool: {tool}. Supported: {SUPPORTED_TOOLS[1]}")

    run_command(cmd, label=f"trim [{name}]")
    ext    = "fq.gz" if sample["r1"].endswith(".gz") else "fq"
    r1_out = os.path.join(out, f"{name}_val_1.{ext}")
    r2_out = os.path.join(out, f"{name}_val_2.{ext}")

    if tool == "trim_galore":
        # Trim Galore names reports after the input fastq filename.
        # Rename all report/fastqc files to use sample name instead.
        import glob as _glob
        r1_base = os.path.splitext(os.path.splitext(os.path.basename(sample["r1"]))[0])[0]
        r2_base = os.path.splitext(os.path.splitext(os.path.basename(sample["r2"]))[0])[0]
        rename_map = [
            # trimming reports
            (f"{r1_base}.fastq.gz_trimming_report.txt", f"{name}_R1_trimming_report.txt"),
            (f"{r2_base}.fastq.gz_trimming_report.txt", f"{name}_R2_trimming_report.txt"),
            (f"{r1_base}_trimming_report.txt",           f"{name}_R1_trimming_report.txt"),
            (f"{r2_base}_trimming_report.txt",           f"{name}_R2_trimming_report.txt"),
            # fastqc reports (zip + html)
            (f"{name}_val_1_fastqc.zip",  f"{name}_R1_fastqc.zip"),
            (f"{name}_val_1_fastqc.html", f"{name}_R1_fastqc.html"),
            (f"{name}_val_2_fastqc.zip",  f"{name}_R2_fastqc.zip"),
            (f"{name}_val_2_fastqc.html", f"{name}_R2_fastqc.html"),
        ]
        for old_name, new_name in rename_map:
            old_path = os.path.join(out, old_name)
            new_path = os.path.join(out, new_name)
            if os.path.exists(old_path) and not os.path.exists(new_path):
                os.rename(old_path, new_path)
                disp(f"  [trim] renamed: {old_name} → {new_name}")

    return r1_out, r2_out


def _step2_align(sample, r1, r2, step_cfg, ref_data, paths, per_cores):
    tool  = step_cfg["tool"]
    extra = step_cfg["params"].get("extra_args", "")
    ref   = ref_data["genome_fa"]
    name  = sample["name"]
    out   = paths["alignment"]
    os.makedirs(out, exist_ok=True)
    bam   = os.path.join(out, f"{name}.bam")

    if tool == "bwameth":
        cmd = (
            f"bwameth.py --reference {ref} -t {per_cores} {extra} {r1} {r2} | "
            f"sambamba view -t {per_cores} "
            f"-F 'not secondary_alignment and not failed_quality_control "
            f"and not supplementary and proper_pair and mapping_quality > 0' "
            f"-f bam -S -l 0 /dev/stdin | "
            f"sambamba sort -t {per_cores} -o {bam} /dev/stdin || exit 1; "
            f"samtools index -@ {per_cores} {bam} || exit 1"
        )
    elif tool == "bismark":
        cmd = (
            f"bismark --genome {os.path.dirname(ref)} -1 {r1} -2 {r2} "
            f"-p {per_cores} -o {out} {extra} || exit 1"
        )
    else:
        sys.exit(f"[step2] unsupported tool: {tool}. Supported: {SUPPORTED_TOOLS[2]}")

    run_command(cmd, label=f"align [{name}]")
    # generate samtools flagstat + stats for MultiQC
    run_command(
        f"samtools flagstat -@ {per_cores} {bam} > {bam}.flagstat",
        label=f"flagstat [{name}]"
    )
    run_command(
        f"samtools stats   -@ {per_cores} {bam} > {bam}.stats",
        label=f"samtools_stats [{name}]"
    )
    return bam


def _step3_markdup(sample, bam_in, step_cfg, ref_data, paths, per_cores):
    tool  = step_cfg["tool"]
    extra = step_cfg["params"].get("extra_args", "")
    ref   = ref_data["genome_fa"]
    name  = sample["name"]
    out   = paths["markdup"]
    os.makedirs(out, exist_ok=True)
    bam_out = os.path.join(out, f"{name}.markdup.bam")

    if tool == "sambamba":
        cmd = (
            f"sambamba markdup -t {per_cores} {extra} {bam_in} {bam_out} || exit 1; "
            f"samtools index -@ {per_cores} {bam_out} || exit 1"
        )
    elif tool == "picard":
        metrics = bam_out.replace(".bam", "_metrics.txt")
        cmd = (
            f"picard MarkDuplicates I={bam_in} O={bam_out} R={ref} "
            f"M={metrics} SORTING_COLLECTION_SIZE_RATIO=0.15 "
            f"ASSUME_SORT_ORDER=coordinate OPTICAL_DUPLICATE_PIXEL_DISTANCE=2500 "
            f"MAX_RECORDS_IN_RAM=1000 {extra} || exit 1; "
            f"samtools index -@ {per_cores} {bam_out} || exit 1"
        )
    elif tool == "samblaster":
        cmd = (
            f"samblaster {extra} --addMateTags "
            f"< {bam_in} > {bam_out} || exit 1; "
            f"samtools index -@ {per_cores} {bam_out} || exit 1"
        )
    else:
        sys.exit(f"[step3] unsupported tool: {tool}. Supported: {SUPPORTED_TOOLS[3]}")

    run_command(cmd, label=f"markdup [{name}]")
    return bam_out


def _step4_methylation(sample, bam_in, step_cfg, ref_data, paths, per_cores):
    tool  = step_cfg["tool"]
    depth = step_cfg["params"].get("min_depth", 10)
    extra = step_cfg["params"].get("extra_args", "")
    ref   = ref_data["genome_fa"]
    name  = sample["name"]
    out   = paths["methylation"]
    os.makedirs(out, exist_ok=True)
    prefix = os.path.join(out, name)

    if tool == "methyldackel":
        # *_mbias.txt is recognised by MultiQC
        mbias_txt = f"{prefix}_mbias.txt"
        mbias_temp = f"{prefix}_mbias_OT_OB.temp"
        cmd = (
            f"MethylDackel mbias -@ {per_cores} {ref} {bam_in} {prefix} "
            f"2>&1 | tee {mbias_txt} > {mbias_temp} || exit 1; "
            f"MethylDackel extract --minDepth {depth} --maxVariantFrac 0.25 "
            f"-@ {per_cores} "
            f"--OT $(grep -oP '(?<=--OT )[^ ]+' {mbias_temp}) "
            f"--OB $(grep -oP '(?<=--OB )[^ ]+' {mbias_temp}) "
            f"-o {prefix} {extra} {ref} {bam_in} || exit 1"
        )
    elif tool == "bismark_extractor":
        cmd = (
            f"bismark_methylation_extractor --paired-end --gzip "
            f"--CpG_only --cytosine_report "
            f"--genome_folder {os.path.dirname(ref)} "
            f"-o {out} {extra} {bam_in} || exit 1"
        )
    else:
        sys.exit(f"[step4] unsupported tool: {tool}. Supported: {SUPPORTED_TOOLS[4]}")

    run_command(cmd, label=f"methylation [{name}]")
    return f"{prefix}_CpG.bedGraph"


# ── CPG merge (auto after step4) ──────────────────────────────────────────────

def _merge_cpg(bedgraph_files, samples, paths):
    """
    Merge per-sample CpG bedGraph files into cpg_matrix.tsv.
    Uses bedtools unionbedg for coordinate-level merging — avoids duplicate
    index errors from MethylDackel outputting both strands at the same position.
    Column names: {group}_{sample_name} so startswith(group) matching works
    in downstream diff/mesa analysis.
    """
    import shutil
    import subprocess
    import pandas as pd

    out_dir = paths["cpg_matrix"]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "cpg_matrix.tsv")

    bedtools = shutil.which("bedtools")
    if bedtools is None:
        sys.exit("[merge] ERROR: bedtools not found in PATH.")

    # col_name = sample["name"] directly (user controls naming in json)
    name_map = {s["name"]: s["name"] for s in samples}
    col_names = []
    for fp in bedgraph_files:
        sample_name = os.path.basename(fp).replace("_CpG.bedGraph", "")
        col_names.append(name_map.get(sample_name, sample_name))
    col_names_str = " ".join(col_names)

    stripped = " ".join(
        f"<(grep -v '^track' {fp} | cut -f1-4)" for fp in bedgraph_files
    )
    tmp_path = out_path + ".tmp"
    cmd = (
        f"bash -c '"
        f"{bedtools} unionbedg -header -names {col_names_str} -filler NA "
        f"-i {stripped}"
        f" | cut -f1,3-"
        f" | sed s/end/pos/"
        f" > {tmp_path}'"
    )

    disp(f"[merge] bedtools unionbedg: {len(bedgraph_files)} files → {out_path}")
    ret = subprocess.run(cmd, shell=True)
    if ret.returncode != 0:
        sys.exit("[merge] ERROR: bedtools unionbedg failed.")

    # read tmp, combine chrom+pos → chrom_pos index (e.g. chr1_17378)
    matrix = pd.read_csv(tmp_path, sep="\t")
    # column 0 is chrom (may be named "chrom" or "chr"), column 1 is pos
    chrom_col = matrix.columns[0]
    pos_col   = matrix.columns[1]
    matrix.index = matrix[chrom_col].astype(str) + "_" + matrix[pos_col].astype(str)
    matrix.index.name = "cpg_id"
    matrix = matrix.drop(columns=[chrom_col, pos_col])
    # drop duplicate positions (e.g. from both strands or overlapping regions)
    n_before = len(matrix)
    matrix = matrix[~matrix.index.duplicated(keep="first")]
    n_dropped = n_before - len(matrix)
    if n_dropped > 0:
        disp(f"[merge] dropped {n_dropped} duplicate CpG positions")
    matrix.to_csv(out_path, sep="\t")
    os.remove(tmp_path)
    disp(f"[merge] {matrix.shape[0]} CpGs × {matrix.shape[1]} samples → {out_path}")
    return out_path


# ── Per-step single-sample runners (used by joblib per step) ─────────────────

def _run_step1(sample, proc, ref, paths, per_cores):
    """Trimming for one sample. Returns (r1, r2) or (None, None)."""
    name = sample["name"]
    ext  = "fq.gz" if sample.get("r1", "").endswith(".gz") else "fq"
    r1_done = os.path.join(paths["trimming"], f"{name}_val_1.{ext}")
    r2_done = os.path.join(paths["trimming"], f"{name}_val_2.{ext}")
    if os.path.exists(r1_done) and os.path.exists(r2_done):
        disp(f"  [step1] {name} — already done, skipping")
        return r1_done, r2_done
    disp(f"  [step1] {name}")
    return _step1_trim(sample, proc["step1_trimming"], ref, paths, per_cores)


def _run_step2(sample, proc, ref, paths, per_cores):
    """Alignment for one sample. Returns bam path, or exits on missing trimmed reads."""
    name  = sample["name"]
    itype = sample["input_type"]
    disp(f"  [step2] {name}")

    if itype == "bam":
        disp(f"  [step2] skip — BAM input: {name}")
        return sample.get("bam")

    # trimmed reads must exist — step1 must have run first
    ext = "fq.gz" if sample["r1"].endswith(".gz") else "fq"
    r1  = os.path.join(paths["trimming"], f"{name}_val_1.{ext}")
    r2  = os.path.join(paths["trimming"], f"{name}_val_2.{ext}")

    missing = [f for f in [r1, r2] if not os.path.exists(f)]
    if missing:
        msg = (
            f"[step2] ERROR: trimmed reads not found for '{name}': "
            + ", ".join(missing)
            + " — run step 1 (trimming) before step 2."
        )
        sys.exit(msg)

    bam_done = os.path.join(paths["alignment"], f"{name}.bam")
    if os.path.exists(bam_done):
        disp(f"  [step2] {name} — already done, skipping")
        return bam_done
    return _step2_align(sample, r1, r2, proc["step2_alignment"],
                        ref, paths, per_cores)


def _run_step3(sample, proc, ref, paths, per_cores):
    """Mark duplicates for one sample. Returns markdup bam path."""
    name    = sample["name"]
    bam_done = os.path.join(paths["markdup"], f"{name}.markdup.bam")
    if os.path.exists(bam_done):
        disp(f"  [step3] {name} — already done, skipping")
        return bam_done
    disp(f"  [step3] {name}")
    bam_in = get_bam(sample, paths)
    if not (bam_in and os.path.exists(bam_in)):
        disp(f"  [step3] ERROR: BAM not found for {sample['name']}: {bam_in}")
        return None
    return _step3_markdup(sample, bam_in,
                           proc["step3_markdup"], ref, paths, per_cores)


def _run_step4(sample, proc, ref, paths, per_cores):
    """Methylation calling for one sample. Returns bedGraph path or None."""
    name    = sample["name"]
    bg_done = os.path.join(paths["methylation"], f"{name}_CpG.bedGraph")
    if os.path.exists(bg_done):
        disp(f"  [step4] {name} — already done, skipping")
        return bg_done
    disp(f"  [step4] {name}")
    bam_in = get_bam(sample, paths)
    if not (bam_in and os.path.exists(bam_in)):
        disp(f"  [step4] ERROR: BAM not found for {sample['name']}: {bam_in}")
        return None
    return _step4_methylation(sample, bam_in,
                               proc["step4_methylation"], ref, paths, per_cores)


# ── Step dispatcher ───────────────────────────────────────────────────────────

_STEP_FN = {
    1: _run_step1,
    2: _run_step2,
    3: _run_step3,
    4: _run_step4,
}


# ── Main entry ────────────────────────────────────────────────────────────────


# ── MultiQC ──────────────────────────────────────────────────────────────────

# keys must match paths dict from init.py get_work_paths()
# only steps 1 and 2 produce MultiQC-recognisable output
_MULTIQC_STEP = {
    1: "trimming",
    2: "alignment",
}

_MULTIQC_TITLE = {
    1: "Trim Galore QC",
    2: "bwameth Alignment",
}


def _run_multiqc(step, paths):
    """
    Run MultiQC on the output directory of a completed step.
    Results saved to {step_dir}/multiqc/.
    Silently skips if multiqc is not installed.
    """
    import shutil
    multiqc = shutil.which("multiqc")
    if not multiqc:
        disp(f"[multiqc] step {step}: multiqc not found, skipping.")
        return

    if step not in _MULTIQC_STEP:
        return  # no MultiQC for this step
    step_dir  = paths.get(_MULTIQC_STEP[step], "")
    if not step_dir or not os.path.exists(step_dir):
        disp(f"[multiqc] step {step}: directory not found ({step_dir}), skipping.")
        return

    out_dir = os.path.join(step_dir, "multiqc")
    os.makedirs(out_dir, exist_ok=True)

    title = _MULTIQC_TITLE.get(step, f"Step {step}")
    scan_dirs = step_dir
    # ignore large fastq/bam files to speed up scanning
    ignore_flags = "--ignore '*.fq.gz' --ignore '*.fastq.gz' --ignore '*.bam' --ignore '*.bai'"
    cmd = (
        f"{multiqc} {scan_dirs} "
        f"--outdir {out_dir} "
        f"--filename multiqc_report.html "
        f"--title \"{title}\" "
        f"--export "
        f"{ignore_flags} "
        f"--force "
        f"--quiet"
    )
    disp(f"[multiqc] step {step}: running MultiQC on {scan_dirs}")
    run_command(cmd, label=f"multiqc [step {step}]")
    html_out = os.path.join(out_dir, "multiqc_report.html")
    if os.path.exists(html_out):
        disp(f"[multiqc] step {step}: report → {html_out}")
    else:
        disp(f"[multiqc] step {step}: WARNING — multiqc_report.html not found")


def process(args, config_path="./cftk_init.json"):
    """
    True step-level parallel processing:
      - All samples run step N in parallel (batched by parallel_samples).
      - Step N+1 starts only after ALL samples finish step N.
    Total cores are split evenly: per_sample_cores = total_cores // parallel_samples.
    CPG matrix is auto-merged after step 4 when >1 sample succeeds.
    """
    cfg     = load_config(config_path)
    paths   = get_work_paths(cfg)
    samples = get_all_samples(cfg)
    steps   = sorted(set(args.step))
    ref     = cfg["reference_data"]
    proc    = cfg["process"]

    # resolve parallel: CLI --parallel > config parallel_samples > 1
    parallel = getattr(args, "parallel", None)                or proc.get("parallel_samples", 1)
    parallel = max(1, int(parallel))

    # per-sample cores: use step-specific total, divide by parallel
    step_cores = {
        s: max(1, proc[key]["params"].get("cores", 20) // parallel)
        for s, key in {
            1: "step1_trimming",
            2: "step2_alignment",
            3: "step3_markdup",
            4: "step4_methylation",
        }.items()
    }

    invalid = [s for s in steps if s not in STEPS]
    if invalid:
        sys.exit(f"[process] Invalid steps: {invalid}. Valid: {list(STEPS.keys())}")

    disp(f"Samples          : {len(samples)}")
    disp(f"Steps            : {[STEPS[s] for s in steps]}")
    disp(f"Parallel samples : {parallel}")
    for s in steps:
        total = proc[{1:'step1_trimming',2:'step2_alignment',
                      3:'step3_markdup',4:'step4_methylation'}[s]
                    ]["params"].get("cores", 20)
        disp(f"  Step {s} cores  : {step_cores[s]} per sample "
             f"(total {total} ÷ {parallel})")

    # ── Step-level parallel: each step completes for ALL samples before next ──
    bedgraph_results = [None] * len(samples)

    for step in steps:
        sep = "=" * 50
        disp(sep)
        disp(f"[step {step}] {STEPS[step]} — {len(samples)} samples "
             f"(parallel={parallel})")
        disp(sep)

        fn        = _STEP_FN[step]
        per_cores = step_cores[step]

        results = Parallel(n_jobs=parallel, backend="multiprocessing")(
            delayed(fn)(s, proc, ref, paths, per_cores)
            for s in samples
        )

        # store bedGraph paths from step4 for later merge
        if step == 4:
            bedgraph_results = results

        disp(f"[step {step}] all samples complete.")
        _run_multiqc(step, paths)

    # ── Auto-merge cpg after step 4 ───────────────────────────────────────────
    if 4 in steps:
        successful = [bg for bg in bedgraph_results
                      if bg and os.path.exists(bg)]
        if len(successful) > 1:
            _merge_cpg(successful, samples, paths)
        elif len(successful) == 1:
            disp("[merge] Single sample — skipping cpg_matrix merge.")
        else:
            disp("[merge] WARNING: no successful bedGraph files found.")

    disp("\nAll samples processed.")
