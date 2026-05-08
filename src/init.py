"""
init.py — validate cftk_init.json and provide path/config helpers.
Does NOT create any directories — each command creates its own dirs at runtime.
"""

import json
import os
import sys
from util import disp, run_command

REQUIRED_TOP  = ["project_name", "output_dir", "comparison", "samples",
                  "reference_data", "process", "analysis"]
REQUIRED_REF  = ["genome_fa", "genome_2bit", "chrom_sizes"]
PROCESS_STEPS = ["step1_trimming", "step2_alignment",
                 "step3_markdup", "step4_methylation"]

_SAFE_CHARS = set("abcdefghijklmnopqrstuvwxyz"
                  "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                  "0123456789-_.")


def load_config(config_path="./cftk_init.json"):
    if not os.path.exists(config_path):
        sys.exit(
            f"[cftk] ERROR: config not found: {config_path}\n"
            "Create cftk_init.json manually."
        )
    with open(config_path) as f:
        raw = json.load(f)
    cfg    = _strip_comments(raw)
    errors = _validate(cfg)
    if errors:
        disp("ERROR: invalid cftk_init.json:")
        for e in errors:
            disp(f"  x {e}")
        sys.exit(1)
    disp(f"Config loaded: {config_path}")
    return cfg


def _strip_comments(obj):
    if isinstance(obj, dict):
        return {k: _strip_comments(v) for k, v in obj.items()
                if not k.startswith("_comment")}
    if isinstance(obj, list):
        return [_strip_comments(v) for v in obj]
    return obj


def _validate(cfg):
    errors = []

    for k in REQUIRED_TOP:
        if k not in cfg:
            errors.append(f"Missing required top-level key: '{k}'")

    if not cfg.get("output_dir", "").strip():
        errors.append("'output_dir' must not be empty.")

    comp = cfg.get("comparison", "")
    if "_vs_" not in comp:
        errors.append("'comparison' must be formatted as 'GroupA_vs_GroupB'.")
    else:
        ga, gb = comp.split("_vs_", 1)
        samples = cfg.get("samples", {})
        if ga not in samples:
            errors.append(f"comparison group_a '{ga}' not found in samples.")
        if gb not in samples:
            errors.append(f"comparison group_b '{gb}' not found in samples.")

    samples = cfg.get("samples", {})
    if len(samples) != 2:
        errors.append(f"'samples' must define exactly 2 groups, got {len(samples)}.")

    for grp, members in samples.items():
        if not isinstance(members, list) or len(members) == 0:
            errors.append(f"Group '{grp}' must be a non-empty list.")
            continue
        for s in members:
            name  = s.get("name", "")
            if not name:
                errors.append(f"A sample in group '{grp}' is missing 'name'.")
            else:
                # sample name: alphanumeric, hyphen, underscore, dot only
                bad = [c for c in name if c not in _SAFE_CHARS]
                if bad:
                    errors.append(
                        f"Sample '{name}' in group '{grp}': name contains "
                        f"invalid characters {bad}. "
                        "Only letters, digits, - _ . are allowed."
                    )
            itype = s.get("input_type", "").lower()
            if itype not in ("fastq", "bam"):
                errors.append(
                    f"Sample '{name}': 'input_type' must be 'fastq' or 'bam'."
                )
            if itype == "fastq" and (not s.get("r1") or not s.get("r2")):
                errors.append(
                    f"Sample '{name}': fastq input requires 'r1' and 'r2'."
                )
            if itype == "bam" and not s.get("bam"):
                errors.append(f"Sample '{name}': bam input requires 'bam'.")

    ref = cfg.get("reference_data", {})
    for k in REQUIRED_REF:
        if not ref.get(k):
            errors.append(f"reference_data.{k} is required.")

    proc = cfg.get("process", {})
    for step in PROCESS_STEPS:
        if step not in proc:
            errors.append(f"process.{step} is missing.")

    return errors


def get_all_samples(cfg):
    """Flat list of all samples with 'group' injected."""
    out = []
    for grp, members in cfg["samples"].items():
        for s in members:
            entry = dict(s)
            entry["group"] = grp
            out.append(entry)
    return out


def get_group_names(cfg):
    """Return (group_a, group_b) from comparison field."""
    ga, gb = cfg["comparison"].split("_vs_", 1)
    return ga, gb


def get_work_paths(cfg):
    """
    All standard subdirectory paths derived from output_dir.
    Directories are NOT created here — each command creates its own at runtime.
    """
    wd = cfg["output_dir"]
    r  = os.path.join(wd, "results")
    p  = os.path.join(r, "1_process")
    f  = os.path.join(r, "4_fragmentomics")
    return {
        "work_dir":      wd,
        "results":       r,
        "power":         os.path.join(r, "0_power"),
        "process":       p,
        "qc":            os.path.join(r, "2_qc"),
        "differential":  os.path.join(r, "3_differential"),
        "fragmentomics": f,
        "mesa":          os.path.join(r, "5_mesa"),
        "report":        os.path.join(r, "report"),
        # process step output dirs
        "trimming":      os.path.join(p, "1_trimming"),
        "alignment":     os.path.join(p, "2_alignment"),
        "markdup":       os.path.join(p, "3_markdup"),
        "methylation":   os.path.join(p, "4_methylation"),
        "cpg_matrix":    os.path.join(p, "5_merged_matrix"),
        # fragmentomics sub-analysis output dirs
        "occ_out":       os.path.join(f, "occupancy"),
        "wps_out":       os.path.join(f, "wps"),
        "delfi_out":     os.path.join(f, "delfi"),
        "end_motif_out": os.path.join(f, "end_motif"),
        "cleavage_out":  os.path.join(f, "cleavage"),
    }


def get_bam(sample, paths):
    """Resolve the best available BAM (markdup > alignment > direct bam)."""
    name = sample["name"]
    for c in [
        os.path.join(paths["markdup"],   f"{name}.markdup.bam"),
        os.path.join(paths["alignment"], f"{name}.bam"),
        sample.get("bam", ""),
    ]:
        if c and os.path.exists(c):
            return c
    return sample.get("bam", "")


def get_matrix_path(paths, modality):
    """Canonical location of the merged feature matrix for each modality."""
    mapping = {
        "cpg":       os.path.join(paths["cpg_matrix"], "cpg_matrix.tsv"),
        "occupancy": os.path.join(paths["occ_out"],    "occupancy_matrix.tsv"),
        "wps":       os.path.join(paths["wps_out"],    "wps_matrix.tsv"),
        "delfi":     os.path.join(paths["delfi_out"],  "delfi_matrix.tsv"),
    }
    if modality in mapping:
        return mapping[modality]
    return os.path.join(paths["fragmentomics"], f"{modality}_matrix.tsv")


def init(args):
    """Validate config and print summary. No directories are created."""
    cfg  = load_config(args.config)
    ga, gb = get_group_names(cfg)

    disp(f"Project    : {cfg['project_name']}")
    disp(f"Output dir : {cfg['output_dir']}/results/")
    disp(f"Comparison : {ga} (label=0) vs {gb} (label=1)")
    for grp, members in cfg["samples"].items():
        names = [s["name"] for s in members]
        disp(f"  {grp} ({len(names)}): {', '.join(names)}")

    ref = cfg["reference_data"]["genome_fa"]
    if getattr(args, "ref_index", False):
        bwameth = cfg["process"]["step2_alignment"]["tool"]
        run_command(f"{bwameth} index {ref}")
        run_command(f"samtools faidx {ref}")
    if getattr(args, "ref_dict", False):
        run_command(f"picard CreateSequenceDictionary R={ref} O={ref}.dict")

    disp("Init complete. Run: cftk process -s 1 2 3 4")
