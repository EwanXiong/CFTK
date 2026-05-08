"""Fragmentomics plots: occupancy, DELFI, end-motif, cleavage profile, WPS."""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

mpl.rcParams.update({
    "font.family": "sans-serif", "font.size": 8,
    "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
})


def _save(fig, png_path, pdf_path):
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_frag] saved → {png_path}")


def _clean_header(df):
    df.columns = df.columns.str.replace("^#", "", regex=True).str.strip()
    return df


# ── Nucleosome occupancy ──────────────────────────────────────────────────────

def plot_occupancy(tsv_path, png_path, pdf_path, figsize=(14, 4)):
    """Genome-wide nucleosome occupancy score line plot from DANPOS3 output."""
    df = pd.read_csv(tsv_path, sep="\t", header=None,
                     names=["chrom", "start", "end", "name", "size",
                            "mean0", "mean", "max", "summit"])
    if "mean" not in df.columns:
        print(f"[plot_occ] unexpected columns in {tsv_path}, skipping.")
        return

    chrom_order = [f"chr{i}" for i in range(1, 23)]
    df["chrom"] = pd.Categorical(df["chrom"], categories=chrom_order, ordered=True)
    df = df.sort_values(["chrom", "start"]).reset_index(drop=True)

    current_pos = 0
    chrom_pos, chrom_ctr = {}, {}
    gpos_list = []
    for chrom in chrom_order:
        sub = df[df["chrom"] == chrom]
        if sub.empty:
            continue
        chrom_pos[chrom] = current_pos
        gpos_list.append(
            pd.Series(current_pos + np.arange(len(sub)), index=sub.index)
        )
        chrom_ctr[chrom]  = current_pos + len(sub) / 2
        current_pos      += len(sub)
    # assign _gpos as aligned Series to avoid loc NaN issue
    df["_gpos"] = pd.concat(gpos_list).reindex(df.index)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(df["_gpos"], df["mean"], color="#e67e22", linewidth=0.6, alpha=0.8)

    for i, chrom in enumerate(chrom_order):
        if chrom not in chrom_pos:
            continue
        pos = chrom_pos[chrom]
        n   = len(df[df["chrom"] == chrom])
        if i % 2 == 0:
            ax.axvspan(pos, pos + n, alpha=0.05, color="gray", zorder=0)
        if i > 0:
            ax.axvline(pos, color="lightgray", linewidth=0.5, linestyle="--", alpha=0.5)

    ax.set_xticks([chrom_ctr[c] for c in chrom_order if c in chrom_ctr])
    ax.set_xticklabels([c.replace("chr", "") for c in chrom_order if c in chrom_ctr],
                       fontsize=7)
    ax.set_xlabel("Chromosome")
    ax.set_ylabel("Occupancy Score")
    ax.set_title(f"{Path(tsv_path).stem} — Nucleosome Occupancy")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
    fig.tight_layout()
    _save(fig, png_path, pdf_path)


# ── DELFI ─────────────────────────────────────────────────────────────────────

def plot_delfi(tsv_path, png_path, pdf_path,
               use_corrected=True, figsize=(20, 4), ylim=None):
    """Genome-wide DELFI score line plot."""
    with open(tsv_path) as f:
        first = f.readline()
    df = pd.read_csv(tsv_path, sep="\t")
    if first.startswith("#"):
        df.columns = df.columns.str.replace("^#", "", regex=True).str.strip()

    ratio_col = "ratio_corrected" if (use_corrected and "ratio_corrected" in df.columns) \
                else "ratio"
    df = df.dropna(subset=[ratio_col])

    chrom_order = [f"chr{i}" for i in range(1, 23)]
    df["contig"] = pd.Categorical(df["contig"], categories=chrom_order, ordered=True)
    df = df.sort_values(["contig", "start"])

    current_pos = 0
    chrom_positions, chrom_centers = {}, {}
    for chrom in chrom_order:
        sub = df[df["contig"] == chrom]
        if sub.empty:
            continue
        chrom_positions[chrom] = current_pos
        df.loc[df["contig"] == chrom, "_gpos"] = current_pos + np.arange(len(sub))
        chrom_centers[chrom]   = current_pos + len(sub) / 2
        current_pos           += len(sub)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(df["_gpos"], df[ratio_col], color="#1f77b4", linewidth=0.8, alpha=0.8)

    for i, chrom in enumerate(chrom_order):
        if chrom not in chrom_positions:
            continue
        pos = chrom_positions[chrom]
        n   = len(df[df["contig"] == chrom])
        if i % 2 == 0:
            ax.axvspan(pos, pos + n, alpha=0.05, color="gray", zorder=0)
        if i > 0:
            ax.axvline(pos, color="lightgray", linewidth=0.5, linestyle="--", alpha=0.5)

    ax.set_xticks([chrom_centers[c] for c in chrom_order if c in chrom_centers])
    ax.set_xticklabels([c.replace("chr", "") for c in chrom_order if c in chrom_centers],
                       fontsize=7)
    ax.set_xlabel("Chromosome")
    ax.set_ylabel("DELFI Score")
    ax.set_title(f"{Path(tsv_path).stem} — DELFI Score")
    if ylim:
        ax.set_ylim(*ylim)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
    fig.tight_layout()
    _save(fig, png_path, pdf_path)


# ── End-motif ─────────────────────────────────────────────────────────────────

def plot_end_motif(tsv_path, png_path, pdf_path, n=20, show_values=True):
    """Horizontal bar chart of top N most frequent k-mers."""
    df   = pd.read_csv(tsv_path, sep="\t", header=None, names=["kmer", "frequency"])
    data = df.nlargest(n, "frequency")

    height = max(6, n * 0.35)
    fig, ax = plt.subplots(figsize=(10, height))
    ax.barh(range(len(data)), data["frequency"],
            color="steelblue", alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(data["kmer"], fontfamily="monospace", fontsize=9)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("K-mer")
    ax.set_title(f"{Path(tsv_path).stem} — Top {n} End Motifs")
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    if show_values:
        for i, val in enumerate(data["frequency"]):
            ax.text(val, i, f" {val:.4f}", va="center", fontsize=7)

    fig.tight_layout()
    _save(fig, png_path, pdf_path)


# ── Cleavage profile ──────────────────────────────────────────────────────────

def plot_cleavage(bw_paths, bed_path, png_path, pdf_path,
                  upstream=1500, downstream=1500, labels=None,
                  smooth_window=10, colors=None):
    """Aggregate cleavage profile over BED regions from bigWig files."""
    try:
        import pyBigWig
    except ImportError:
        print("[plot_cleavage] pyBigWig not installed, skipping.")
        return

    default_colors = ["darkred", "darkblue", "darkgreen", "darkorange", "purple"]
    colors = colors or default_colors
    labels = labels or [Path(b).stem for b in bw_paths]
    window = upstream + downstream

    def _extract(bw_file):
        bw      = pyBigWig.open(bw_file)
        regions = pd.read_csv(bed_path, sep="\t", header=None,
                              names=["chrom", "start", "end", "name", "score", "strand"])
        mat = []
        for _, r in regions.iterrows():
            center = (r["start"] + r["end"]) // 2
            ws, we = max(0, center - upstream), center + downstream
            try:
                vals = bw.values(str(r["chrom"]), ws, we)
                if vals and len(vals) == window:
                    mat.append(np.nan_to_num(np.array(vals)))
            except Exception:
                pass
        bw.close()
        if not mat:
            return None
        sig = np.mean(mat, axis=0)
        if smooth_window > 1:
            sig = np.convolve(sig, np.ones(smooth_window) / smooth_window, mode="same")
        return sig

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(-upstream, downstream)

    for bw_path, label, color in zip(bw_paths, labels, colors):
        sig = _extract(bw_path)
        if sig is None:
            continue
        ax.plot(x, sig * 100, linewidth=2, label=label, color=color, alpha=0.8)

    ax.axvline(0, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.set_xlabel("Distance to CTCF motif (bp)")
    ax.set_ylabel("Cleavage Proportion (%)")
    ax.set_title("Cleavage Profile")
    ax.set_xticks([-1500, -750, 0, 750, 1500])
    ax.set_xticklabels(["-1.5kb", "-750", "0", "750", "1.5kb"])
    ax.legend(frameon=True, fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, png_path, pdf_path)


# ── WPS ───────────────────────────────────────────────────────────────────────

def plot_wps(tsv_path, png_path, pdf_path, figsize=(14, 4)):
    """Mean WPS per region, plotted genome-wide."""
    df = pd.read_csv(tsv_path, sep="\t")
    if "mean_WPS" not in df.columns:
        print(f"[plot_wps] mean_WPS column not found in {tsv_path}, skipping.")
        return

    chrom_order = [f"chr{i}" for i in range(1, 23)]
    df["chr"]   = pd.Categorical(df["chr"], categories=chrom_order, ordered=True)
    df          = df.sort_values(["chr", "start"]).reset_index(drop=True)

    current_pos  = 0
    chrom_pos, chrom_ctr = {}, {}
    for chrom in chrom_order:
        sub = df[df["chr"] == chrom]
        if sub.empty:
            continue
        chrom_pos[chrom] = current_pos
        df.loc[df["chr"] == chrom, "_gpos"] = current_pos + np.arange(len(sub))
        chrom_ctr[chrom]  = current_pos + len(sub) / 2
        current_pos      += len(sub)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(df["_gpos"], df["mean_WPS"], color="#2c7bb6", linewidth=0.6, alpha=0.8)

    for i, chrom in enumerate(chrom_order):
        if chrom not in chrom_pos:
            continue
        pos = chrom_pos[chrom]
        n   = len(df[df["chr"] == chrom])
        if i % 2 == 0:
            ax.axvspan(pos, pos + n, alpha=0.05, color="gray", zorder=0)
        if i > 0:
            ax.axvline(pos, color="lightgray", linewidth=0.5, linestyle="--", alpha=0.5)

    ax.set_xticks([chrom_ctr[c] for c in chrom_order if c in chrom_ctr])
    ax.set_xticklabels([c.replace("chr", "") for c in chrom_order if c in chrom_ctr],
                       fontsize=7)
    ax.set_xlabel("Chromosome")
    ax.set_ylabel("Mean WPS")
    ax.set_title(f"{Path(tsv_path).stem} — WPS")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
    fig.tight_layout()
    _save(fig, png_path, pdf_path)
