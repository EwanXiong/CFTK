"""QC plots: methylation distribution, fragment length, dinucleotide freq, power curves."""

import glob
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed

mpl.rcParams.update({
    "font.family": "sans-serif", "font.size": 8,
    "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
})


def _save(fig, png_path, pdf_path):
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_qc] saved → {png_path}")


# ── Methylation distribution ──────────────────────────────────────────────────

def plot_methylation_distribution(matrix_path, png_path, pdf_path, args):
    df   = pd.read_csv(matrix_path, sep="\t", index_col=0)
    step = getattr(args, "step_size", 2000)

    xmin = float(np.nanmin(df.values))
    xmax = float(np.nanmax(df.values))

    sns.set_context("paper", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(4, 4))
    df.iloc[::step, :].plot.density(
        ax=ax, ind=np.linspace(xmin, xmax, 300), legend=False
    )
    ax.set_xlabel("Methylation β-value")
    ax.set_ylabel("Density")
    ax.set_title(getattr(args, "title", None) or "Methylation Distribution")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, png_path, pdf_path)


# ── Fragment length ───────────────────────────────────────────────────────────

def _load_fragment_ratios(prefix, clip_r1=0, clip_r2=0):
    """
    Load all per-sample raw CSVs, cap at 500bp, return:
      sample_ratios : dict {sample_name: ratio Series (index=size_arr)}
      size_arr      : np.array of fragment sizes (len=500)
      base          : pd.DataFrame Size index for merging
    """
    raw_files = sorted(glob.glob(f"{prefix}.*.raw.csv"))
    if not raw_files:
        return {}, None, None

    base     = pd.DataFrame({"Size": np.arange(500)})
    size_arr = np.arange(500) + clip_r1 + clip_r2

    sample_ratios = {}
    for fp in raw_files:
        # stem: fragment_length.{stem}.raw.csv → extract stem
        stem = os.path.basename(fp).replace(
            os.path.basename(prefix) + ".", ""
        ).replace(".raw.csv", "")
        # strip .markdup suffix to recover sample name
        name = stem.replace(".markdup", "")

        t = pd.read_table(fp, skiprows=1).iloc[:, :2]
        t = t[t["Size"] < 500].copy()

        merged = (
            pd.merge(base, t, on="Size", how="left")
              .sort_values("Size")["Occurrences"]
              .fillna(0)
        )
        mean_vals = merged.values.astype(float)
        total     = mean_vals.sum()
        ratio     = pd.Series(
            100 * mean_vals / total if total > 0 else mean_vals,
            index=size_arr,
        )
        sample_ratios[name] = ratio

    return sample_ratios, size_arr, base


def _frag_ax_style(ax, size_arr, peak):
    """Apply consistent x-axis style: fixed range 50-250, peak vline, rotated ticks."""
    fixed_ticks = sorted(set([50, 100, peak, 200, 250]))
    ax.set_xlim(50, 250)
    ax.set_xticks(fixed_ticks)
    ax.set_xticklabels([str(t) for t in fixed_ticks], rotation=90)
    ax.axvline(peak, color="red", linestyle="-.", linewidth=1, alpha=0.7,
               label=f"peak={peak}bp")
    ax.set_xlabel("Fragment length (bp)")
    ax.set_ylabel("% fragments")
    ax.spines[["top", "right"]].set_visible(False)


def plot_fragment_length(prefix, png_path, pdf_path, args):
    """
    Figure 1: all-sample mean curve.
    Figures 2 & 3: per-group and comparison curves (when group_labels provided).
    """
    clip_r1      = getattr(args, "clip_r1", 0)
    clip_r2      = getattr(args, "clip_r2", 0)
    group_labels = getattr(args, "group_labels", None)

    sample_ratios, size_arr, _ = _load_fragment_ratios(prefix, clip_r1, clip_r2)
    if not sample_ratios:
        print("[plot_qc] No fragment length files found, skipping.")
        return

    # ── Figure 1: all-sample mean ─────────────────────────────────────────────
    all_ratio = pd.concat(list(sample_ratios.values()), axis=1).mean(axis=1)
    peak      = int(size_arr[all_ratio.values.argmax()])

    sns.set_context("paper", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(size_arr, all_ratio.values, linewidth=2, color="#2c7bb6")
    _frag_ax_style(ax, size_arr, peak)
    ax.set_ylim(all_ratio.min() - 0.05, all_ratio.max() + 0.05)
    ax.set_title(getattr(args, "title", None) or "Fragment Length Distribution")
    fig.tight_layout()
    _save(fig, png_path, pdf_path)

    # ── Figures 2 & 3: per-group and comparison ───────────────────────────────
    if not group_labels:
        return

    out_dir = os.path.dirname(png_path)

    # build group → mean ratio
    group_mean = {}
    for grp, col_names in group_labels.items():
        # match sample_ratios keys to col_names (strip .markdup already done)
        grp_ratios = [v for k, v in sample_ratios.items() if k in col_names]
        if not grp_ratios:
            print(f"[plot_qc] WARNING: no fragment data found for group '{grp}'")
            continue
        group_mean[grp] = pd.concat(grp_ratios, axis=1).mean(axis=1)

    colors = ["#2c7bb6", "#d7191c", "#1a9641", "#fdae61"]

    # Figure 2: one plot per group
    for i, (grp, ratio) in enumerate(group_mean.items()):
        grp_peak = int(size_arr[ratio.values.argmax()])
        fig, ax  = plt.subplots(figsize=(4, 4))
        ax.plot(size_arr, ratio.values, linewidth=2, color=colors[i % len(colors)])
        _frag_ax_style(ax, size_arr, grp_peak)
        ax.set_ylim(ratio.min() - 0.05, ratio.max() + 0.05)
        ax.set_title(f"Fragment Length Distribution — {grp}")
        fig.tight_layout()
        _save(fig,
              os.path.join(out_dir, f"fragment_length_{grp}.png"),
              os.path.join(out_dir, f"fragment_length_{grp}.pdf"))

    # Figure 3: both groups on one plot
    if len(group_mean) >= 2:
        fig, ax = plt.subplots(figsize=(4, 4))
        all_peaks = []
        for i, (grp, ratio) in enumerate(group_mean.items()):
            grp_peak = int(size_arr[ratio.values.argmax()])
            all_peaks.append(grp_peak)
            ax.plot(size_arr, ratio.values, linewidth=2,
                    color=colors[i % len(colors)], label=grp)
            ax.axvline(grp_peak, color=colors[i % len(colors)],
                       linestyle="-.", linewidth=1, alpha=0.7,
                       label=f"peak={grp_peak}bp")
        # x-axis ticks: union of fixed + both group peaks
        fixed_ticks = sorted(set([50, 100, 200, 250] + all_peaks))
        ax.set_xlim(50, 250)
        ax.set_xticks(fixed_ticks)
        ax.set_xticklabels([str(t) for t in fixed_ticks], rotation=90)
        ax.set_xlabel("Fragment length (bp)")
        ax.set_ylabel("% fragments")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
        ax.set_title("Fragment Length Distribution — Comparison")
        fig.tight_layout()
        _save(fig,
              os.path.join(out_dir, "fragment_length_comparison.png"),
              os.path.join(out_dir, "fragment_length_comparison.pdf"))


# ── Dinucleotide frequency ────────────────────────────────────────────────────

def plot_dinucleotide_freq(result_prefix, png_path, pdf_path, args):
    """Plot AT/GC dinucleotide frequencies from pre-computed bedtools nuc output."""
    frag_len = getattr(args, "fragment", 167)
    dinu_AT  = ["AA", "AT", "TA", "TT"]
    dinu_GC  = ["GG", "GC", "CG", "CC"]

    def _load(dinucs):
        frames = []
        for d in dinucs:
            fp = f"{result_prefix}.all_fragment_{d}.txt"
            if not os.path.exists(fp):
                continue
            t = (
                pd.read_table(fp, usecols=["4_usercol", "15_user_patt_count"])
                .groupby("4_usercol").sum()
            )
            frames.append(t)
        if not frames:
            return pd.Series(dtype=float)
        return pd.concat(frames, axis=1).mean(axis=1)

    at_sig = _load(dinu_AT)
    gc_sig = _load(dinu_GC)

    if at_sig.empty and gc_sig.empty:
        print("[plot_qc] No dinucleotide files found, skipping.")
        return

    combined = pd.concat([at_sig, gc_sig], axis=1)
    combined.columns = ["AA/AT/TA/TT", "GG/GC/CG/CC"]
    combined.index   = np.arange(-125, 125)

    pct = 100 * combined / combined.values.sum()

    sns.set_context("paper", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(6, 4))
    pct.plot.line(ax=ax, linewidth=1.5)
    ax.set_xlabel(f"Position relative to {frag_len}bp fragment center")
    ax.set_ylabel("Dinucleotide fraction (%)")
    ax.set_title(getattr(args, "title", None) or "Dinucleotide Frequency")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, png_path, pdf_path)


# ── Power curves ──────────────────────────────────────────────────────────────

def plot_power_curves(data, png_path, pdf_path, threshold=0.8):
    """Complementary ECDF of per-CpG power for different read depths."""
    def _ecdf_compl(arr):
        x = np.sort(arr)
        y = 1 - np.arange(1, len(x) + 1) / len(x)
        return np.append(x, x[-1]), np.append(y, 0)

    # parse depth levels from column names like "10_mean", "20_mean", ...
    depth_cols = [c for c in data.columns if c.endswith("_mean")]
    depths     = [c.split("_")[0] for c in depth_cols]

    sns.set_context("paper", font_scale=1.2)
    sns.set_style("ticks")
    fig, ax = plt.subplots(figsize=(6, 5))

    for col, depth in zip(depth_cols, depths):
        x, y = _ecdf_compl(data[col].dropna().values)
        line = ax.plot(x, y, label=str(depth))[0]
        color = line.get_color()
        # CI band if available
        cl = col.replace("_mean", "_CI_l")
        cu = col.replace("_mean", "_CI_u")
        if cl in data.columns and cu in data.columns:
            xl, yl = _ecdf_compl(data[cl].dropna().values)
            xu, yu = _ecdf_compl(data[cu].dropna().values)
            n = min(len(xl), len(xu))
            ax.fill_betweenx(np.linspace(0, 1, n),
                             np.interp(np.linspace(0, 1, n), yl[::-1], xl[::-1]),
                             np.interp(np.linspace(0, 1, n), yu[::-1], xu[::-1]),
                             color=color, alpha=0.1)

    ax.axvline(threshold, color="red", linewidth=1, linestyle="--", alpha=0.7)
    ax.set_xlabel("Minimal Power")
    ax.set_ylabel("Proportion of CpGs with power")
    ax.set_title("Power Analysis — CpG Detection")
    sns.move_legend(ax, title="Avg. read depth",
                    loc="upper left", bbox_to_anchor=(1, 1),
                    frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, png_path, pdf_path)
