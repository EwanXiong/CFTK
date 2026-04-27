"""
mesa_cv_plot.py
===============
Visualization for MESA LOOCV results.

Generates three figure types from the loocv_predictions.tsv output:
  A. ROC curves  — one curve per modality + Multimodal
  B. Probability heatmap — samples × modalities, sorted by label
  C. Spearman correlation matrix — pairwise correlation between modality predictions

Input file format (loocv_predictions.tsv)
------------------------------------------
    sample_id | y_true | Modality1 | Modality2 | ... | Multimodal

Usage (called from mesa_loocv.py, or standalone)
-------------------------------------------------
    from mesa_cv_plot import plot_mesa_loocv
    pred_df = pd.read_csv("loocv_predictions.tsv", sep="\\t", index_col=0)
    plot_mesa_loocv(pred_df, output_dir="./output")
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.metrics import roc_curve, roc_auc_score

# ── Style ─────────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family":     "sans-serif",
    "font.size":       7,
    "axes.titlesize":  8,
    "axes.labelsize":  7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "figure.dpi":      150,
})

# Multimodal always drawn first and in a distinct color
MULTIMODAL_COLOR = "#cc0000"   # red

# Default palette for single modalities (cycles if more than 8)
MODALITY_COLORS = [
    "#9b59b6",   # purple   — Methylation
    "#3498db",   # blue     — Occupancy
    "#2ecc71",   # green    — Fuzziness
    "#e67e22",   # orange   — WPS
    "#1abc9c",   # teal
    "#e74c3c",   # red2
    "#34495e",   # dark grey
    "#f39c12",   # yellow-orange
]


def _build_color_map(modality_names):
    """
    Assign a color to each modality.
    'Multimodal' always gets MULTIMODAL_COLOR.
    Others get colors from MODALITY_COLORS in order.
    """
    color_map = {}
    mod_idx = 0
    for name in modality_names:
        if name == "Multimodal":
            color_map[name] = MULTIMODAL_COLOR
        else:
            color_map[name] = MODALITY_COLORS[mod_idx % len(MODALITY_COLORS)]
            mod_idx += 1
    return color_map


# ══════════════════════════════════════════════════════════════════════════════
# Figure A: ROC curves
# ══════════════════════════════════════════════════════════════════════════════

def _plot_roc(ax, pred_df, modality_names, color_map, title="LOOCV"):
    """
    Draw ROC curves for all modalities on a single axes.

    Multimodal is drawn last (on top), others in input order.
    AUC values are shown in the legend, sorted descending.

    Parameters
    ----------
    ax            : matplotlib Axes
    pred_df       : DataFrame with y_true and one column per modality
    modality_names: list of str, column names to plot (excluding y_true)
    color_map     : dict {name: color}
    title         : str, axes title
    """
    y_true = pred_df["y_true"].values

    # Compute AUC for all, then sort descending for legend order
    auc_dict = {}
    for name in modality_names:
        auc_dict[name] = roc_auc_score(y_true, pred_df[name].values)

    # Draw in AUC-ascending order so highest AUC is on top visually
    draw_order = sorted(modality_names, key=lambda n: auc_dict[n])

    for name in draw_order:
        fpr, tpr, _ = roc_curve(y_true, pred_df[name].values)
        auc          = auc_dict[name]
        lw           = 1.8 if name == "Multimodal" else 1.2
        label        = f"{auc:.4f} ({name})"
        ax.plot(fpr, tpr, color=color_map[name], linewidth=lw, label=label)

    # Reference diagonal
    ax.plot([0, 1], [0, 1], color="grey", linewidth=0.5,
            linestyle="--", alpha=0.6)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)

    # Legend: highest AUC first, placed inside upper-left area
    handles, labels = ax.get_legend_handles_labels()
    # reverse so highest AUC appears at top of legend
    ax.legend(
        handles[::-1], labels[::-1],
        title="AUC",
        frameon=False,
        loc="lower right",
        handlelength=1.0,
        fontsize=6,
        title_fontsize=6,
    )
    sns.despine(ax=ax)


# ══════════════════════════════════════════════════════════════════════════════
# Figure B: Probability heatmap
# ══════════════════════════════════════════════════════════════════════════════

def _plot_prob_heatmap(fig, gs_row, pred_df, modality_names,
                       color_map, title="LOOCV"):
    """
    Draw a probability heatmap using a GridSpec sub-layout.

    Layout (top to bottom):
      - Condition bar  (Cancer=dark red, Non-Cancer=light grey)
      - One heatmap row per modality (predicted probability, 0–1)

    Samples are sorted: Cancer first (left), Non-Cancer second (right).

    Parameters
    ----------
    fig           : matplotlib Figure
    gs_row        : GridSpec row object (one row of the outer grid)
    pred_df       : DataFrame
    modality_names: list of str (single modalities only, no Multimodal)
    color_map     : dict
    title         : str
    """
    y_true = pred_df["y_true"].values

    # Sort samples: Cancer (1) first, then Non-Cancer (0)
    sort_idx    = np.argsort(-y_true)   # descending sort (1s first)
    y_sorted    = y_true[sort_idx]
    n_cancer    = int(y_sorted.sum())
    n_total     = len(y_sorted)

    # Build sub-gridspec: 1 condition row + len(modality_names) heatmap rows
    n_rows    = 1 + len(modality_names)
    row_ratios = [0.25] + [1.0] * len(modality_names)
    inner_gs   = gridspec.GridSpecFromSubplotSpec(
        n_rows, 1,
        subplot_spec=gs_row,
        hspace=0.05,
        height_ratios=row_ratios,
    )

    # ── Condition bar ─────────────────────────────────────────────────────────
    ax_cond = fig.add_subplot(inner_gs[0])
    cond_colors = ["#8b1a1a"] * n_cancer + ["#d3d3d3"] * (n_total - n_cancer)
    ax_cond.bar(
        np.arange(n_total), [1] * n_total,
        color=cond_colors, width=1.0, align="edge",
    )
    ax_cond.set_xlim(0, n_total)
    ax_cond.set_ylim(0, 1)
    ax_cond.set_yticks([])
    ax_cond.set_xticks([])
    ax_cond.set_ylabel("Conditions", fontsize=5, rotation=0,
                        labelpad=40, va="center")
    ax_cond.set_title(title, fontsize=8, pad=4)

    # Condition legend (right side of condition bar)
    legend_patches = [
        Patch(facecolor="#8b1a1a", label="Cancer"),
        Patch(facecolor="#d3d3d3", label="Non-Cancer"),
    ]
    ax_cond.legend(
        handles=legend_patches,
        loc="upper right",
        bbox_to_anchor=(1.15, 1.0),
        frameon=False,
        fontsize=5,
    )

    # ── One heatmap row per modality ──────────────────────────────────────────
    for row_i, name in enumerate(modality_names):
        ax_hm = fig.add_subplot(inner_gs[row_i + 1])

        # Sort predictions by the same sample order as the condition bar
        probs_sorted = pred_df[name].values[sort_idx].reshape(1, -1)

        im = ax_hm.imshow(
            probs_sorted,
            aspect="auto",
            cmap="RdBu_r",          # Red=high probability, Blue=low
            vmin=0, vmax=1,
            interpolation="nearest",
        )
        ax_hm.set_yticks([0])
        ax_hm.set_yticklabels([name], fontsize=5)
        ax_hm.set_xticks([])

        # Colorbar only on the last row
        if row_i == len(modality_names) - 1:
            cbar = fig.colorbar(
                im, ax=ax_hm,
                orientation="vertical",
                fraction=0.02, pad=0.01,
                ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            )
            cbar.ax.tick_params(labelsize=5)
            cbar.set_label("Probability", fontsize=5)


# ══════════════════════════════════════════════════════════════════════════════
# Figure C: Spearman correlation heatmap
# ══════════════════════════════════════════════════════════════════════════════

def _plot_spearman(ax, pred_df, modality_names, title="LOOCV"):
    """
    Draw a lower-triangular Spearman correlation heatmap.

    Correlation values are computed between LOOCV predicted probabilities
    of each modality pair, then displayed in the lower triangle.
    Diagonal = 1.

    Parameters
    ----------
    ax            : matplotlib Axes
    pred_df       : DataFrame
    modality_names: list of str (single modalities only, no Multimodal)
    title         : str
    """
    n = len(modality_names)

    # Compute pairwise Spearman correlations
    corr_matrix = np.eye(n)
    for i in range(n):
        for j in range(i):
            r, _ = spearmanr(
                pred_df[modality_names[i]].values,
                pred_df[modality_names[j]].values,
            )
            corr_matrix[i, j] = r
            corr_matrix[j, i] = r   # symmetric (needed for masking)

    # Mask the upper triangle (show lower triangle + diagonal only)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)

    # Draw heatmap using seaborn (handles annotation and color scale)
    sns.heatmap(
        corr_matrix,
        ax=ax,
        mask=mask,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7, "weight": "bold"},
        cmap="YlOrRd",
        vmin=0, vmax=1,
        linewidths=0.5,
        linecolor="white",
        square=True,
        cbar_kws={"shrink": 0.6, "label": "Spearman correlation"},
        xticklabels=modality_names,
        yticklabels=modality_names,
    )

    ax.set_title(title, pad=6)
    ax.tick_params(axis="x", labelrotation=30)
    ax.tick_params(axis="y", labelrotation=0)

    # Add "Spearman correlation" label below x-axis (matching reference figure)
    ax.set_xlabel("Spearman correlation", fontsize=6, labelpad=4)


# ══════════════════════════════════════════════════════════════════════════════
# Main public function
# ══════════════════════════════════════════════════════════════════════════════

def plot_mesa_loocv(
    pred_df,
    output_dir: str     = ".",
    cohort_label: str   = "LOOCV",
    dpi: int            = 300,
):
    """
    Generate all three MESA LOOCV visualization figures.

    Parameters
    ----------
    pred_df      : pd.DataFrame
                   Output of run_mesa_loocv(), or loaded from
                   loocv_predictions.tsv (index_col=0).
                   Must have columns: y_true, <modality names>, Multimodal.
    output_dir   : str, directory where figures are saved.
    cohort_label : str, label shown in each figure title.
                   e.g. "LOOCV on Cohort 1"
    dpi          : int, output resolution (default 300).

    Output files
    ------------
    <output_dir>/mesa_roc_<cohort_label>.pdf
    <output_dir>/mesa_heatmap_<cohort_label>.pdf
    <output_dir>/mesa_spearman_<cohort_label>.pdf
    """
    os.makedirs(output_dir, exist_ok=True)

    # Separate modality columns from y_true
    all_cols       = [c for c in pred_df.columns if c != "y_true"]
    # Single modalities: everything except Multimodal
    single_modal   = [c for c in all_cols if c != "Multimodal"]
    # All modalities for ROC (Multimodal drawn last/on top)
    roc_order      = single_modal + (["Multimodal"] if "Multimodal" in all_cols else [])

    color_map = _build_color_map(roc_order)

    # Sanitize cohort label for use in filenames
    label_slug = cohort_label.replace(" ", "_").replace("/", "-")

    # ── Figure A: ROC curves ─────────────────────────────────────────────────
    fig_roc, ax_roc = plt.subplots(figsize=(3.5, 3.5))
    _plot_roc(
        ax=ax_roc,
        pred_df=pred_df,
        modality_names=roc_order,
        color_map=color_map,
        title=cohort_label,
    )
    fig_roc.tight_layout()
    roc_path = os.path.join(output_dir, f"mesa_roc_{label_slug}.pdf")
    fig_roc.savefig(roc_path, dpi=dpi, bbox_inches="tight")
    print(f"[saved] {roc_path}")
    plt.close(fig_roc)

    # ── Figure B: Probability heatmap ────────────────────────────────────────
    # Height: condition bar (0.3) + one row per single modality (0.5 each)
    heatmap_height = 0.4 + len(single_modal) * 0.55
    fig_hm = plt.figure(figsize=(8, max(heatmap_height, 2.0)))
    gs_hm  = gridspec.GridSpec(1, 1, figure=fig_hm)
    _plot_prob_heatmap(
        fig=fig_hm,
        gs_row=gs_hm[0],
        pred_df=pred_df,
        modality_names=single_modal,
        color_map=color_map,
        title=cohort_label,
    )
    fig_hm.tight_layout()
    hm_path = os.path.join(output_dir, f"mesa_heatmap_{label_slug}.pdf")
    fig_hm.savefig(hm_path, dpi=dpi, bbox_inches="tight")
    print(f"[saved] {hm_path}")
    plt.close(fig_hm)

    # ── Figure C: Spearman correlation heatmap ───────────────────────────────
    n_modal     = len(single_modal)
    cell_size   = 1.2   # inches per cell
    fig_size    = max(n_modal * cell_size, 3.0)
    fig_sp, ax_sp = plt.subplots(figsize=(fig_size, fig_size))
    _plot_spearman(
        ax=ax_sp,
        pred_df=pred_df,
        modality_names=single_modal,
        title=cohort_label,
    )
    fig_sp.tight_layout()
    sp_path = os.path.join(output_dir, f"mesa_spearman_{label_slug}.pdf")
    fig_sp.savefig(sp_path, dpi=dpi, bbox_inches="tight")
    print(f"[saved] {sp_path}")
    plt.close(fig_sp)

    print(f"\n[mesa_cv_plot] All figures saved to: {output_dir}")