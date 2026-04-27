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

MULTIMODAL_COLOR = "#cc0000"

MODALITY_COLORS = [
    "#9b59b6",   # purple
    "#3498db",   # blue
    "#2ecc71",   # green
    "#e67e22",   # orange
    "#1abc9c",   # teal
    "#e74c3c",   # red2
    "#34495e",   # dark grey
    "#f39c12",   # yellow-orange
]


def _build_color_map(modality_names):
    color_map = {}
    mod_idx = 0
    for name in modality_names:
        if name == "Multimodal":
            color_map[name] = MULTIMODAL_COLOR
        else:
            color_map[name] = MODALITY_COLORS[mod_idx % len(MODALITY_COLORS)]
            mod_idx += 1
    return color_map


def _infer_group_names(y_true, sample_index):
    """
    Infer group names from sample IDs (index of pred_df).

    Samples are assumed to be named <GroupName>_<number>, e.g. "Control_1",
    "sALS_3". The prefix before the last underscore+digit is the group name.
    y_true=1 → group with the higher numeric label (positive class).
    y_true=0 → the other group.

    Falls back to ("Group1", "Group0") if names cannot be determined.

    Returns
    -------
    label1 : str  name for y_true == 1  (positive class, drawn first)
    label0 : str  name for y_true == 0  (negative class)
    """
    import re
    names_1 = set()
    names_0 = set()
    for sid, label in zip(sample_index, y_true):
        # Extract prefix: everything before trailing _<digits>
        match = re.match(r'^(.+?)(?:_\d+)?$', str(sid))
        prefix = match.group(1) if match else str(sid)
        if label == 1:
            names_1.add(prefix)
        else:
            names_0.add(prefix)

    label1 = ", ".join(sorted(names_1)) if names_1 else "Group1"
    label0 = ", ".join(sorted(names_0)) if names_0 else "Group0"
    return label1, label0


# ══════════════════════════════════════════════════════════════════════════════
# Figure A: ROC curves
# ══════════════════════════════════════════════════════════════════════════════

def _plot_roc(ax, pred_df, modality_names, color_map, title="LOOCV"):
    """
    ROC curves with full 4-sided box frame and slight x-axis left margin.
    """
    y_true = pred_df["y_true"].values

    auc_dict = {}
    for name in modality_names:
        auc_dict[name] = roc_auc_score(y_true, pred_df[name].values)

    draw_order = sorted(modality_names, key=lambda n: auc_dict[n])

    for name in draw_order:
        fpr, tpr, _ = roc_curve(y_true, pred_df[name].values)
        auc  = auc_dict[name]
        lw   = 1.8 if name == "Multimodal" else 1.2
        label = f"{auc:.4f} ({name})"
        ax.plot(fpr, tpr, color=color_map[name], linewidth=lw, label=label)

    ax.plot([0, 1], [0, 1], color="grey", linewidth=0.5,
            linestyle="--", alpha=0.6)

    # Leave a small margin on the left so the y-axis line is visible
    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)

    # Full 4-sided box frame
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[::-1], labels[::-1],
        title="AUC",
        frameon=False,
        loc="lower right",
        handlelength=1.0,
        fontsize=6,
        title_fontsize=6,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Figure B: Probability heatmap
# ══════════════════════════════════════════════════════════════════════════════

def _plot_prob_heatmap(fig, gs_row, pred_df, modality_names,
                       color_map, title="LOOCV"):
    """
    Probability heatmap.

    - Group names inferred from sample IDs (not hardcoded as Cancer/Non-Cancer).
    - Condition legend placed outside the heatmap area to the right of the figure.
    - Probability colorbar placed outside to the right, below the condition legend.
    - Samples sorted: positive class (y_true=1) first (left).
    """
    y_true       = pred_df["y_true"].values
    sample_index = pred_df.index

    # Infer actual group names from sample IDs
    label1, label0 = _infer_group_names(y_true, sample_index)

    # Sort samples: positive class first
    sort_idx = np.argsort(-y_true)
    y_sorted = y_true[sort_idx]
    n_pos    = int(y_sorted.sum())
    n_total  = len(y_sorted)

    # Sub-gridspec: 1 condition row + N heatmap rows
    n_rows     = 1 + len(modality_names)
    row_ratios = [0.25] + [1.0] * len(modality_names)
    inner_gs   = gridspec.GridSpecFromSubplotSpec(
        n_rows, 1,
        subplot_spec=gs_row,
        hspace=0.05,
        height_ratios=row_ratios,
    )

    # ── Condition bar ─────────────────────────────────────────────────────────
    ax_cond = fig.add_subplot(inner_gs[0])
    # Positive class = dark red, negative = light grey
    POS_COLOR = "#8b1a1a"
    NEG_COLOR = "#d3d3d3"
    cond_colors = [POS_COLOR] * n_pos + [NEG_COLOR] * (n_total - n_pos)
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

    # Condition legend: placed to the right of the figure using fig.legend
    # (stored as patches; added to fig after all axes are created)
    condition_patches = [
        Patch(facecolor=POS_COLOR, edgecolor="none", label=label1),
        Patch(facecolor=NEG_COLOR, edgecolor="none", label=label0),
    ]

    # ── Heatmap rows ──────────────────────────────────────────────────────────
    last_im = None
    for row_i, name in enumerate(modality_names):
        ax_hm = fig.add_subplot(inner_gs[row_i + 1])
        probs_sorted = pred_df[name].values[sort_idx].reshape(1, -1)
        last_im = ax_hm.imshow(
            probs_sorted,
            aspect="auto",
            cmap="RdBu_r",
            vmin=0, vmax=1,
            interpolation="nearest",
        )
        ax_hm.set_yticks([0])
        ax_hm.set_yticklabels([name], fontsize=5)
        ax_hm.set_xticks([])

    return condition_patches, last_im


# ══════════════════════════════════════════════════════════════════════════════
# Figure C: Spearman correlation heatmap
# ══════════════════════════════════════════════════════════════════════════════

def _plot_spearman(ax, pred_df, modality_names, title="LOOCV"):
    n = len(modality_names)
    corr_matrix = np.eye(n)
    for i in range(n):
        for j in range(i):
            r, _ = spearmanr(
                pred_df[modality_names[i]].values,
                pred_df[modality_names[j]].values,
            )
            corr_matrix[i, j] = r
            corr_matrix[j, i] = r

    mask = np.triu(np.ones((n, n), dtype=bool), k=1)

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
                   Must have columns: y_true, <modality names>, Multimodal.
                   index = sample IDs (used to infer group names).
    output_dir   : str, directory where figures are saved.
    cohort_label : str, label shown in each figure title.
    dpi          : int, output resolution (default 300).

    Output files
    ------------
    <output_dir>/mesa_roc_<cohort_label>.pdf
    <output_dir>/mesa_heatmap_<cohort_label>.pdf
    <output_dir>/mesa_spearman_<cohort_label>.pdf
    """
    os.makedirs(output_dir, exist_ok=True)

    all_cols     = [c for c in pred_df.columns if c != "y_true"]
    single_modal = [c for c in all_cols if c != "Multimodal"]
    roc_order    = single_modal + (["Multimodal"] if "Multimodal" in all_cols else [])
    color_map    = _build_color_map(roc_order)
    label_slug   = cohort_label.replace(" ", "_").replace("/", "-")

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
    heatmap_height = 0.4 + len(single_modal) * 0.55
    # Extra right margin for the external legends (condition + colorbar)
    fig_hm = plt.figure(figsize=(9, max(heatmap_height, 2.0)))

    # Reserve right portion for legends: heatmap takes left 88%, legends right 12%
    gs_outer = gridspec.GridSpec(
        1, 2,
        figure=fig_hm,
        width_ratios=[0.88, 0.12],
        wspace=0.02,
    )

    condition_patches, last_im = _plot_prob_heatmap(
        fig=fig_hm,
        gs_row=gs_outer[0],
        pred_df=pred_df,
        modality_names=single_modal,
        color_map=color_map,
        title=cohort_label,
    )

    # Right panel: condition legend (top) + probability colorbar (below)
    ax_legend = fig_hm.add_subplot(gs_outer[1])
    ax_legend.axis("off")

    # Condition legend at the top of the right panel
    ax_legend.legend(
        handles=condition_patches,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        frameon=True,
        framealpha=0.0,
        edgecolor="none",
        fontsize=6,
        title="Conditions",
        title_fontsize=6,
        labelspacing=0.4,
        handlelength=1.0,
        handleheight=1.0,
    )

    # Probability colorbar below the condition legend
    # Use a small inset axes inside the right panel for the colorbar
    cbar_ax = fig_hm.add_axes([
        ax_legend.get_position().x0 + 0.01,   # x start
        ax_legend.get_position().y0 + 0.05,   # y start
        0.018,                                  # width
        ax_legend.get_position().height * 0.55, # height (55% of right panel)
    ])
    cbar = fig_hm.colorbar(last_im, cax=cbar_ax,
                            ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.ax.tick_params(labelsize=5)
    cbar.set_label("Probability", fontsize=5, labelpad=3)

    hm_path = os.path.join(output_dir, f"mesa_heatmap_{label_slug}.pdf")
    fig_hm.savefig(hm_path, dpi=dpi, bbox_inches="tight")
    print(f"[saved] {hm_path}")
    plt.close(fig_hm)

    # ── Figure C: Spearman correlation heatmap ───────────────────────────────
    n_modal   = len(single_modal)
    cell_size = 1.2
    fig_size  = max(n_modal * cell_size, 3.0)
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


if __name__ == "__main__":
    import argparse
 
    parser = argparse.ArgumentParser(
        description="Generate MESA LOOCV visualizations from loocv_predictions.tsv.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python mesa_cv_plot.py \\\n"
            "      --input  output/loocv_predictions.tsv \\\n"
            "      --output output/ \\\n"
            "      --cohort 'LOOCV on Cohort 1'"
        ),
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to loocv_predictions.tsv (output of mesa_loocv.py).",
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Output directory for PDF figures.",
    )
    parser.add_argument(
        "--cohort", default="LOOCV",
        help="Cohort label shown in figure titles. Default: 'LOOCV'.",
    )
    parser.add_argument(
        "--dpi", type=int, default=300,
        help="Output resolution. Default: 300.",
    )
 
    cli = parser.parse_args()
 
    if not os.path.exists(cli.input):
        import sys
        sys.exit(f"ERROR: input file not found: {cli.input}")
 
    print(f"[mesa_cv_plot] Loading: {cli.input}")
    pred_df = pd.read_csv(cli.input, sep="\t", index_col=0)
    print(f"[mesa_cv_plot] Shape: {pred_df.shape}  "
          f"Columns: {pred_df.columns.tolist()}")
 
    plot_mesa_loocv(
        pred_df,
        output_dir   = cli.output,
        cohort_label = cli.cohort,
        dpi          = cli.dpi,
    )
 