"""MESA visualization: ROC curves, probability heatmap, Spearman correlation."""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, roc_curve
import re

mpl.rcParams.update({
    "font.family": "sans-serif", "font.size": 7,
    "axes.titlesize": 8, "axes.labelsize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 6,
    "legend.fontsize": 6,
})

MULTIMODAL_COLOR = "#cc0000"
MOD_COLORS = ["#9b59b6", "#3498db", "#2ecc71", "#e67e22",
               "#1abc9c", "#e74c3c", "#34495e", "#f39c12"]


def _save(fig, png_path, pdf_path):
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_mesa] saved → {png_path}")


def _color_map(names):
    cmap, idx = {}, 0
    for n in names:
        if n == "Multimodal":
            cmap[n] = MULTIMODAL_COLOR
        else:
            cmap[n] = MOD_COLORS[idx % len(MOD_COLORS)]
            idx += 1
    return cmap


def _infer_group_names(y_true, sample_index):
    """Infer group names from sample IDs (prefix before trailing _digit)."""
    pos, neg = set(), set()
    for sid, lbl in zip(sample_index, y_true):
        m = re.match(r'^(.+?)(?:_\d+)?$', str(sid))
        prefix = m.group(1) if m else str(sid)
        (pos if lbl == 1 else neg).add(prefix)
    return (", ".join(sorted(pos)) or "Positive",
            ", ".join(sorted(neg)) or "Negative")


# ── ROC ───────────────────────────────────────────────────────────────────────

def plot_roc(pred_df, png_path, pdf_path, title="LOOCV"):
    y_true  = pred_df["y_true"].values
    modals  = [c for c in pred_df.columns if c != "y_true"]
    cmap    = _color_map(modals)

    auc_scores = {n: roc_auc_score(y_true, pred_df[n].values) for n in modals}
    order      = sorted(modals, key=lambda n: auc_scores[n])

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    for n in order:
        fpr, tpr, _ = roc_curve(y_true, pred_df[n].values)
        lw = 1.8 if n == "Multimodal" else 1.2
        ax.plot(fpr, tpr, color=cmap[n], linewidth=lw,
                label=f"{auc_scores[n]:.4f} ({n})")

    ax.plot([0, 1], [0, 1], color="grey", linewidth=0.5, linestyle="--", alpha=0.6)
    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.8)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1],
              title="AUC", frameon=False, loc="lower right",
              handlelength=1.0, fontsize=6, title_fontsize=6)

    fig.tight_layout()
    _save(fig, png_path, pdf_path)


# ── Probability heatmap ───────────────────────────────────────────────────────

def plot_prob_heatmap(pred_df, png_path, pdf_path, title="LOOCV"):
    y_true   = pred_df["y_true"].values
    modals   = [c for c in pred_df.columns if c not in ("y_true", "Multimodal")]
    label1, label0 = _infer_group_names(y_true, pred_df.index)

    sort_idx = np.argsort(-y_true)
    n_pos    = int(y_true.sum())
    n_total  = len(y_true)

    POS_COLOR = "#8b1a1a"
    NEG_COLOR = "#d3d3d3"

    n_rows      = 1 + len(modals)
    row_ratios  = [0.25] + [1.0] * len(modals)
    hm_height   = max(0.4 + len(modals) * 0.55, 2.0)

    fig = plt.figure(figsize=(9, hm_height))
    gs_outer = gridspec.GridSpec(1, 2, figure=fig,
                                 width_ratios=[0.88, 0.12], wspace=0.02)
    inner_gs = gridspec.GridSpecFromSubplotSpec(
        n_rows, 1, subplot_spec=gs_outer[0],
        hspace=0.05, height_ratios=row_ratios,
    )

    # condition bar
    ax_cond = fig.add_subplot(inner_gs[0])
    ccolors = [POS_COLOR] * n_pos + [NEG_COLOR] * (n_total - n_pos)
    ccolors = [ccolors[i] for i in sort_idx]
    ax_cond.bar(np.arange(n_total), [1] * n_total,
                color=ccolors, width=1.0, align="edge")
    ax_cond.set_xlim(0, n_total)
    ax_cond.set_ylim(0, 1)
    ax_cond.set_yticks([])
    ax_cond.set_xticks([])
    ax_cond.set_ylabel("Conditions", fontsize=5, rotation=0, labelpad=40, va="center")
    ax_cond.set_title(title, fontsize=8, pad=4)

    cond_patches = [
        Patch(facecolor=POS_COLOR, edgecolor="none", label=label1),
        Patch(facecolor=NEG_COLOR, edgecolor="none", label=label0),
    ]

    # heatmap rows
    last_im = None
    for i, name in enumerate(modals):
        ax_hm = fig.add_subplot(inner_gs[i + 1])
        probs = pred_df[name].values[sort_idx].reshape(1, -1)
        last_im = ax_hm.imshow(probs, aspect="auto", cmap="RdBu_r",
                                vmin=0, vmax=1, interpolation="nearest")
        ax_hm.set_yticks([0])
        ax_hm.set_yticklabels([name], fontsize=5)
        ax_hm.set_xticks([])

    # right panel: condition legend + colorbar
    ax_leg = fig.add_subplot(gs_outer[1])
    ax_leg.axis("off")
    ax_leg.legend(handles=cond_patches, loc="upper left",
                  bbox_to_anchor=(0.0, 1.0), frameon=False,
                  fontsize=6, title="Conditions", title_fontsize=6,
                  labelspacing=0.4, handlelength=1.0)

    pos = ax_leg.get_position()
    cbar_ax = fig.add_axes([pos.x0 + 0.01, pos.y0 + 0.05,
                             0.018, pos.height * 0.55])
    cbar = fig.colorbar(last_im, cax=cbar_ax,
                        ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.ax.tick_params(labelsize=5)
    cbar.set_label("Probability", fontsize=5, labelpad=3)

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_mesa] saved → {png_path}")


# ── Spearman correlation ──────────────────────────────────────────────────────

def plot_spearman(pred_df, png_path, pdf_path, title="LOOCV"):
    modals = [c for c in pred_df.columns if c not in ("y_true", "Multimodal")]
    n      = len(modals)
    mat    = np.eye(n)
    for i in range(n):
        for j in range(i):
            r, _ = spearmanr(pred_df[modals[i]].values, pred_df[modals[j]].values)
            mat[i, j] = mat[j, i] = r

    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    sz   = max(n * 1.2, 3.0)
    fig, ax = plt.subplots(figsize=(sz, sz))

    sns.heatmap(mat, ax=ax, mask=mask, annot=True, fmt=".2f",
                annot_kws={"size": 7, "weight": "bold"},
                cmap="YlOrRd", vmin=0, vmax=1,
                linewidths=0.5, linecolor="white", square=True,
                cbar_kws={"shrink": 0.6, "label": "Spearman r"},
                xticklabels=modals, yticklabels=modals)

    ax.set_title(title, pad=6)
    ax.tick_params(axis="x", labelrotation=30)
    ax.tick_params(axis="y", labelrotation=0)

    fig.tight_layout()
    _save(fig, png_path, pdf_path)
