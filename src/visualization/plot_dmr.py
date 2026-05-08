"""DMR volcano plot from annotated DMR bed file."""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from adjustText import adjust_text

mpl.rcParams.update({
    "font.family": "sans-serif", "font.size": 8,
    "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7,
})

HYPER_COLOR = "#cc2936"
HYPO_COLOR  = "#08415c"
NS_COLOR    = "lightgrey"
_NONCODING  = r'^(LOC|LINC|MIR)\d'


def _save(fig, png_path, pdf_path):
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_dmr] saved → {png_path}")


def plot_dmr_volcano(
    df, group_a="GroupA", group_b="GroupB",
    png_path="dmr_volcano.png", pdf_path="dmr_volcano.pdf",
    q_thr=0.05, effect_thr=None, top_n=20,
    xlim=None, ylim=None, y_cap=None,
    figsize=(5, 4),
):
    # filter to promoter regions, drop noncoding
    bg = (
        df
        .query("`annot.type` == 'hg38_genes_promoters'")
        .dropna(subset=["annot.symbol", "q_value", "mean_diff"])
        .loc[lambda x: ~x["annot.symbol"].str.match(_NONCODING)]
        .loc[lambda x: ~x["annot.symbol"].str.endswith("AS1")]
        .sort_values("q_value")
        .drop_duplicates(subset=["annot.symbol"])
        .reset_index(drop=True)
        .copy()
    )
    if bg.empty:
        print("[plot_dmr] WARNING: no promoter DMRs found, skipping.")
        return

    sig = bg.query("q_value < @q_thr").copy()
    print(f"[plot_dmr] background={len(bg)}, significant={len(sig)}")

    if effect_thr is None:
        effect_thr = float(np.percentile(sig["mean_diff"].abs(), 10)) if len(sig) else 0.0

    # -log10(q)
    for frame in [bg, sig]:
        nz = frame.loc[frame["q_value"] > 0, "q_value"].min()
        frame["q_value"] = frame["q_value"].replace(0, nz)
        frame["_nlq"] = -np.log10(frame["q_value"])

    if y_cap is None:
        y_cap = float(np.ceil(bg["_nlq"].max() / 10) * 10) or 10.0
    for frame in [bg, sig]:
        frame["_nlq_c"] = frame["_nlq"].clip(upper=y_cap)

    def _cls(row):
        if abs(row["mean_diff"]) > effect_thr:
            return "Hyper" if row["mean_diff"] > 0 else "Hypo"
        return "NS"

    sig["_dir"] = sig.apply(_cls, axis=1)
    sig["_abs"] = sig["mean_diff"].abs()
    n_hyper = (sig["_dir"] == "Hyper").sum()
    n_hypo  = (sig["_dir"] == "Hypo").sum()

    label_df = pd.concat([
        sig[sig["_dir"] == "Hyper"].nlargest(top_n, "_abs"),
        sig[sig["_dir"] == "Hypo"].nlargest(top_n, "_abs"),
    ])

    if xlim is None:
        xabs = bg["mean_diff"].abs().max()
        xlim = (-xabs * 1.15, xabs * 1.15)
    if ylim is None:
        ylim = (0, y_cap)

    sns.set_theme("paper", style="ticks")
    mpl.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(bg["mean_diff"], bg["_nlq_c"],
               s=4, alpha=0.25, color=NS_COLOR, edgecolor="none", zorder=1)

    for grp, color in [("Hypo", HYPO_COLOR), ("Hyper", HYPER_COLOR)]:
        sub = sig[sig["_dir"] == grp]
        if not sub.empty:
            ax.scatter(sub["mean_diff"], sub["_nlq_c"],
                       s=10, alpha=0.7, color=color, edgecolor="none", zorder=2)

    lkw = dict(linestyle="--", color="black", linewidth=0.4, alpha=0.5)
    ax.axvline(-effect_thr, **lkw)
    ax.axvline(effect_thr, **lkw)
    ax.axhline(-np.log10(q_thr), **lkw)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    q_tick = round(-np.log10(q_thr), 2)
    yticks = sorted(set([t for t in ax.get_yticks() if ylim[0] <= t <= ylim[1]] + [q_tick]))
    ax.set_yticks(yticks)
    ax.set_xlabel(f"Mean difference ({group_b} − {group_a})")
    ax.set_ylabel(r"$-\log_{10}(\mathrm{q})$")
    ax.set_title(f"{group_a}−{group_b}", pad=8)

    texts = []
    for _, row in label_df.iterrows():
        col = HYPER_COLOR if row["_dir"] == "Hyper" else HYPO_COLOR
        texts.append(ax.text(row["mean_diff"], row["_nlq_c"],
                             str(row["annot.symbol"]),
                             fontsize=5, color=col, va="center", ha="left"))
    if texts:
        adjust_text(texts, ax=ax,
                    arrowprops=dict(arrowstyle="-", color="grey", lw=0.4, alpha=0.6),
                    expand=(1.2, 1.4), force_text=(0.3, 0.5))

    legend_handles = [
        mlines.Line2D([], [], color="white", marker="o",
                      markerfacecolor=HYPER_COLOR, markersize=5,
                      label=f"Hyper (n={n_hyper})"),
        mlines.Line2D([], [], color="white", marker="o",
                      markerfacecolor=HYPO_COLOR, markersize=5,
                      label=f"Hypo  (n={n_hypo})"),
        mlines.Line2D([], [], color="white", marker="o",
                      markerfacecolor=NS_COLOR, markersize=5,
                      label=f"NS    (n={len(bg) - len(sig)})"),
    ]
    ax.legend(handles=legend_handles, frameon=False,
              loc="upper right", handletextpad=0.2, fontsize=7)

    fig.tight_layout()
    _save(fig, png_path, pdf_path)
