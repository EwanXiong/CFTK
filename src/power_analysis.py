import argparse
import os
from joblib import Parallel, delayed
from statsmodels.stats.power import tt_ind_solve_power
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from numpy import random
from scipy.stats import nbinom

sns.set_theme(context="talk", style="ticks")

parser = argparse.ArgumentParser(description="Power analysis")

parser.add_argument(
    "-s",
    "--sample-size",
    dest="total_sample_size",
    type=int,
    help="Total sample size for power analysis",
)
parser.add_argument(
    "-e",
    "--effect-size",
    dest="meth_diff",
    type=float,
    help="Effect size (methylation difference)",
)
parser.add_argument(
    "-o", "--output-dir", dest="output_dir", help="Directory to save output files"
)
parser.add_argument(
    "--step-size",
    dest="step",
    type=int,
    help="Step size for CpG site sampling",
    default=1000,
)
parser.add_argument(
    "--cpg-std",
    dest="cpg_std",
    type=str,
    default=os.getcwd() + "/twist_panel_std.pkl",
    help="Path to CpG standard deviation file",
)
parser.add_argument(
    "-p",
    dest="alpha",
    type=float,
    help="Significance threshold (default: Bonferroni-corrected for EWAS)",
)
parser.add_argument(
    "-@",
    dest="n_jobs",
    type=int,
    default=-1,
    help="Number of parallel jobs to run (-1 uses all available cores)",
)
parser.add_argument(
    "--ratio",
    dest="ratio",
    type=float,
    default=1,
    help="Case/control ratio (default: 1)",
)
parser.add_argument(
    "--depth",
    dest="depth",
    nargs="+",
    type=int,
    default=[5, 10, 20, 30],
    help="Read depths to analyze (default: 5 10 20 30)",
)
parser.add_argument(
    "--plot", dest="plot", action="store_true", help="Generate plots for power analysis"
)
parser.add_argument(
    "--plot-threshold",
    dest="plot_threshold",
    type=float,
    default=0.8,
    help="Power threshold for plots (default: 0.8)",
)
parser.add_argument(
    "--diff-min",
    dest="diff_min",
    type=float,
    help="Minimum methylation difference to analyze",
)
parser.add_argument(
    "--diff-max",
    dest="diff_max",
    type=float,
    help="Maximum methylation difference to analyze",
)
parser.add_argument(
    "--diff-num",
    dest="diff_num",
    type=int,
    default=5,
    help="Number of methylation differences to analyze (default: 5)",
)
parser.add_argument(
    "--sample-min", dest="sample_min", type=int, help="Minimum sample size to analyze"
)
parser.add_argument(
    "--sample-max", dest="sample_max", type=int, help="Maximum sample size to analyze"
)
parser.add_argument(
    "--sample-num",
    dest="sample_num",
    type=int,
    default=10,
    help="Number of sample sizes to analyze (default: 10)",
)

args = parser.parse_args()
cpg_std = pd.read_pickle(args.cpg_std)


def meth_adjust_on_read_depth(meth_ratio, read_depth, missing_proba):
    """
    Adjust methylation ratios based on read depth and simulate missing values.

    Args:
        meth_ratio: Array of methylation ratios
        read_depth: Array of read depths
        missing_proba: Probability of missing values

    Returns:
        Adjusted methylation ratios with simulated missing values
    """
    temp = np.array(
        [
            (
                np.argmin(np.abs((np.arange(rd + 1) / rd) - mr)) / rd
                if rd > 0
                else np.nan
            )
            for mr, rd in zip(meth_ratio, read_depth)
        ]
    )
    # simulate the missing values
    for idx in range(len(temp)):
        if random.random() <= missing_proba:
            temp[idx] = np.nan
    return temp


def get_nbimon_on_mean(mean_depth):
    """
    Get negative binomial parameters based on mean read depth.

    Args:
        mean_depth: Mean read depth

    Returns:
        mean_depth, k, n, p, missing_pct
    """
    k = 1.3409309326892476 * mean_depth + 3.526927718332715
    n = (mean_depth / k - 0.6291928395185754) / 0.02390549241316919
    p = n / (n + mean_depth)
    missing_pct = mean_depth * n * (-0.0007582750922856895) + 4.56905302776041
    return mean_depth, k, n, p, missing_pct / 100


def read_depth_on_nbinom_distribution(r, p, size, random_state=0):
    """
    Generate read depths based on negative binomial distribution.

    Args:
        r: Size parameter for negative binomial
        p: Probability parameter for negative binomial
        size: Number of read depths to generate
        random_state: Random seed

    Returns:
        Array of read depths
    """
    read_depth = nbinom.rvs(r, p, size=size, loc=0, random_state=random_state)
    return read_depth


def pwr_by_sample_and_diff(
    total_sample_size=200,
    meth_diff=0.1,
    step=1000,
    ratio=1,
    depth=[5, 10, 20, 30],
    n_jobs=-1,
):
    """
    Calculate power by sample size and methylation difference.

    Args:
        total_sample_size: Total sample size
        meth_diff: Methylation difference
        step: Step size for CpG site sampling
        ratio: Case/control ratio
        depth: Read depths to analyze
        n_jobs: Number of parallel jobs

    Returns:
        DataFrame of power values
    """
    sample_size = total_sample_size * ratio / (1 + ratio)
    alpha = 0.05 / 3771981  # rep_cpg_summary_mean.shape[0]
    power_all_depth = []

    for mean_depth in depth:
        cpg_std = cpg_std[f"{mean_depth}_std"]
        effect_sizes = meth_diff / cpg_std  # [cpg_std >= 0.01]

        if mean_depth >= 15:
            mu, k, r, p, missing_pct = get_nbimon_on_mean(mean_depth)
        else:
            mu, k, r, p, missing_pct = get_nbimon_on_mean(15)
            missing_pct = missing_pct + (
                nbinom.cdf((15 - mean_depth), r, p) - nbinom.cdf(0, r, p)
            )

        true_sample_size = round(sample_size * (1 - missing_pct))
        powers = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
            delayed(tt_ind_solve_power)(
                effect_size=es,
                nobs1=true_sample_size,
                ratio=ratio,
                alpha=alpha,
                alternative="two-sided",
            )
            for es in effect_sizes[::step]
        )
        power_all_depth.append(powers)

    return pd.DataFrame(
        power_all_depth,
        index=depth,
    ).T


def pwd_min_for_diffs(
    total_sample_size=200,
    step=30000,
    ratio=1,
    depth=[5, 10, 20, 30],
    n_jobs=-1,
    diff_range=[0, 0.1],
    diff_num=5,
    pwr_threshold=0.8,
):
    """
    Calculate proportion of CpGs with power above threshold for different methylation differences.

    Args:
        total_sample_size: Total sample size
        step: Step size for CpG site sampling
        ratio: Case/control ratio
        depth: Read depths to analyze
        n_jobs: Number of parallel jobs
        diff_range: Range of methylation differences to analyze [min, max]
        diff_num: Number of methylation differences to analyze
        pwr_threshold: Power threshold

    Returns:
        DataFrame of proportions
    """
    diff_to_inspect = np.linspace(diff_range[0], diff_range[1], diff_num)
    pwr_result = [
        pwr_by_sample_and_diff(
            total_sample_size=total_sample_size,
            step=step,
            n_jobs=n_jobs,
            depth=depth,
            meth_diff=diff,
            ratio=ratio,
        )
        for diff in diff_to_inspect
    ]
    return pd.DataFrame(
        [(_ >= pwr_threshold).sum() / len(_) for _ in pwr_result],
        index=diff_to_inspect,
    )


def pwd_min_for_samples(
    meth_diff=0.1,
    step=30000,
    ratio=1,
    depth=[5, 10, 20, 30],
    n_jobs=-1,
    sample_range=[50, 300],
    sample_num=10,
    pwr_threshold=0.8,
):
    """
    Calculate proportion of CpGs with power above threshold for different sample sizes.

    Args:
        meth_diff: Methylation difference
        step: Step size for CpG site sampling
        ratio: Case/control ratio
        depth: Read depths to analyze
        n_jobs: Number of parallel jobs
        sample_range: Range of sample sizes to analyze [min, max]
        sample_num: Number of sample sizes to analyze
        pwr_threshold: Power threshold

    Returns:
        DataFrame of proportions
    """
    sample_size_to_inspect = np.linspace(sample_range[0], sample_range[1], sample_num)
    pwr_result = [
        pwr_by_sample_and_diff(
            total_sample_size=sample_size,
            step=step,
            n_jobs=n_jobs,
            depth=depth,
            meth_diff=meth_diff,
            ratio=ratio,
        )
        for sample_size in sample_size_to_inspect
    ]
    return pd.DataFrame(
        [(_ >= pwr_threshold).sum() / len(_) for _ in pwr_result],
        index=sample_size_to_inspect,
    )


def power_analysis(
    total_sample_size=None,
    meth_diff=None,
    diff_min=None,
    diff_max=None,
    diff_num=None,
    sample_min=None,
    sample_max=None,
    sampele_num=None,
    depth=[5, 10, 20, 30],
    ratio=1,
    step=30000,
    alpha=0.05,
    n_jobs=-1,
):
    """
    Main power analysis function.

    Args:
        total_sample_size: Total sample size
        meth_diff: Methylation difference
        diff_min: Minimum methylation difference
        diff_max: Maximum methylation difference
        diff_num: Number of methylation differences
        sample_min: Minimum sample size
        sample_max: Maximum sample size
        sampele_num: Number of sample sizes
        depth: Read depths to analyze
        ratio: Case/control ratio
        step: Step size for CpG site sampling
        alpha: Significance threshold
        n_jobs: Number of parallel jobs

    Returns:
        Power analysis results
    """
    if total_sample_size is not None and meth_diff is not None:
        return pwr_by_sample_and_diff(
            total_sample_size=total_sample_size,
            meth_diff=meth_diff,
            step=step,
            ratio=ratio,
            depth=depth,
            n_jobs=n_jobs,
        )
    sample_size = total_sample_size / 2
    cpg_std = cpg_std[f"{meth_diff}_std"]
    effect_sizes = meth_diff / cpg_std
    powers = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(tt_ind_solve_power)(
            effect_size=es, nobs1=sample_size, alpha=alpha, alternative="two-sided"
        )
        for es in effect_sizes[::step]
    )
    return powers


def power_analyis_output(es_all, output_dir="."):
    """
    Save power analysis results to files and generate plots.

    Args:
        es_all: Power analysis results
        output_dir: Directory to save output files
    """
    f, ax = plt.subplots(figsize=(6, 5))
    sns.set_theme(context="talk", style="ticks")
    sns.ecdfplot(np.array(es_all), complementary=True, ax=ax)
    ax.set(xlabel="Effect size", ylabel="Proportion of CpG sites (%)")
    ax.figure.savefig(
        f"{output_dir}/power_analysis_ecdf_plot.png", bbox_inches="tight", dpi=500
    )

    len_x = len(es_all)
    y = [(es_all >= i).sum() / len_x for i in np.arange(0.0, 1.01, 0.01)]
    pd.DataFrame(
        [np.arange(0.0, 1.01, 0.01), y], index=["Effect size", "Prop."]
    ).T.round(4).to_csv(
        f"{output_dir}/power_analysis_effect_size_cumulative_dist.tsv",
        sep="\t",
        index=None,
    )


# Main execution
if args.total_sample_size is not None and args.meth_diff is not None:
    power_basic = pwr_by_sample_and_diff(
        total_sample_size=args.total_sample_size,
        meth_diff=args.meth_diff,
        step=args.step,
        ratio=args.ratio,
        depth=args.depth,
        n_jobs=args.n_jobs,
        pwr_threshold=args.plot_threshold,
    )
    power_basic.to_csv(
        f"{args.output_dir}/power_analysis_effect_size_cumulative_dist.tsv",
        sep="\t",
        index=None,
    )

    if args.plot:
        f, ax = plt.subplots(figsize=(6, 5))
        sns.ecdfplot(power_basic, complementary=True, ax=ax, legend=True)
        ax.set(
            xlabel="Minimal Power",
            ylabel="Prop. of CpG with power",
            title=f"Power analysis for different read depth\n{args.total_sample_size} Samples, Methylation diff. {args.meth_diff}",
        )
        sns.move_legend(
            ax,
            title="Avg. RD",
            loc="upper left",
            bbox_to_anchor=(1, 1),
            frameon=False,
            fontsize=15,
        )
        ax.axvline(args.plot_threshold, zorder=0, c="red", lw=1, ls="--", alpha=0.7)
        ax.figure.savefig(
            f"{args.output_dir}/power_analysis_effect_size_cumulative_plot.png",
            bbox_inches="tight",
            dpi=500,
        )

elif (
    args.total_sample_size is not None
    and args.diff_min is not None
    and args.diff_max is not None
):
    power_diff = pwd_min_for_diffs(
        total_sample_size=args.total_sample_size,
        diff_range=[args.diff_min, args.diff_max],
        step=args.step,
        ratio=args.ratio,
        depth=args.depth,
        n_jobs=args.n_jobs,
        pwr_threshold=args.plot_threshold,
        diff_num=args.diff_num,
    )
    power_diff.to_csv(
        f"{args.output_dir}/power_analysis_effect_size_by_diffs.tsv",
        sep="\t",
        index=None,
    )

    if args.plot:
        f, ax = plt.subplots(figsize=(6, 5))
        sns.lineplot(data=power_diff, markers=True, dashes=False)
        ax.set(
            xlabel="Mean methylation diff.",
            ylabel=f"CpG >= {args.plot_threshold:.1f} power (%)",
            title=f"Power analysis for different read depth\n{args.total_sample_size} Samples",
        )
        sns.move_legend(
            ax,
            title="Avg. RD",
            loc="upper left",
            bbox_to_anchor=(1, 1),
            frameon=False,
            fontsize=15,
        )
        ax.figure.savefig(
            f"{args.output_dir}/power_analysis_effect_size_by_diffs_plot.png",
            bbox_inches="tight",
            dpi=500,
        )

elif (
    args.meth_diff is not None
    and args.sample_min is not None
    and args.sample_max is not None
):
    power_sample = pwd_min_for_samples(
        meth_diff=args.meth_diff,
        step=args.step,
        ratio=args.ratio,
        depth=args.depth,
        n_jobs=args.n_jobs,
        pwr_threshold=args.plot_threshold,
        sample_range=[args.sample_min, args.sample_max],
        sample_num=args.sample_num,
    )
    power_sample.to_csv(
        f"{args.output_dir}/power_analysis_effect_size_by_samples.tsv",
        sep="\t",
        index=None,
    )

    if args.plot:
        f, ax = plt.subplots(figsize=(6, 5))
        sns.lineplot(data=power_sample, markers=True, dashes=False)
        ax.set(
            xlabel="Total sample size",
            ylabel=f"CpG >= {args.plot_threshold:.1f} power (%)",
            title=f"Power analysis for different read depth\n Methylation diff. {args.meth_diff}",
        )
        sns.move_legend(
            ax,
            title="Avg. RD",
            loc="upper left",
            bbox_to_anchor=(1, 1),
            frameon=False,
            fontsize=15,
        )
        ax.figure.savefig(
            f"{args.output_dir}/power_analysis_effect_size_by_samples_plot.png",
            bbox_inches="tight",
            dpi=500,
        )
