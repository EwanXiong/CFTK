import argparse
import os
import sys
from joblib import Parallel, delayed
from statsmodels.stats.power import tt_ind_solve_power
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.stats.power import NormalIndPower, TTestIndPower
from numpy import random
from scipy.stats import nbinom

sns.set_theme(context="talk", style="ticks")

parser = argparse.ArgumentParser(description="Power analysis")

parser.add_argument(
    "-m",
    "--mode",
    dest="mode",
    type=int,
    choices=[1, 2, 3],
    help="Mode of power analysis: 1 for a fixed methylation difference and different sample sizes, 2 for a fixed sample size and different methylation, 3. for a fix sample size and a fixed methylation difference",
)

parser.add_argument(
    "-s",
    "--sample-size",
    dest="sample_size",
    type=int,
    help="Total sample size for power analysis",
)

parser.add_argument(
    "-e",
    "--meth-diff",
    dest="meth_diff",
    type=float,
    help="Effect size (methylation difference)",
)

parser.add_argument(
    "--sample-range",
    dest="sample_range",
    type=int,
    nargs=2,
    help="Sample size range for mode 1",
)

parser.add_argument(
    "--n-sample-size",
    dest="n_sample_size",
    type=int,
    default=5,
    help="Number of sample size points to analyze within the specified range",
)

parser.add_argument(
    "--diff-range",
    dest="diff_range",
    type=float,
    nargs=2,
    help="Methylation difference range for mode 2",
    default=[10, 20, 50],
    help="Read depths to analyze (default: [10, 20, 50])",
)

parser.add_argument(
    "--n-meth-diff",
    dest="n_meth_diff",
    type=int,
    default=5,
    help="Number of methylation difference points to analyze within the specified range (default: 5)",
)

parser.add_argument(
    "-o", "--output-dir", dest="output_dir", help="Directory to save output files"
)
parser.add_argument(
    "--step-size",
    dest="step",
    type=int,
    help="Step size for CpG site sampling to reduce computation burden(default: 10000)",
    default=10000,
)

parser.add_argument(
    "--ci",
    dest="ci",
    action="store_true",
    help="Calculate 95% confidence intervals for power analysis",
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
    default=0.05,
    help="Significance threshold (default: 0.05)",
)

parser.add_argument(
    "--ratio",
    dest="ratio",
    type=float,
    default=1,
    help="Case/control ratio (default: 1)",
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


args = parser.parse_args()
cpg_std = pd.read_pickle(args.cpg_std)


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
    step=5000,
    ratio=1,
    n_jobs=-1,
    depth=[5, 10, 20, 30],
):
    sample_size = total_sample_size * ratio / (1 + ratio)
    alpha = 0.05 / 3771981  # rep_cpg_summary_mean.shape[0]
    power_all_depth = []
    for mean_depth in depth:  # [3, 10, 15, 25 , 50, 75, 90, 100]:

        # mean_depth = 15
        # mu, k, r, p, missing_pct = get_nbimon_on_mean(mean_depth)
        if mean_depth >= 15:
            mu, k, r, p, missing_pct = get_nbimon_on_mean(mean_depth)
        else:
            mu, k, r, p, missing_pct = get_nbimon_on_mean(15)
            missing_pct = missing_pct + (
                nbinom.cdf((15 - mean_depth), r, p) - nbinom.cdf(0, r, p)
            )
        true_sample_size = round(sample_size * (1 - missing_pct))
        # true_sample_size = true_sample_size if true_sample_size !=95 else true_sample_size+1
        # modify this to fix the issue of missing power output for specific sample sizes(appear after pacakge updates)
        solve_power = (
            NormalIndPower().solve_power
            if true_sample_size > 3
            else TTestIndPower().solve_power
        )
        poewr_95ci = []
        for col in [
            f"{mean_depth}_mean",
            f"{mean_depth}_CI_l",
            f"{mean_depth}_CI_u",
        ]:
            cpg_std = cpg_std_summary[col]
            effect_sizes = meth_diff / cpg_std  # [cpg_std >= 0.01]
            powers = [
                solve_power(
                    effect_size=es,
                    nobs1=true_sample_size,
                    ratio=ratio,
                    alpha=alpha,
                    alternative="two-sided",
                )
                for es in effect_sizes[::step].values
            ]
            poewr_95ci.append(powers)
        # cpg_detectable_diff = effect_size * cpg_std#[cpg_std <= 0.01]
        power_all_depth.append(
            pd.DataFrame(
                poewr_95ci,
                index=[
                    f"{mean_depth}_mean",
                    f"{mean_depth}_CI_l",
                    f"{mean_depth}_CI_u",
                ],
            ).T
        )
    return power_all_depth
    # sns.ecdfplot(required_mean_diff.values, complementary=False)


def ecdf(data):
    """
    Compute the empirical cumulative distribution function for a dataset.

    This function extends the ECDF to the full range [0,1] by adding points at
    the extremes of the data range.

    Args:
        data: numpy array or list of values

    Returns:
        x: sorted values from data with extended endpoints
        y: corresponding cumulative probabilities from 0 to 1
    """
    x = np.sort(data)
    y = np.arange(1, len(x) + 1) / len(x)
    # Extend x with values at the ends
    x = np.concatenate((x, [x[-1]], [1]))

    # Extend y to range [0,1]
    y = np.concatenate((y, [1], [1]))

    return x, y


def pwd_by_diff(
    total_sample_size=200,
    step=10000,
    ratio=1,
    depth=[3, 10, 20, 30],
    n_jobs=-1,
    diff_range=[0, 0.1],
    diff_num=5,
    pwr_threshold=0.8,
):
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
    temp = pd.concat(
        [
            pd.concat(
                [(_ >= pwr_threshold).sum() / len(_) for _ in test_diff],
                axis=0,
            )
            for test_diff in pwr_result
        ],
        axis=1,
    ).T
    temp.index = diff_to_inspect
    return temp


def pwd_by_sample(
    meth_diff=0.1,
    step=10000,
    ratio=1,
    depth=[10, 20, 30],
    n_jobs=-1,
    sample_range=[50, 300],
    sample_num=10,
    pwr_threshold=0.8,
):
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
    temp = pd.concat(
        [
            pd.concat(
                [(_ >= pwr_threshold).sum() / len(_) for _ in test_sample],
                axis=0,
            )
            for test_sample in pwr_result
        ],
        axis=1,
    ).T
    temp.index = sample_size_to_inspect
    return temp

# By mode
if args.mode == 1:
    assert (
        args.sample_size is not None
        and args.meth_diff is not None
    ), "For mode 1, --sample-size and --meth-diff must be specified."
    sys.exit(1)

if args.mode == 2:
    assert ( 
        args.sample_size is not None
        and args.diff_range is not None
    ), "For mode 2, --sample-size and --diff-range must be specified."
    sys.exit(1)

if args.mode == 3:
    assert (
        args.meth_diff is not None
        and args.sample_range is not None
    ), "For mode 3, --meth-diff and --sample-range must be specified."
    sys.exit(1)


# Automatically select the mode based on the provided arguments
if args.sample_size is not None and args.meth_diff is not None:
    pwr_result = pwr_by_sample_and_diff(
        total_sample_size=args.sample_size,
        meth_diff=args.meth_diff,
        step=args.step,
        ratio=args.ratio,
        depth=args.depth,
    )
    pwr_result.to_csv(
        f"{args.output_dir}/power_analysis_effect_size_cumulative_dist.tsv",
        sep="\t",
        index=None,
    )
    if args.plot:
        f, ax = plt.subplots(figsize=(6, 5))
        sns.set_theme(context="talk", style="ticks")
        for i, data in enumerate(pwr_result):
            x1, y1 = ecdf(data.iloc[:, 0].values)
            x2, y2 = ecdf(data.iloc[:, 1].values)
            x3, y3 = ecdf(data.iloc[:, 2].values)
            y1, y2, y3 = 1 - y1, 1 - y2, 1 - y3
            line = sns.lineplot(
                x=x1, y=y1, label=f"{data.columns[0].split('_')[0]}", estimator=None
            )
            line_color = line.get_lines()[-1].get_color()
            # Use the same color for the fill_betweenx with lower alpha
            ax.fill_betweenx(
                y3, x2, x3, color=line_color, alpha=0.1, interpolate=True, lw=0
            )
        ax.set(
            xlabel="Minimal Power",
            ylabel="Prop. of CpG with power",
            title=f"Power analysis for different read depth\n{args.sample_size} Samples, Methylation diff. {args.meth_diff}",
        )
        sns.move_legend(
            ax,
            title="Avg. read depth",
            loc="upper left",
            bbox_to_anchor=(1, 1),
            frameon=False,
            fontsize=15,
        )
        ax.axvline(args.plot_threshold, zorder=0, c="red", lw=1, ls="--", alpha=0.7)
        f.savefig(
            f"{args.output_dir}/m1_power_analysis_basic_cumulative_plot_{args.sample_size}smp_{args.meth_diff}diff.png",
            bbox_inches="tight",
            dpi=500,
        )

if args.sample_size is not None and args.diff_range is not None:
    pwr_result_by_diff = pwd_by_diff(
        total_sample_size=args.sample_size,
        step=args.step,
        ratio=args.ratio,
        depth=args.depth,
        diff_range=args.diff_range,
        diff_num=args.n_meth_diff,
        pwr_threshold=args.plot_threshold,
    )
    pwr_result_by_diff.to_csv(
        f"{args.output_dir}/power_analysis_effect_size_by_diffs.tsv",
        sep="\t",
        index=None,
    )
    if args.plot:
        f, ax = plt.subplots(figsize=(6, 5))
        sns.set_theme(context="talk", style="ticks")
        for idx, d in enumerate(args.depth):
            data = pwr_result_by_diff.iloc[:, 3 * idx : 3 * (idx + 1)]
            x = data.index
            y1 = data.iloc[:, 0]
            y2 = data.iloc[:, 1]
            y3 = data.iloc[:, 2]
            line = sns.lineplot(
                x=x, y=y1, label=f"{data.columns[0].split('_')[0]}", estimator=None
            )
            # Get the color of the line
            line_color = line.get_lines()[-1].get_color()
            # Use the same color for the fill_betweenx with lower alpha
            ax.fill_between(
                x, y2, y3, color=line_color, alpha=0.1, interpolate=True, lw=0
            )
        ax.set(
            xlabel="Mean methylation diff.",
            ylabel=f"CpG >= {args.plot_threshold} power (%)",
            title=f"Power analysis for different read depth\n{args.sample_size} Samples",
        )
        sns.move_legend(
            ax,
            title="Avg. read depth",
            loc="upper left",
            bbox_to_anchor=(1, 1),
            frameon=False,
            fontsize=15,
        )
        f.savefig(
            f"{args.output_dir}/m2_power_analysis_by_diff_{args.sample_size}smp.png",
            bbox_inches="tight",
            dpi=500,
        )  
    
    
    
if args.meth_diff is not None and args.sample_range is not None:
    pwr_result_by_sample = pwd_by_sample(
        meth_diff=args.meth_diff,
        step=args.step,
        ratio=args.ratio,
        depth=args.depth,
        sample_range=args.sample_range,
        sample_num=args.n_sample_size,
        pwr_threshold=args.plot_threshold,
    )
    pwr_result_by_sample.to_csv(
        f"{args.output_dir}/power_analysis_effect_size_by_samples.tsv",
        sep="\t",
        index=None,
    )
    if args.plot:
        f, ax = plt.subplots(figsize=(6, 5))
        sns.set_theme(context="talk", style="ticks")
        for idx, d in enumerate(args.depth):
            data = pwr_result_by_sample.iloc[:, 3 * idx : 3 * (idx + 1)]
            x = data.index
            y1 = data.iloc[:, 0]
            y2 = data.iloc[:, 1]
            y3 = data.iloc[:, 2]
            line = sns.lineplot(
                x=x, y=y1, label=f"RD {data.columns[0].split('_')[0]}", estimator=None
            )
            # Get the color of the line
            line_color = line.get_lines()[-1].get_color()
            # Use the same color for the fill_betweenx with lower alpha
            ax.fill_between(
                x, y2, y3, color=line_color, alpha=0.1, interpolate=True, lw=0
            )
        ax.set(
            xlabel="Total sample size",
            ylabel=f"CpG >= {args.plot_threshold} power (%)",
            title=f"Power analysis for different read depth\n Methylation diff. {args.meth_diff}",
        )
        sns.move_legend(
            ax,
            title="Avg. read depth",
            loc="upper left",
            bbox_to_anchor=(1, 1),
            frameon=False,
            fontsize=15,
        )
        f.savefig(
            f"{args.output_dir}/m3_power_analysis_by_sample_{args.meth_diff}diff.png",
            bbox_inches="tight",
            dpi=500,
        )  






