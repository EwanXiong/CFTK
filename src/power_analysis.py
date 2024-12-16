import argparse
import os
from joblib import Parallel, delayed
from statsmodels.stats.power import tt_ind_solve_power
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description="Power analysis")

parser.add_argument(
    "-s", "--sample-size", dest="sample_size", type=int, help="Sample size"
)
parser.add_argument(
    "-e", "--effect-size", dest="effect_size", type=float, help="Effect size"
)
parser.add_argument(
    "-o", "--output-dir", dest="output_dir", help="output_directory"
)
parser.add_argument("--step-size", dest="step_size", type=int, help="Alpha", default=1)
parser.add_argument(
    "--cpg-std",
    dest="cpg_std",
    type=str,
    default=os.getcwd() + "/twist_497sample_cpg_std.pkl",
)
parser.add_argument("-p", dest="alpha", type=float)
parser.add_argument("-@", dest="cores", type=int, default=-1)
args = parser.parse_args()
cpg_std = pd.read_pickle(args.cpg_std)


def power_calculator(es, sample_size, alpha):
    return tt_ind_solve_power(
        nobs1=sample_size / 2,
        ratio=1,
        effect_size=es,
        alternative="two-sided",
        alpha=alpha,
        power=None,
    )


def power_analysis(sample_size, effect_size, step_size, alpha=2.7050713203440227e-08):
    cpg_effect_size = effect_size / cpg_std[::step_size]
    return Parallel(n_jobs=args.cores, backend="multiprocessing", verbose=1)(
        delayed(power_calculator)(es, sample_size, alpha) for es in cpg_effect_size
    )


def power_analyis_output(es_all, output_dir="."):
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


es_all = power_analysis(args.sample_size, args.effect_size, args.step_size, args.alpha)
power_analyis_output(es_all, args.output_dir)
