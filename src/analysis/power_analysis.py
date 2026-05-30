"""Statistical power analysis for CpG methylation studies."""

import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import nbinom
from statsmodels.stats.power import NormalIndPower, TTestIndPower


def _get_nbinom_params(mean_depth):
    """Fit negative-binomial parameters for given mean read depth."""
    k = 1.3409309326892476 * mean_depth + 3.526927718332715
    n = (mean_depth / k - 0.6291928395185754) / 0.02390549241316919
    p = n / (n + mean_depth)
    missing = mean_depth * n * (-0.0007582750922856895) + 4.56905302776041
    return n, p, missing / 100


def _power_at_depth(mean_depth, meth_diff, cpg_std, sample_size, ratio, alpha, step):
    """Compute per-CpG power for a single read depth."""
    if mean_depth >= 15:
        r, p, miss = _get_nbinom_params(mean_depth)
    else:
        r, p, miss = _get_nbinom_params(15)
        miss += float(nbinom.cdf(15 - mean_depth, r, p) - nbinom.cdf(0, r, p))

    eff_n = round(sample_size * ratio / (1 + ratio) * (1 - miss))
    solver = NormalIndPower().solve_power if eff_n > 3 else TTestIndPower().solve_power

    results = []
    for col in [f"{mean_depth}_mean", f"{mean_depth}_CI_l", f"{mean_depth}_CI_u"]:
        std = cpg_std[col] if col in cpg_std else cpg_std.iloc[:, 0]
        es  = meth_diff / std.replace(0, np.nan).dropna()
        pwr = [
            solver(effect_size=e, nobs1=eff_n, ratio=ratio,
                   alpha=alpha, alternative="two-sided")
            for e in es.iloc[::step].values
        ]
        results.append(pwr)
    return pd.DataFrame(results,
                        index=[f"{mean_depth}_mean",
                               f"{mean_depth}_CI_l",
                               f"{mean_depth}_CI_u"]).T


def run_power(args):
    """Compute power curves and save TSV + plots."""
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.cpg_std):
        print(f"[power] WARNING: cpg_std file not found: {args.cpg_std}")
        return

    cpg_std = pd.read_pickle(args.cpg_std)
    alpha   = 0.05 / 3_771_981   # Bonferroni for TWIST panel

    depths  = getattr(args, "depth", [10, 20, 50])
    results = [
        _power_at_depth(d, args.effect_size, cpg_std,
                        args.sample_size, args.ratio, alpha, args.step_size)
        for d in depths
    ]

    # save cumulative power table
    out_tsv = os.path.join(args.output_dir, "power_cumulative.tsv")
    pd.concat(results, axis=1).to_csv(out_tsv, sep="\t")
    print(f"[power] saved → {out_tsv}")
    return results
