"""
Author: Chaorong Chen
Date: 03/06/2025

Differential Analysis Module

This script performs statistical differential analysis between two groups using the Mann-Whitney U test.
It processes data matrices and identifies features that show significant differences between groups.

Usage:
    python differential_analysis.py -i input_matrix.tsv -a groupA -b groupB -o output.tsv

The input matrix should have features as rows and samples as columns, with sample names
prefixed by their group identifiers (e.g., "g1_sample1", "g2_sample2").
"""

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import argparse, sys
from joblib import Parallel, delayed
from util import disp

parser = argparse.ArgumentParser(
    description="""
    
Differential Analysis Module

This script performs statistical differential analysis between two groups using the Mann-Whitney U test.
It processes data matrices and identifies features that show significant differences between groups.

Usage:
    python differential_analysis.py -i input_matrix.tsv -a groupA -b groupB -o output.tsv

The input matrix should have features as rows and samples as columns, with sample names
prefixed by their group identifiers (e.g., "g1_sample1", "g2_sample2").
"""
)

parser.add_argument(
    "-a",
    "--groupA",
    dest="groupA",
    type=str,
    default=None,
    help="name of group A (default:'g1')",
)

parser.add_argument(
    "-b",
    "--groupB",
    dest="groupB",
    type=str,
    default=None,
    help="name of group B (default:'g2')",
)

parser.add_argument(
    "-i",
    "--input",
    dest="input_maxtrix",
    type=str,
    default="matrix.tsv",
    help="path to the matrix file (default: 'matrix.tsv')",
)

parser.add_argument(
    "-@", "--cores", dest="cores", type=int, help="Cores assigned.", default=-1
)

parser.add_argument(
    "-o",
    "--output",
    dest="output",
    type=str,
    default="differntial_analysis_output.tsv",
    help="output file name",
)

parser.add_argument(
    "--chunk",
    dest="chunk_size",
    type=int,
    default=500,
    help="chunk size for parallel processing (default: 500)",
)

matrix = pd.read_csv(
    parser.input_maxtrix, index_col=0, header=0
)  # Load your matrix here
chunk_size = parser.chunk_size
feature_size = matrix.shape[0]  # Use actual size rather than hardcoded value

if parser.groupA is None or parser.groupB is None:
    disp(
        "--groupA and --groupB are not specified. Identifying group names from the matrix header."
    )
    group_names = np.unique([_.split("_")[0] for _ in matrix.columns])
    disp(f"Identified group names: {group_names}")
    if len(group_names) != 2:
        disp(
            "Error: Group number > 2. Differential analysis only works for two groups comparison. --groupA and --groupB are not specified."
        )
        sys.exit(1)
    else:
        groupA = group_names[0]
        groupB = group_names[1]
        disp(f"Group A: {groupA}, Group B: {groupB}")
else:
    groupA = parser.groupA
    groupB = parser.groupB

# Filter the matrix for the two groups
matrix_A = matrix.filter(like=groupA, axis=1).T
matrix_B = matrix.filter(like=groupB, axis=1).T
output_index = matrix.index
del matrix


chunk_count = int(np.ceil(feature_size / chunk_size))


def compute_mann_whitney_pvalue(chunk_idx):
    """
    Compute Mann-Whitney U test p-values for a chunk of features.

    This function applies the Mann-Whitney U test to compare values between
    two groups for a specified chunk of features.

    Args:
        chunk_idx (int): Index of the chunk to process

    Returns:
        numpy.ndarray: Array of p-values from Mann-Whitney U tests
    """
    start_idx = chunk_size * chunk_idx
    end_idx = min(chunk_size * (chunk_idx + 1), feature_size)

    return mannwhitneyu(
        matrix_A.iloc[:, start_idx:end_idx],
        matrix_B.iloc[:, start_idx:end_idx],
        nan_policy="omit",
        alternative="two-sided",
    )[1]


# Store original p-values for comparison
pvalue_all_C9 = Parallel(n_jobs=parser.cores, backend="multiprocessing")(
    delayed(compute_mann_whitney_pvalue)(chunk_idx) for chunk_idx in range(chunk_count)
)

# Flatten the results
pd.DataFrame(
    np.hstack(pvalue_all_C9), columns=["MWU pvalue"], index=output_index
).to_csv(parser.output, sep="\t")
