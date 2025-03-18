#!/usr/bin/env python3
# filepath: /Users/crchen/Wei Li Lab Dropbox/ChaoRong Chen/Research/Project/TWIST/src/region_average.py
"""
Author: ChaoRong Chen
Date: March 18, 2025

Region Average Calculation for TWIST Package
===========================================

This script calculates the average methylation values for genomic regions defined in a BED file.
It takes a methylation matrix as input, where rows are CpGs (format: chr_position) and 
columns are samples. For each region in the BED file, it computes either the mean value 
or both mean and standard deviation across all CpGs within that region.

Usage:
    python region_average.py -i matrix.tsv -r regions.bed -o output.tsv [-@ cores] [--std]

Arguments:
    -i, --input     : Path to methylation matrix file (TSV format with header)
    -r, --region    : Path to BED file containing regions of interest
    -@, --core      : Number of cores for parallel processing
    --std           : Include standard deviation in the output
    -o, --out       : Output file path

Output:
    A TSV file containing average methylation values for each region,
    with regions as rows (format: chr:start-end) and samples as columns.
    If --std is specified, additional columns with standard deviations are included.
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from optparse import OptionParser

parser = OptionParser()
parser.add_option(
    "-i",
    "--input",
    dest="input_maxtrix",
    help="path to the matrix file (default: 'matrix.tsv')",
    default="matrix.tsv",
)
parser.add_option(
    "-r",
    "--region",
    dest="region_path",
    help="The bed file for regions to calculate average values.",
)

parser.add_option(
    "-@",
    "--core",
    dest="core",
    help="The core number used for parallel computing (default -1)",
    default=-1,
    type="int",
)

parser.add_option(
    "--std", dest="std", help="Return standard deviation for each region", action="store_true"
)

parser.add_option(
    "-o",
    "--out",
    dest="outfile",
    help="Output file name for output (default: 'output.tsv')",
    default="output.tsv",
)
(options, args) = parser.parse_args()

matrix = pd.read_csv(
    options.input_maxtrix, header=0, index_col=0, sep="\t"
)  # Load your matrix here
regions = pd.read_csv(options.region_path, sep="\t", header=None)
regions.columns = ["chrom", "chromStart", "chromEnd"]
regions_all_chrom = regions["chrom"].unique()

matrix_index = pd.DataFrame([_.split("_")[:2] for _ in matrix.index])
keep = matrix_index.iloc[:, 1].astype(str).str.isnumeric().values
matrix = matrix.iloc[keep]
matrix_index = matrix_index.iloc[keep]
matrix_columns = matrix.columns

matrix_index.columns = ["chr", "cpg"]

matrix_index = matrix_index.astype({"cpg": int})


def cal_chrom_with_std(region_start, region_end):
    """
    Calculate mean and standard deviation of methylation values within a genomic region.
    
    Args:
        region_start (int): Start position of the region
        region_end (int): End position of the region
    
    Returns:
        tuple: Mean and standard deviation of methylation values
    """
    temp = matrix_chrom.iloc[
        (
            (matrix_index_chrom["cpg"] > region_start)
            & (matrix_index_chrom["cpg"] <= region_end)
        ).values
    ]
    if len(temp) == 0:
        return pd.Series(dtype=float)  # np.nan
    else:
        return temp.mean(axis=0), temp.std(axis=0)


def cal_chrom(region_start, region_end):
    """
    Calculate mean methylation values within a genomic region.
    
    Args:
        region_start (int): Start position of the region
        region_end (int): End position of the region
    
    Returns:
        pandas.Series: Mean methylation values
    """
    temp = matrix_chrom.iloc[
        (
            (matrix_index_chrom["cpg"] > region_start)
            & (matrix_index_chrom["cpg"] <= region_end)
        ).values
    ]
    if len(temp) == 0:
        return pd.Series(dtype=float)  # np.nan
    else:
        return temp.mean(axis=0)


calculation_func = cal_chrom_with_std if options.std else cal_chrom


average_result_output = pd.DataFrame([], columns=matrix_columns)
for chrom in regions_all_chrom:
    print(f"Processing {chrom}")
    matrix_chrom = matrix.iloc[np.where(matrix_index["chr"] == chrom)]
    matrix_index_chrom = matrix_index.iloc[np.where(matrix_index["chr"] == chrom)]
    regions_chrom = regions.loc[regions["chrom"] == chrom]
    average_result_chrom = Parallel(n_jobs=options.core, verbose=1, backend="multiprocessing")(
        delayed(calculation_func)(region[0], region[1])
        for idx, region in regions_chrom[["chromStart", "chromEnd"]].iterrows()
    )
    average_result_chrom = pd.DataFrame(
        average_result_chrom,
        index=regions_chrom.apply(
            lambda x: f"{chrom}:{x['chromStart']}-{x['chromEnd']}", axis=1
        ),
    )
    average_result_output = pd.concat([average_result_output, average_result_chrom])

average_result_output.to_csv(options.outfile, sep="\t", index=True)
