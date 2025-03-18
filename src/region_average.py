import numpy as np
import pandas as pd
import pickle
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
    help="The core number used for parallel computing (default 1)",
    default=1,
    type="int",
)

parser.add_option(
    "--std", dest="std", help="Return mean WPS for each region", action="store_true"
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
    parser.input_maxtrix, header=0, index_col=0, sep="\t"
)  # Load your matrix here
regions = pd.read_csv(options.region_path, sep="\t", header=None)
regions.columns = ["chrom", "chromStart", "chromEnd"]
regions_chrom = regions["chrom"].unique()

matrix_index = pd.DataFrame([_.split("_")[:2] for _ in matrix.index])
keep = matrix_index.iloc[:, 1].astype(str).str.isnumeric().values
matrix = matrix.iloc[keep]
matrix_index = matrix_index.iloc[keep]
matrix_columns = matrix.columns

matrix_index.columns = ["chr", "cpg"]

matrix_index = matrix_index.astype({"cpg": int})


if options.std:

    def cal_chrom(region_start, region_end):
        temp = matrix_chrom.iloc[
            (
                (matrix_index_chrom["cpg"] > region_start)
                & (matrix_index_chrom["cpg"] <= region_end)
            )
        ].values
        return np.nanmean(temp), np.nanstd(temp)

else:

    def cal_chrom(region_start, region_end):
        return matrix_chrom.iloc[
            (
                (matrix_index_chrom["cpg"] > region_start)
                & (matrix_index_chrom["cpg"] <= region_end)
            ).values
        ].mean()


average_result_output = pd.DataFrame([],columns=matrix_columns)

for chrom in regions_chrom:
    print(f"Processing {chrom}")
    matrix_chrom = matrix_index.loc[matrix_index["chr"] == chrom]
    matrix_index_chrom = matrix_index.loc[matrix_index["chr"] == chrom]
    regions_chrom = regions.loc[regions["chrom"] == chrom]
    average_result_chrom = Parallel(n_jobs=-1, verbose=10, backend="multiprocessing")(
        delayed(cal_chrom)(region[0], region[1])
        for idx, region in regions_chrom[["chromStart", "chromEnd"]].iterrows()
    )
    average_result_chrom = pd.DataFrame(
        average_result_chrom,
        columns=matrix_columns,
        index=regions_chrom.apply(
            lambda x: f"{chrom}:{x['chromStart']}-{x['chromEnd']}", axis=1
        ),
    )
    average_result_output = pd.concat([average_result_output, average_result_chrom])

average_result_output.to_csv(args.outfile, sep="\t", index=True)
