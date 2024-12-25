#!/usr/bin/env python

"""
:Author: Chaorong Chen
:Date: 06/06/2024
"""

import sys, os
from optparse import OptionParser
from bx.intervals.intersection import Intersecter, Interval
import pysam
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import pickle

parser = OptionParser()
parser.add_option(
    "-b",
    "--bam",
    dest="bam_path",
    help="The file for sequenced fragments in bed4 format.",
)
parser.add_option(
    "-r",
    "--region",
    dest="region_path",
    help="The bed file for regions to calculate WPS.",
)

parser.add_option(
    "-w",
    "--window",
    dest="windowSize",
    help="The window size used for calculate WPS (default 120)",
    default=120,
    type="int",
)

parser.add_option(
    "--wpsstep",
    dest="wpsstep",
    help="The step width used for calculate WPS (default 10)",
    default=10,
    type="int",
)

parser.add_option(
    "-t",
    "--core",
    dest="core",
    help="The core number used for parallel computing (default 1)",
    default=1,
    type="int",
)

parser.add_option(
    "--mean", dest="mean", help="Return mean WPS for each region", action="store_true"
)

parser.add_option(
    "-o",
    "--out",
    dest="outfile",
    help="Output file name for WPS in wig format",
    default="OFF",
)
(options, args) = parser.parse_args()

"""
Read bed file for sequencing fragments
"""
bamfile = pysam.AlignmentFile(options.bam_path, "rb")
regions = pd.read_csv(options.region_path, sep="\t", header=None)
windowSize = options.windowSize // 2
step = options.wpsstep
chrom_list = ["chr" + str(_) for _ in list(range(1, 23)) + ["X", "Y"]]
core = options.core

norm_factor = 1000000 / bamfile.count()


def WPS_chrom(chrom="chr1", step=10):
    chrom_reads = Intersecter()
    chrom_regions = regions[regions[0] == chrom][[1, 2]].astype(int)
    if len(chrom_regions) == 0:
        return []
    for read in bamfile.fetch(chrom, multiple_iterators=True):
        chrom_reads.add_interval(Interval(read.reference_start, read.reference_end))
    print("Read fetching done: %s" % chrom)
    region_wps = []
    for ra, rb in chrom_regions.values:
        single_pos_wps = []
        for pos in range(ra, rb + 1, step):
            endCount, comCount = 0, 0
            wa, wb = pos - windowSize, pos + windowSize
            for read in chrom_reads.find(wa, wb):
                if (read.start > wa) or (read.end < wb):
                    endCount += 1
                else:
                    comCount += 1
            single_pos_wps.append(comCount - endCount)
        # region_wps.append(np.mean(single_pos_wps))
        region_wps.append((chrom, ra, rb, np.array(single_pos_wps)))
    print("WPS calculation done: %s" % chrom)
    region_wps = pd.DataFrame(region_wps, columns=["chr", "start", "end", "WPS"])
    region_wps["WPS"] = region_wps["WPS"] * norm_factor
    region_wps["mean_WPS"] = region_wps["WPS"].apply(lambda x: np.mean(x))
    return region_wps


all_chrom_WPS = Parallel(n_jobs=core, verbose=1, backend="multiprocessing")(
    delayed(WPS_chrom)(chrom, step) for chrom in chrom_list
)
pd.concat(all_chrom_WPS).to_csv(options.outfile, sep="\t", index=False)
