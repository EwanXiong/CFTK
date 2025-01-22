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
import time
from tqdm import tqdm

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
    "--all",
    dest="all",
    help="Calculate WPS for fragments with any length",
    action="store_true",
)

parser.add_option(
    "--short", dest="short", help="Calculate S-WPS(35-80bp)", action="store_true"
)

parser.add_option(
    "--long", dest="long", help="Calculate L-WPS(120-180bp)", action="store_true"
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
chrom_list = [_ for _ in chrom_list if _ in regions.iloc[:, 0].unique()]
core = options.core

#norm_factor = 1000000 / bamfile.count()
print("@%s \t%s" % (time.asctime(), 'Files loaded.'), file=sys.stderr)

def WPS_chrom(chrom="chr1", step=10, short=False, long=False):
    chrom_reads = Intersecter()
    chrom_regions = regions[regions[0] == chrom][[1, 2]].astype(int)
    bamfile = pysam.AlignmentFile(options.bam_path, "rb")
    for read in bamfile.fetch(chrom, multiple_iterators=True):
        chrom_reads.add_interval(Interval(read.reference_start, read.reference_end))
    bamfile.close()
    print("Read fetching done: %s" % chrom, file=sys.stderr)
    print("WPS calculation: %s" % chrom, file=sys.stderr)
    region_wps = []
    if short:
        for ra, rb in chrom_regions.values:
            single_pos_wps = []
            for pos in range(ra, rb + 1, step):
                endCount, comCount = 0, 0
                wa, wb = pos - windowSize, pos + windowSize
                for read in chrom_reads.find(wa, wb):
                    if (read.end-read.start+1)<=80 and (read.end-read.start+1)>=35:
                        if (read.start > wa) or (read.end < wb):
                            endCount += 1
                        else:
                            comCount += 1
                single_pos_wps.append(comCount - endCount)
            # region_wps.append(np.mean(single_pos_wps))
            region_wps.append((chrom, ra, rb, np.array(single_pos_wps)))
    elif long:
        for ra, rb in chrom_regions.values:
            single_pos_wps = []
            for pos in range(ra, rb + 1, step):
                endCount, comCount = 0, 0
                wa, wb = pos - windowSize, pos + windowSize
                for read in chrom_reads.find(wa, wb):
                    if (read.end-read.start+1)<=180 and (read.end-read.start+1)>=120:
                        if (read.start > wa) or (read.end < wb):
                            endCount += 1
                        else:
                            comCount += 1
                single_pos_wps.append(comCount - endCount)
            # region_wps.append(np.mean(single_pos_wps))
            region_wps.append((chrom, ra, rb, np.array(single_pos_wps)))
    else:
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
    
    print("WPS calculation done: %s" % chrom, file=sys.stderr)
    region_wps = pd.DataFrame(region_wps, columns=["chr", "start", "end", "WPS"])
    #region_wps["WPS"] = region_wps["WPS"] * norm_factor
    region_wps["mean_WPS"] = region_wps["WPS"].apply(lambda x: np.mean(x))
    return region_wps


if options.short:
    print("@%s \t%s" % (time.asctime(), 'Short WPS calculation starts.'), file=sys.stderr)
    all_chrom_sWPS = Parallel(n_jobs=core, verbose=10, backend="multiprocessing")(
        delayed(WPS_chrom)(chrom, step, short=True) for chrom in chrom_list
    )
    print("@%s \t%s" % (time.asctime(), 'Short WPS calculation completed.'), file=sys.stderr)
    print("@%s \t%s" % (time.asctime(), 'Saving to %s' %(options.outfile.rsplit(".", 1)[0]+ ".short."+ options.outfile.rsplit(".", 1)[1])), file=sys.stderr)
    pd.concat(all_chrom_sWPS).to_csv(
        options.outfile.rsplit(".", 1)[0]
        + ".short."
        + options.outfile.rsplit(".", 1)[1],
        sep="\t",
        index=False,
    )
if options.long:
    print("@%s \t%s" % (time.asctime(), 'Long WPS calculation starts.'), file=sys.stderr)
    all_chrom_lWPS = Parallel(n_jobs=core, verbose=10, backend="multiprocessing")(
        delayed(WPS_chrom)(chrom, step, long=True) for chrom in chrom_list
    )
    print("@%s \t%s" % (time.asctime(), 'Long WPS calculation completed.'), file=sys.stderr)
    print("@%s \t%s" % (time.asctime(), 'Saving to %s' %(options.outfile.rsplit(".", 1)[0]+ ".long."+ options.outfile.rsplit(".", 1)[1])), file=sys.stderr)
    pd.concat(all_chrom_lWPS).to_csv(
        options.outfile.rsplit(".", 1)[0]
        + ".long."
        + options.outfile.rsplit(".", 1)[1],
        sep="\t",
        index=False,
    )

if not (options.short or options.long):
    options.all = True
    
if options.all:
    all_chrom_WPS = Parallel(n_jobs=core, verbose=10, backend="multiprocessing")(
        delayed(WPS_chrom)(chrom, step) for chrom in chrom_list
    )
    pd.concat(all_chrom_WPS).to_csv(
        options.outfile.rsplit(".", 1)[0] + ".all." + options.outfile.rsplit(".", 1)[1],
        sep="\t",
        index=False,
    )
sys.exit(0)
