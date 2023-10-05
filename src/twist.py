#!/usr/bin/env python3

import argparse, time, sys
import util

parser = argparse.ArgumentParser(description="TWIST methylome processing package")
subparsers = parser.add_subparsers(dest="mode", help="processing mode")
parser.add_argument("infiles", help="input aligned files")  ## positional argument
### traditional mode subparser
init_parser = subparsers.add_parser("init", help="Initialization.")
init_parser.add_argument(
    "-o",
    "--output-dir",
    dest="output_dir",
    type=str,
    help="output directory",
    default="",
)
init_parser.add_argument(
    "-i",
    "--input-dir",
    dest="input_dir",
    type=str,
    help="output methylation ratio file name. [default: STDOUT]",
    default="",
)
init_parser.add_argument(
    "-r",
    "--ref",
    dest="ref",
    type=str,
    help="reference genome fasta file. (required)",
    default="",
)
init_parser.add_argument(
    "-t",
    "--keep-temporary",
    dest="keep_temp",
    action="store_true",
    help="keep temporary files",
)

# process

process_parser = subparsers.add_parser("process", help="Processes")
process_parser.add_argument(
    "-s",
    "--step",
    dest="step",
    type=str,
    help="Step of processing: [1. trimming(trim_galore), 2. alignment and sorting(bwameth and Samtools), 3.mark duplicates(Picard), 4. methylation ratio calling(MethylDackel), 5. Nucleosome occupancy(DANPOS2)]",
    default="",
)

process_parser.add_argument(
    "--qc",
    dest="qc_step",
    type=str,
    help="Step of quality control: [1. fastqc after trimming, 2. fragment length distribution, 3. dinucleotide frequency along 147bp fragments, 4. methyaltion beta value distribution]",
    default="",
)

# aegument for specific step
process_parser.add_argument(
    "--trimgalore-args",
    dest="trimming_args",
    type=str,
    help="",
    default="",
)

process_parser.add_argument(
    "--bwameth-args",
    dest="bwameth_args",
    type=str,
    help="",
    default="",
)

process_parser.add_argument(
    "--alignment-args",
    dest="alignment_args",
    type=str,
    help="",
    default="",
)

process_parser.add_argument(
    "--picard-args",
    dest="picard_args",
    type=str,
    help="",
    default="",
)

process_parser.add_argument(
    "--methyldackel-args",
    dest="methyldackel_args",
    type=str,
    help="",
    default="",
)

process_parser.add_argument(
    "--danpos-args",
    dest="danpos_args",
    type=str,
    help="",
    default="",
)


def disp(txt, nt=0):
    if not args.quiet:
        print >> sys.stderr, "[methratio] @%s \t%s" % (time.asctime(), txt)


args = parser.parse_args()
if args.mode == "init":
    print('init')
    
if args.mode == "init":
    print('init')
