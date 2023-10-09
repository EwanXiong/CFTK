#!/usr/bin/env python3

import argparse, time, sys, os
import util

parser = argparse.ArgumentParser(description="TWIST methylome processing package")
subparsers = parser.add_subparsers(dest="mode", help="processing mode")

### traditional mode subparser
init_parser = subparsers.add_parser("init", help="Initialization.")
init_parser.add_argument(
    "-o",
    "--output-dir",
    dest="output_dir",
    type=str,
    help="output directory",
    default=os.getcwd(),
)
init_parser.add_argument(
    "-i",
    "--input-dir",
    dest="input_dir",
    type=str,
    help="output methylation ratio file name. [default: STDOUT]",
    default=None,
)
init_parser.add_argument(
    "-r",
    "--ref",
    dest="ref",
    type=str,
    help="reference genome fasta file. (required)",
    default=None,
    required=True,
)
init_parser.add_argument(
    "-t",
    "--keep-temporary",
    dest="keep_temp",
    action="store_true",
    help="keep temporary files",
)
init_parser.add_argument(
    "--picard_jar_path",
    dest="picard_jar_path",
    action="store_true",
    help="Path to picard.jar.",
    default="picard.jar"
)
# process

process_parser = subparsers.add_parser("process", help="Processes")
process_parser.add_argument(
    "infile", type=str, help="input file"
)  ## positional argument
process_parser.add_argument(
    "-s",
    "--step",
    dest="step",
    type=int,
    nargs="+",
    help="Step of processing: [1. trimming(trim_galore), 2. alignment and sorting(bwameth and Samtools), 3.mark duplicates(Picard), 4. methylation ratio calling(MethylDackel), 5. Nucleosome occupancy(DANPOS2)]",
    default=None,
    required=True,
    choices=range(1, 6),
)

process_parser.add_argument(
    "--qc",
    dest="qc_step",
    type=str,
    nargs="+",
    help="Step of quality control: [1. fragment length distribution, 2. dinucleotide frequency along 147bp fragments, 3. methyaltion beta value distribution]",
    default=None,
    choices=range(1, 5),
)

process_parser.add_argument(
    "--prefix", dest="prefix", type=str, help="Prefix for output files.", default=None
)

process_parser.add_argument(
    "-@", "--cores", dest="core", type=int, help="Cores assigned.", default=1
)

# argument for specific step
process_parser.add_argument(
    "--trimgalore-args",
    dest="trimming_args",
    type=str,
    help="",
    default=None,
)

process_parser.add_argument(
    "--trimgalore-output-dir",
    dest="trimgalore_output_dir",
    type=str,
    help="",
    default=os.getcwd() + "/trimgalore_output",
)

process_parser.add_argument(
    "--bwameth-args",
    dest="bwameth_args",
    type=str,
    help="",
    default="",
)

process_parser.add_argument(
    "--bwameth-output-dir",
    dest="bwameth_output_dir",
    type=str,
    help="",
    default=os.getcwd() + "/bwameth_output",
)

process_parser.add_argument(
    "--picard-args",
    dest="picard_args",
    type=str,
    help="",
    default="",
)

process_parser.add_argument(
    "--picard-output-dir",
    dest="picard_output_dir",
    type=str,
    help="",
    default=os.getcwd() + "/picard_output",
)

process_parser.add_argument(
    "--methyldackel-args",
    dest="methyldackel_args",
    type=str,
    help="",
    default="",
)

process_parser.add_argument(
    "--methyldackel-output-dir",
    dest="methyldackel_output_dir",
    type=str,
    help="",
    default=os.getcwd() + "/methyldackel_output",
)

process_parser.add_argument(
    "--danpos-args",
    dest="danpos_args",
    type=str,
    help="",
    default="",
)

process_parser.add_argument(
    "--danpos-output-dir",
    dest="danpos_output_dir",
    type=str,
    help="",
    default=os.getcwd() + "/danpos_output",
)


args = parser.parse_args()
if args.mode == "init":
    util.disp("Initialization.")
    util.init(args)

if args.mode == "process":
    util.disp("Processing.")
    util.process(args)
