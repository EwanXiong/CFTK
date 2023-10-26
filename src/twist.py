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
    "--ref-index",
    dest="ref_index",
    action="store_true",
    help="Index reference genome. (required by bwameth)",
)

init_parser.add_argument(
    "--ref-dict",
    dest="ref_dict",
    action="store_true",
    help="Create a dictionary for reference genome. (required by Picard)",
)

init_parser.add_argument(
    "--picard_jar_path",
    dest="picard_jar_path",
    action="store_true",
    help="Path to picard.jar.",
    default="picard",
)

init_parser.add_argument(
    "--danpos-path",
    dest="danpos_path",
    type=str,
    help="",
    default="danpos.py",
)
# process

process_parser = subparsers.add_parser("process", help="Processes")
process_parser.add_argument(
    "infile", type=str, help="input file", nargs="+"
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
    "-@", "--cores", dest="cores", type=int, help="Cores assigned.", default=1
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

merge_parser = subparsers.add_parser("merge", help="Merge")

merge_parser.add_argument(
    "infile", type=str, help="input file", nargs="+"
)  ## positional argument

merge_parser.add_argument(
    "-c",
    "--column",
    required=True,
    type=int,
    help="Specify the column number to process(column number start from 0).",
)

merge_parser.add_argument(
    "-d",
    "--index",
    nargs="*",
    type=int,
    help="Specify the column number to use(column number start from 0).",
    default=None,
)

merge_parser.add_argument("--skiprows", nargs="*", type=int, default=None)

merge_parser.add_argument(
    "-o",
    "--output",
    required=True,
    help="Output file",
)

QC_parser = subparsers.add_parser("QC", help="QC")

QC_parser.add_argument(
    "infile", type=str, help="input file", nargs="+"
)  ## positional argument

QC_parser.add_argument(
    "-s",
    "--step",
    dest="step",
    type=int,
    help="QC option: [1. methylation beta value distribution, 2. fragment length distribution, 3. dinucleotide frequency]",
    default=None,
    required=True,
    choices=range(1, 4),
)

QC_parser.add_argument(
    "-o",
    "--output",
    required=True,
    help="Output file",
)

QC_parser.add_argument(
    "--clip_R1",
    help="Timmed length from 5' end of read 1",
    type=int,
    default=0
)

QC_parser.add_argument(
    "--clip_R2",
    help="Timmed length from 5' end of read 2",
    type=int,
    default=0
)

QC_parser.add_argument(
    "-f",
    "--fragment",
    help="Fragment length to check",
    type=int,
    default=167
)


args = parser.parse_args()
if args.mode == "init":
    util.disp("Initialization.")
    util.init(args)

if args.mode == "process":
    util.disp("Processing.")
    util.process(args)

if args.mode == "merge":
    util.disp("Merging.")
    util.merge(args)

if args.mode == "qc":
    util.disp("QC.")
    util.qc(args)
