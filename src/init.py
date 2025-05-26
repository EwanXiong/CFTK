import pandas as pd
import time, sys, os, json, subprocess, re, os
import shlex, subprocess

pd.set_option("display.max_columns", None)  # Show all columns
pd.set_option("display.expand_frame_repr", False)  # Prevent line breaking
pd.set_option("display.colheader_justify", "center")  # Center the headers
# from power_analysis import *

message = ""
command = ""


def disp(txt):
    print("@%s \t%s" % (time.asctime(), txt), file=sys.stderr)


def run_command(cmd):
    args = shlex.split(cmd)
    print(args)
    p = subprocess.Popen(args)


def Merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res


def is_number(s):
    return bool(re.match(r"^-?\d+(?:\.\d+)?$", s))



def init(args):
    if os.path.exists("./twist_init.json"):
        disp("Initialization file found. Loading.")
        args.__dict__ = Merge(
            json.load(open("./twist_init.json", "r")), args.__dict__
        )  # This operation will overwrite the values in the init JSON file if the same key is present in the args object.
    else:
        disp("Initialization file not found. Creating one.")
        with open("./twist_init.json", "w") as f:
            json.dump(args.__dict__, f, indent=2)
    if args.output_dir == os.getcwd():
        disp(
            "Output directory is not sepecified. Using current directory: %s"
            % args.output_dir
        )
    if args.ref_index:
        disp("Indexing reference genome.\n\n")
        command = "bwameth.py index %s;" % args.ref
        disp("Running:\n %s\n" % command)
        os.system(command)
        command = "samtools faidx %s > %s.fai" % (
            args.ref,
            args.ref,
        )
        disp("Running:\n %s\n" % command)
        os.system(command)
    else:
        disp("Skip indexing reference genome.")

    if args.ref_dict:
        disp("Creating dictionary for reference genome.")
        command = "%s CreateSequenceDictionary R=%s O=%s.dict" % (
            args.picard_jar_path,
            args.ref,
            args.ref,
        )
        os.system(command)
    else:
        disp("Skip creating a dictionary for reference genome.")