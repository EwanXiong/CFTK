import pickle
import pandas as pd
import numpy as np
import time, sys, os, json, argparse, subprocess, re, glob, os
import seaborn as sns
import matplotlib.pyplot as plt
import shlex, subprocess
from modality_performance import *
from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from MESA.MESA import *

pd.set_option("display.max_columns", None)  # Show all columns
pd.set_option("display.expand_frame_repr", False)  # Prevent line breaking
pd.set_option("display.colheader_justify", "center")  # Center the headers
# from power_analysis import *
np.int = np.int64

steps = {
    1: "trimming(trim_galore)",
    2: "alignment and sorting(bwameth and Samtools)",
    3: "mark duplicates(Picard)",
    4: "methylation ratio calling(MethylDackel)",
    5: "nucleosome occupancy calculation(DANPOS3)",
    6: "window protection score calculation",
}
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


# Merge two dictionaries


# Process Module
def process(args):
    # check initilization
    if not os.path.exists("./twist_init.json"):
        disp("Initialization infomation is not found. Initilization required.")
        return 1
    # store json informaiton
    if not args.prefix:
        if (
            ("R1" in args.infile[0])
            or ("r1" in args.infile[0])
            or (".fa" in args.infile[0])
        ):
            args.prefix = re.split(
                "_R1_001|.R1_001|.fa",
                args.infile[0].split("/")[-1],
                flags=re.IGNORECASE,
            )[0]
        else:
            args.prefix = args.infile[0].split("/")[-1]
        disp(
            "Prefix for output is not specified. Using input files' name as prefix: %s\n"
            % args.prefix,
        )
    args.__dict__ = Merge(
        json.load(open("./twist_init.json", "r")), args.__dict__
    )  # This operation will overwrite the values in the init JSON file if the same key is present in the args object.
    if len(args.infile) == 0:
        disp("Input file error!\nCurrent input file(s):\n")
        print(args.infile, file=sys.stderr)
        return 1
    message = "Processing sample(s):\n"
    print(message, file=sys.stderr)
    print(args.infile, file=sys.stderr)
    message = "\nAnalysis steps:\n"
    for step in args.step:
        message = message + str(step) + ". %s \n" % str(steps[step])
    disp(message)
    infile = " ".join(args.infile)

    # Trimming
    if 1 in args.step:
        disp("Start: %s" % steps[1])
        output_dir = args.trimgalore_output_dir
        if os.path.exists(args.trimgalore_output_dir):
            disp("Outputting to: %s" % args.trimgalore_output_dir)
        else:
            disp("%s doesn't exist. Creating it for you." % args.trimgalore_output_dir)
            try:
                os.mkdir(args.trimgalore_output_dir)
            except:
                disp("Failed to create directory.")
        disp("Output:\n")
        print(
            "%s_val_1.fq(.gz)\n%s_val_2.fq(.gz)" % (args.prefix, args.prefix),
            file=sys.stderr,
        )
        #
        if args.trimming_args:
            command = (
                "trim_galore --paired --2colour 20 --cores %s -o %s --basename %s %s "
                % (
                    str(int(args.cores)),
                    str(args.trimgalore_output_dir).strip(),
                    str(args.prefix),
                    str(args.trimming_args).strip(),
                )
                + infile
                + " || exit 1;"
            )
            disp("Running:\n %s\n" % command)
            os.system(command)
        else:
            command = (
                "trim_galore --paired --2colour 20 --cores %s -o %s --basename %s "
                % (
                    str(int(args.cores)),
                    str(args.trimgalore_output_dir).strip(),
                    str(args.prefix),
                )
                + infile
                + " || exit 1;"
            )
            disp("Running:\n %s\n" % command)
            os.system(command)
        disp("Complete: %s" % steps[1])

    # Alignment and sorting
    if 2 in args.step:
        disp("Start: %s" % steps[2])
        output_dir = args.bwameth_output_dir
        if os.path.exists(args.bwameth_output_dir):
            disp("Outputting to: %s" % args.bwameth_output_dir)
        else:
            disp("%s doesn't exist. Creating it for you." % args.bwameth_output_dir)
            try:
                os.mkdir(args.bwameth_output_dir)
            except:
                disp("Failed to create directory.")
        disp("Output:\n")
        print(
            "%s.bam" % args.prefix,
            file=sys.stderr,
        )
        if 1 in args.step:
            if args.infile[0].endswith(".gz") and args.infile[1].endswith(".gz"):
                r1_input = (
                    str(args.trimgalore_output_dir).strip()
                    + "/%s_val_1.fq.gz" % args.prefix
                )
                r2_input = (
                    str(args.trimgalore_output_dir).strip()
                    + "/%s_val_2.fq.gz" % args.prefix
                )
            else:
                r1_input = (
                    str(args.trimgalore_output_dir).strip()
                    + "/%s_val_1.fq" % args.prefix
                )
                r2_input = (
                    str(args.trimgalore_output_dir).strip()
                    + "/%s_val_2.fq" % args.prefix
                )
        else:
            message = "Processing sample(s):\n"
            print(message, file=sys.stderr)
            print(args.infile, file=sys.stderr)
            r1_input, r2_input = args.infile
        if args.bwameth_args:
            command = (
                "bwameth.py --reference %s -t %s %s %s %s | "
                % (
                    args.ref,
                    args.cores,
                    str(args.bwameth_args).strip(),
                    r1_input,
                    r2_input,
                )
                + "sambamba view -t %s -F 'not secondary_alignment and not failed_quality_control and not supplementary and proper_pair and mapping_quality > 0' -f bam -S -l 0 /dev/stdin | "
                % args.cores
                + "sambamba sort -t %s -o %s/%s.bam /dev/stdin || exit 1;"
                % (args.cores, args.bwameth_output_dir, args.prefix)
                + "samtools index -@ %s %s/%s.bam || exit 1;"
                % (args.cores, args.bwameth_output_dir, args.prefix)
            )
        else:
            command = (
                "bwameth.py --reference %s -t %s %s %s | "
                % (args.ref, args.cores, r1_input, r2_input)
                + "sambamba view -t %s -F 'not secondary_alignment and not failed_quality_control and not supplementary and proper_pair and mapping_quality > 0' -f bam -S -l 0 /dev/stdin | "
                % args.cores
                + "sambamba sort -t %s -o %s/%s.bam /dev/stdin || exit 1;"
                % (args.cores, args.bwameth_output_dir, args.prefix)
                + "samtools index -@ %s %s/%s.bam || exit 1;"
                % (args.cores, args.bwameth_output_dir, args.prefix)
            )
        disp("Running:\n %s\n" % command)
        os.system(command)
        disp("Complete: %s" % steps[2])

    # Mark duplicates
    if 3 in args.step:
        disp("Start: %s" % steps[3])
        output_dir = args.picard_output_dir
        if os.path.exists(args.picard_output_dir):
            disp("Outputting to: %s" % args.picard_output_dir)
        else:
            disp("%s doesn't exist. Creating it for you." % args.picard_output_dir)
            try:
                os.mkdir(args.picard_output_dir)
            except:
                disp("Failed to create directory.")
        disp("Output:\n")
        print(
            "%s.markdup.bam" % args.prefix,
            file=sys.stderr,
        )
        if 2 in args.step:
            bam_input = str(args.bwameth_output_dir).strip() + "/%s.bam" % args.prefix
        else:
            message = "Processing sample(s):\n"
            print(message, file=sys.stderr)
            print(args.infile, file=sys.stderr)
            bam_input = args.infile[0]
        if args.picard_args:
            command = (
                "%s MarkDuplicates I=%s O=%s/%s.markdup.bam R=%s M=%s/%s.markdup_raw_metrics \
                SORTING_COLLECTION_SIZE_RATIO=0.15 ASSUME_SORT_ORDER=coordinate \
                OPTICAL_DUPLICATE_PIXEL_DISTANCE=2500 MAX_RECORDS_IN_RAM=1000 %s || exit 1;"
                % (
                    args.picard_jar_path,
                    bam_input,
                    args.picard_output_dir,
                    args.prefix,
                    args.ref,
                    args.picard_output_dir,
                    args.prefix,
                    args.picard_args,
                )
                + "samtools index -@ %s %s/%s.markdup.bam|| exit 1;"
                % (args.cores, args.picard_output_dir, args.prefix)
            )
        else:
            command = (
                "%s MarkDuplicates I=%s O=%s/%s.markdup.bam R=%s M=%s/%s.markdup_raw_metrics \
                SORTING_COLLECTION_SIZE_RATIO=0.15 ASSUME_SORT_ORDER=coordinate \
                OPTICAL_DUPLICATE_PIXEL_DISTANCE=2500 MAX_RECORDS_IN_RAM=1000 || exit 1;"
                % (
                    args.picard_jar_path,
                    bam_input,
                    args.picard_output_dir,
                    args.prefix,
                    args.ref,
                    args.picard_output_dir,
                    args.prefix,
                )
                + "samtools index -@ %s %s/%s.markdup.bam|| exit 1;"
                % (args.cores, args.picard_output_dir, args.prefix)
            )
        disp("Running:\n %s\n" % command)
        os.system(command)
        disp("Complete: %s" % steps[3])

    # Methylation ratio calling
    if 4 in args.step:
        disp("Start: %s" % steps[4])
        output_dir = args.methyldackel_output_dir
        if os.path.exists(args.methyldackel_output_dir):
            disp("Outputting to: %s" % args.methyldackel_output_dir)
        else:
            disp(
                "%s doesn't exist. Creating it for you." % args.methyldackel_output_dir
            )
            try:
                os.mkdir(args.methyldackel_output_dir)
            except:
                disp("Failed to create directory.")
        disp("Output:\n")
        print(
            "%s_CpG.bedGraph" % args.prefix,
            file=sys.stderr,
        )
        if 3 in args.step:
            bam_input = (
                str(args.picard_output_dir).strip() + "/%s.markdup.bam" % args.prefix
            )
        else:
            message = "Processing sample(s):\n"
            print(message, file=sys.stderr)
            print(args.infile, file=sys.stderr)
            bam_input = args.infile[0]
        if args.methyldackel_args:
            command = (
                "MethylDackel mbias -@ %s %s %s %s/%s &> %s/%s_mbias_OT_OB.temp || exit 1;"
                % (
                    args.cores,
                    args.ref,
                    bam_input,
                    args.methyldackel_output_dir,
                    args.prefix,
                    args.methyldackel_output_dir,
                    args.prefix,
                )
                + "MethylDackel extract --minDepth 10 --maxVariantFrac 0.25 -@ %s --OT $(cat %s/%s_mbias_OT_OB.temp | \
                grep -oP '(?<=--OT )[^ ]+') --OB $(cat %s/%s_mbias_OT_OB.temp | \
                grep -oP '(?<=--OB )[^ ]+') -o %s/%s %s \
                %s %s || exit 1;"
                % (
                    args.cores,
                    args.methyldackel_output_dir,
                    args.prefix,
                    args.methyldackel_output_dir,
                    args.prefix,
                    args.methyldackel_output_dir,
                    args.prefix,
                    args.methyldackel_args,
                    args.ref,
                    bam_input,
                )
            )
        else:
            command = (
                "MethylDackel mbias -@ %s %s %s %s/%s &> %s/%s_mbias_OT_OB.temp || exit 1;"
                % (
                    args.cores,
                    args.ref,
                    bam_input,
                    args.methyldackel_output_dir,
                    args.prefix,
                    args.methyldackel_output_dir,
                    args.prefix,
                )
                + "MethylDackel extract --minDepth 10 --maxVariantFrac 0.25 -@ %s --OT $(cat %s/%s_mbias_OT_OB.temp | \
                grep -oP '(?<=--OT )[^ ]+') --OB $(cat %s/%s_mbias_OT_OB.temp | \
                grep -oP '(?<=--OB )[^ ]+') -o %s/%s \
                %s %s || exit 1;"
                % (
                    args.cores,
                    args.methyldackel_output_dir,
                    args.prefix,
                    args.methyldackel_output_dir,
                    args.prefix,
                    args.methyldackel_output_dir,
                    args.prefix,
                    args.ref,
                    bam_input,
                )
            )
        disp("Running:\n %s\n" % command)
        os.system(command)
        disp("Complete: %s" % steps[4])

    # Nucleosome occupancy calculation
    if 5 in args.step:
        disp("Start: %s" % steps[5])
        output_dir = args.danpos_output_dir
        if os.path.exists(args.danpos_output_dir):
            disp("Outputting to: %s" % args.danpos_output_dir)
        else:
            disp("%s doesn't exist. Creating it for you." % args.danpos_output_dir)
            try:
                os.mkdir(args.danpos_output_dir)
            except:
                disp("Failed to create directory.")
        disp("Output:\n")
        print(
            "%s.occupancy.tsv" % args.prefix,
            file=sys.stderr,
        )
        if 3 in args.step:
            bam_input = (
                str(args.picard_output_dir).strip() + "/%s.markdup.bam" % args.prefix
            )
            danpos_wig_output = "%s/pooled/%s.markdup.Fnor.smooth.wig" % (
                args.danpos_output_dir,
                args.prefix,
            )
            danpos_bw_output = "%s/pooled/%s.bw" % (
                args.danpos_output_dir,
                args.prefix,
            )
            danpos_occupancy_output = "%s/pooled/%s.occupancy.tsv" % (
                args.danpos_output_dir,
                args.prefix,
            )
        else:
            message = "Processing sample(s):\n"
            print(message, file=sys.stderr)
            print(args.infile, file=sys.stderr)
            bam_input = args.infile[0]
            danpos_wig_output = "%s/pooled/%s.Fnor.smooth.wig" % (
                args.danpos_output_dir,
                bam_input.split("/")[-1].rsplit(".", 1)[0],
            )
            danpos_bw_output = "%s/pooled/%s.bw" % (
                args.danpos_output_dir,
                bam_input.split("/")[-1].rsplit(".", 1)[0],
            )
            danpos_occupancy_output = "%s/pooled/%s.occupancy.tsv" % (
                args.danpos_output_dir,
                bam_input.split("/")[-1].rsplit(".", 1)[0],
            )
        src_path = os.path.dirname(__file__)
        chrom_sizes = os.path.dirname(src_path) + "/hg38.chrom.sizes"
        # region_file = (
        #     os.path.dirname(src_path) + "/hg38_annotated_collapsed_TSS_PAS_1kb.bed"
        # )
        region_file = args.region
        if args.danpos_args:
            command = (
                "python %s dpos %s --paired 1 -u 0 -c 1000000 -o %s %s && \
            ./wigToBigWig -clip %s %s %s && \
            ./bigWigAverageOverBed %s %s %s || exit 1;"
                % (
                    args.danpos_path,
                    bam_input,
                    args.danpos_output_dir,
                    danpos_wig_output,
                    chrom_sizes,
                    danpos_bw_output,
                    danpos_bw_output,
                    region_file,
                    danpos_occupancy_output,
                    args.danpos_args,
                )
            )
        else:
            command = (
                "python %s dpos %s --paired 1 -u 0 -c 1000000 -o %s && \
                ./wigToBigWig -clip %s %s %s && \
                ./bigWigAverageOverBed %s %s %s || exit 1;"
                % (
                    args.danpos_path,
                    bam_input,
                    args.danpos_output_dir,
                    danpos_wig_output,
                    chrom_sizes,
                    danpos_bw_output,
                    danpos_bw_output,
                    region_file,
                    danpos_occupancy_output,
                )
            )
        disp("Running:\n %s\n" % command)
        os.system(command)
        disp("Complete: %s" % steps[5])

    # Window protection score(WPS) calculation
    if 6 in args.step:
        disp("Start: %s" % steps[6])
        output_dir = args.wps_output_dir
        if os.path.exists(args.wps_output_dir):
            disp("Outputting to: %s" % args.wps_output_dir)
        else:
            disp("%s doesn't exist. Creating it for you." % args.wps_output_dir)
            try:
                os.mkdir(args.wps_output_dir)
            except:
                disp("Failed to create directory.")
        disp("Output:\n")
        if args.prefix:
            print(
                "%s.wps.txt" % args.prefix,
                file=sys.stderr,
            )
        else:
            print(
                "%s.wps.txt" % bam_input.split("/")[-1].rsplit(".", 1)[0],
                file=sys.stderr,
            )
        if 3 in args.step:
            bam_input = (
                str(args.picard_output_dir).strip() + "/%s.markdup.bam" % args.prefix
            )
            wps_output = "%s/%s.wps.txt" % (
                args.wps_output_dir,
                args.prefix,
            )
        else:
            message = "Processing sample(s):\n"
            print(message, file=sys.stderr)
            print(args.infile, file=sys.stderr)
            bam_input = args.infile[0]
            wps_output = "%s/%s.wps.txt" % (
                args.wps_output_dir,
                bam_input.split("/")[-1].rsplit(".", 1)[0],
            )
        src_path = os.path.dirname(__file__)
        # region_file = (
        #     os.path.dirname(src_path) + "/hg38_annotated_collapsed_TSS_PAS_1kb.bed"
        # )
        region_file = args.region
        if args.wps_args:
            command = (
                "python %s/WPS_region.py -b %s -r %s -t %s -o %s --mean %s|| exit 1;"
                % (
                    src_path,
                    bam_input,
                    region_file,
                    args.cores,
                    wps_output,
                    args.wps_args,
                )
            )
        else:
            command = (
                "python %s/WPS_region.py -b %s -r %s -t %s -o %s --mean|| exit 1;"
                % (
                    src_path,
                    bam_input,
                    region_file,
                    args.cores,
                    wps_output,
                )
            )

        disp("Running:\n %s\n" % command)
        os.system(command)
        disp("Complete: %s" % steps[6])
    disp("Completing all processes.")
    try:
        with open("./twist_init.json", "w") as f:
            json.dump(args.__dict__, f, indent=2)
    except:
        disp("Failed to write initilization file.")


# Merge Module
def merge(args):
    if len(args.infile) == 1 and (
        "*" in args.infile[0]
    ):  # file names in regular expression
        file_list = glob.glob(args.infile[0])
    else:
        file_list = args.infile

    if not args.index:
        print("No index column specified. Using numerical index.")
        index_column = []
        index = False
    elif len(args.index) == 1:
        index = args.index
        index_column = index
        print(f"Single index column specified: {args.index}")
    else:
        index = args.index
        index_column = []
        print(f"Multiple index columns specified: {args.index}")

    print(f"Using value column: {args.column}")
    matrix_columns_name = [str(_).split("/")[-1].split(".")[0] for _ in file_list]
    # Read matrix values from files, assign matrix column and index name
    if len(index_column) == 0:
        merged_matrix = pd.concat(
            [
                pd.read_table(
                    f, usecols=[args.column], header=None, skiprows=args.skiprows
                )
                for f in file_list
            ],
            axis=1,
        )
        matrix_temp = []
        if index:  # when multiple index columns specified or no column specified
            for f in file_list:
                temp = pd.read_table(
                    f, usecols=[args.column], header=None, skiprows=args.skiprows
                )
                temp.index = list(
                    pd.read_table(
                        f,
                        usecols=index,
                        header=None,
                        index_col=False,
                        skiprows=args.skiprows,
                    ).apply(lambda row: "_".join(row.astype(str)), axis=1)
                )  # type: ignore
                matrix_temp.append(temp)
            merged_matrix = pd.concat(matrix_temp, axis=1)
    else:  # when single index column specified
        merged_matrix = pd.concat(
            [
                pd.read_table(
                    f,
                    usecols=index_column + [args.column],
                    header=None,
                    index_col=index_column,
                    skiprows=args.skiprows,
                )
                for f in file_list
            ],
            axis=1,
        )
    merged_matrix.columns = matrix_columns_name
    merged_matrix.to_csv(args.output, sep="\t")
    disp("Merging completed.")


def qc(args):
    if args.step == 1:
        disp("Plotting distribution plot for methylation beta values.")
        input_matrix = pd.read_table(args.infile)
        f, ax = plt.subplots(figsize=(4, 4))
        #ax.set_xlim(0, 1)
        xmin, xmax = np.nanmin(input_matrix), np.nanmax(input_matrix)
        input_matrix.iloc[::args.step_size, :].plot.density(ax=ax,
                                                        ind=np.linspace(xmin, xmax, 300))
        ax.set(xlabel='Methylation')
        ax.legend(bbox_to_anchor=(1, 1), frameon=False, fontsize='small').set_visible(args.legend)
        f.savefig(args.output, dpi=500, bbox_inches="tight")

    if args.step == 2:
        disp("Plotting distribution plot for fragment length.")
        for f in args.infile:
            command = (
                "bamPEFragmentSize --outRawFragmentLengths %s.%s.raw_length.csv -b %s"
                % (args.output, f.split("/")[-1].split(".")[0], f)
            )
            os.system(command)
        fragment_length_all = []
        for i in glob.glob("%s.*.raw_length.csv" % args.output):
            temp = pd.read_table(i, skiprows=1).iloc[:, :2]
            temp["Sample"] = i.rsplit("/", 1)[1].split("_")[0]
            fragment_length_all.append(temp)

        mean_fragment_length = pd.concat(
            [
                pd.merge(
                    _,
                    pd.DataFrame(np.arange(300), columns=["Size"]),
                    on="Size",
                    how="outer",
                )
                .sort_values(by="Size")
                .reset_index(drop=True)
                .Occurrences
                for _ in fragment_length_all
            ],
            axis=1,
        ).mean(axis=1)

        sns.set_context("paper", font_scale=1.5)
        f, ax = plt.subplots(figsize=(4, 4))
        temp = 100 * mean_fragment_length / mean_fragment_length.sum()
        sns.lineplot(temp.values, ax=ax, linewidth=2)
        ax.plot(
            (temp.argmax(), temp.argmax()),
            (-1, temp.max()),
            linestyle="-.",
            linewidth=1,
            color="red",
            alpha=0.7,
        )
        extraticks = [temp.argmax()]
        # plt.xticks(list(plt.xticks()[0]) + extraticks)
        ax.set(xticks=list(plt.xticks()[0]) + extraticks)
        ax.set(
            xlim=[temp.argmax() - 99, temp.argmax() + 99],
            ylim=[temp.min() - 0.05, temp.max() + 0.05],
            xlabel="Fragment length(bp)",
            ylabel="% of fragments",
        )
        ax.figure.savefig(
            args.output,
            dpi=500,
            bbox_inches="tight",
        )

    if args.step == 3:
        disp("Plotting dinucleotide frequency of fragments.")

        for f in args.infile:
            sample_id = str(f).split("/")[-1].split(".")[0]
            # bed file for every complete fragment

            command = (
                "bedtools bamtobed -bedpe -mate1 -i ${bwameth_output_dir}/${sample_id}_sorted.bam |"
                + "awk -v OFS='\t' -v sample=%s -v cr1=%s -v cr2=%s\
                '{if($9=='+')\
                {($2-cr1<$5)?start=$2-cr1:start=$5;\
                ($3>$6+cr2)?end=$3:end=$6+cr2;print $1,start,end,sample} \
                else{($2<$5-cr1)?start=$2:start=$5-cr1; \
                ($3+cr2>$6)?end=$3+cr2:end=$6; print $1,start,end,sample}}' | \
                awk -v OFS='\t' '$3-$2==%s{print $1,$2,$3,$4}' >> %s.all_fragment"
                % (sample_id, args.clip_R1, args.clip_R2, args.fragment, args.output)
            )
            os.system(command)

        fragment_windows = pd.read_table("%s.all_fragment" % args.output, header=None)
        temp_ = int((250 - args.fragment / 2))
        with open(args.output, "ab") as f:
            for idx, row in fragment_windows.iterrows():
                # f.write(b"\n")
                np.savetxt(
                    f,
                    np.array(
                        [
                            [row[0]] * 250,
                            np.arange(row[1] - temp_, row[2] + temp_ + 1),
                            np.arange(row[1] - temp_ + 2, row[2] + temp_ + 3),
                            np.arange(250),
                            [row[3]] * 250,
                        ]
                    ).T,
                    delimiter="\t",
                    fmt="%s",
                )

    disp("QC completed.")


classifier_dist = {
    1: RandomForestClassifier(random_state=0, n_jobs=-1),
    2: LogisticRegression(random_state=0, n_jobs=-1),
    3: SVC(random_state=0),
    4: XGBClassifier(random_state=0, n_jobs=-1),
}


def mesa_performance(args):
    modality_name = args.modality
    modality_matrix = [pd.read_csv(_, sep="\t", index_col=0).T for _ in args.infile]
    if len(modality_name) != len(modality_matrix):
        disp("Number of modalities and matrices are not equal.")
        return 1

    label = pd.read_table(args.label, header=None, index_col=0).values.reshape(-1)
    # if len(args.clf) == len(modality_name):
    #     classifiers = [classifier_dist[_] for _ in args.clf]
    # else:
    #     classifiers = classifier_dist[args.clf[0]] * len(modality_name)
    classifiers = [classifier_dist[_] for _ in args.clf]
    performance = modality_performance(
        modality_name,
        modality_matrix,
        label,
        classifiers,
        feature_size=args.size,
        subset=args.subset,
        repeat=args.repeat,
        n_jobs=args.cores,
    )
    return performance


def mesa(args):
    if args.performance:
        performance = mesa_performance(args)
        # print(performance[0], file=sys.stderr)
        disp("Modality performance summary\n%s\n" % performance[0])
        performance[0].to_csv(args.output_dir + "/modality_performance.tsv", sep="\t")
    if args.mesa:
        disp("Constructing MESA model.")
        if args.performance:
            selected_modality = performance.head(args.max_modality)
            selected_modality_name = selected_modality.index.values
            disp(
                "Modality performance test is done earlier. Using the top %s modalities: %s"
                % (args.max_modality, selected_modality_name)
            )
            modality_matrix = [
                pd.read_csv(
                    args.modality_matrix[np.where(args.modality == _)[0]],
                    sep="\t",
                    index_col=0,
                ).T
                for _ in selected_modality_name
            ]
            modality_clf = [
                classifier_dist[_[1]]
                for _ in selected_modality["best_classifier, idx"].values
            ]
        else:
            disp(
                "Modality performance test skipped. Loading modality(s) matrix from input: %s"
                % args.infile
            )
            modality_matrix = [
                pd.read_csv(_, sep="\t", index_col=0).T for _ in args.infile
            ]
            # modality_clf = [classifier_dist[_] for _ in args.clf]
            if len(args.clf) == len(modality_matrix):
                disp("Classifier for each modality is specified.")
                modality_clf = [classifier_dist[_] for _ in args.clf]
                disp(modality_clf)
            elif len(args.clf) > 0:
                disp(
                    "Number of classifier > or < number of modality(s). Use first classifer for all modality(s)."
                )
                disp(classifier_dist[args.clf[0]])
                modality_clf = [classifier_dist[args.clf[0]]] * len(modality_matrix)
            else:
                disp(
                    "Classifier for each modality is not specified properly. Use default classifier for all modality(s)."
                )
                modality_clf = [classifier_dist[1]] * len(modality_matrix)
                disp(modality_clf)
        disp("Loading label from input: %s" % args.label)
        y = pd.read_table(args.label, header=None, index_col=0).values.reshape(-1)
        disp("Fitting base estimators.")
        modalities = [
            MESA_modality(classifier=clf, top_n=100).fit(X, y)
            for X, clf in zip(modality_matrix, modality_clf)
        ]
        disp("Constructing MESA model.")
        mesa_model = MESA(
            meta_estimator=LogisticRegression(random_state=0, n_jobs=-1)
        ).fit(modalities, modality_matrix, y)
        pickle.dump(mesa_model, open(args.output_dir + "/MESA_model.pkl", "wb"))
        disp("MESA model saved to %s" % args.output_dir + "/MESA_model.pkl")
        # save trained MESA model
    if args.cv_mesa:
        disp("Loading modality(s) matrix from input: %s" % args.infile)
        modality_matrix = [pd.read_csv(_, sep="\t", index_col=0).T for _ in args.infile]

        if len(args.clf) == len(modality_matrix):
            disp("Classifier for each modality is specified.")
            modality_clf = [classifier_dist[_] for _ in args.clf]
            disp(modality_clf)
        elif len(args.clf) > 0:
            disp(
                "Number of classifier > or < number of modality(s). Use first classifer for all modality(s)."
            )
            disp(classifier_dist[args.clf[0]])
            modality_clf = [classifier_dist[args.clf[0]]] * len(modality_matrix)
        else:
            disp(
                "Classifier for each modality is not specified properly. Use default classifier for all modality(s)."
            )
            modality_clf = [classifier_dist[1]] * len(modality_matrix)
            disp(modality_clf)

        disp("Loading label from input: %s" % args.label)
        y = pd.read_table(args.label, header=None, index_col=0).values.reshape(-1)
        if len(modality_matrix) == 1:
            disp("MESA cross-validation for one modality.")
            modality_matrix = modality_matrix[0]
        else:
            disp("MESA cross-validation for multiple modalities.")
        mesa_cv = MESA_CV(
            selector=GenericUnivariateSelect(
                score_func=wilcoxon, mode="k_best", param=args.size
            ),
            meta_estimator=LogisticRegression(random_state=0, n_jobs=-1),
        ).fit(modality_matrix, y)
        disp("MESA cross-validation AUC: %s" % mesa_cv.get_performance())
    return


def power(args):
    # disp("Power analysis.")
    alpha = 2.7050713203440227e-08
    if args.lr:
        alpha = 2.221510618472339e-20
    if args.p:
        alpha = args.p

    src_path = os.path.dirname(__file__)
    cpg_std_path = os.path.dirname(src_path) + "/twist_497sample_cpg_std.pkl"
    alpha = 2.7050713203440227e-08
    if args.lr:
        alpha = 2.221510618472339e-20
    if args.p:
        alpha = args.p
    disp("Output result to %s" % args.output_dir)
    if not (os.path.exists(args.output_dir) and os.path.isdir(args.output_dir)):
        disp("Directory does not exist! Creating it for you.")
        try:
            os.mkdir(args.output_dir)
        except:
            disp("Failed to create directory.")
    command = (
        "python %s/power_analysis.py -s %s -e %s -o %s --cpg-std %s -p %s -@ %s --step-size %s|| exit 1;"
        % (
            src_path,
            args.sample_size,
            args.effect_size,
            args.output_dir,
            cpg_std_path,
            alpha,
            args.cores,
            args.step_size,
        )
    )
    os.system(command)
    disp(f"Power analysis completed. Results saved in {args.output_dir}.")
    return


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def analysis(args):
    if args.pca:
        disp("Performing PCA analysis.")
        input = pd.read_csv(args.infile, sep="\t", index_col=0).T
        pca = PCA(n_components=10)
        pc = pd.DataFrame(
            pca.fit_transform(input), index=input.index, columns=[f"PC{i}" for i in range(1, 11)]
        )
        f, ax = plt.subplots(figsize=(5, 5))
        sns.set_theme(context="talk", style="ticks")
        if args.label:
            label = pd.read_table(args.label, header=None, index_col=0)
            sns.scatterplot(
                data=pc, x="PC1", y="PC2", hue=label, ax=ax
            )  # add explanation ratio
        else:
            sns.scatterplot(data=pc, x="PC1", y="PC2", ax=ax)
        ax.set(
            ylabel=f"PC2({pca.explained_variance_ratio_[1]:.3f})",
            xlabel=f"PC1({pca.explained_variance_ratio_[0]:.3f})",
        )
        ax.figure.savefig(
            args.output_dir + "/PCA_plot.png", bbox_inches="tight", dpi=500
        )
        pca.to_csv(args.output_dir + "/10PCs.tsv", sep="\t")
        disp("PCA analysis completed.")
    if args.violin:
        disp("Violin plot.")
        input = pd.read_csv(args.infile, sep="\t", index_col=0)
        if args.label:
            label = pd.read_table(args.label, header=None, index_col=0)
            input = pd.concat([input, label], axis=1)
            input = input.melt(id_vars=input.columns[-1], value_vars=input.columns[:-1])
            f, ax = plt.subplots(figsize=(4, 4))
            sns.violinplot(
                data=input, x="variable", y="value", hue=input.columns[-1], ax=ax
            )
            ax.figure.savefig(
                args.output_dir + "/violin_plot.png", bbox_inches="tight", dpi=500
            )
        else:
            disp("No label provided. Skipping violin plot analysis.")
        disp("Violin plot done")
    if args.heatmap:
        disp("Heatmap plot.")
        input = pd.read_csv(args.infile, sep="\t", index_col=0)
        if args.label:
            label = pd.read_table(args.label, header=None, index_col=0)
            label_unique = label.iloc[:, 0].unique()
            ax = sns.clustermap(
                input,
                col_colors=[
                    "g" if _.startswith(label_unique[0]) else "r" for _ in label.columns
                ],
                yticklabels=False,
                xticklabels=True,
                cmap="rocket_r",
                figsize=(4, 5),
            )
        else:
            ax = sns.clustermap(
                input,
                yticklabels=False,
                xticklabels=True,
                cmap="rocket_r",
                figsize=(4, 5),
            )
        plt.savefig(args.output_dir + "/heatmap.png", dpi=500, bbox_inches="tight")
        disp("Heatmap plot done.")
    if args.diff:
        disp("Differential analysis.")
    # plot
    return


"""
bamPEFragmentSize \
-hist ${fragment_length_output_dir}/${sample_id}_fragmentSize.png \
-T "Fragment size" \
-p 40 \
--outRawFragmentLengths ${fragment_length_output_dir}/${sample_id}_fragmentSize_raw.csv \
--table ${fragment_length_output_dir}/${sample_id}_fragmentSize_table.csv \
-b ${bwameth_output_dir}/${sample_id}_sorted.bam;


for sample_id in ${sample_list[@]}; 
do 
bedtools bamtobed -bedpe -mate1 -i ${bwameth_output_dir}/${sample_id}_sorted.bam > ${sample_id}.mate1First.bedpe;
awk -v OFS="\t" -v sample=$sample_id '{if($9=="+"){($2-10<$5)?start=$2-10:start=$5;($3>$6+10)?end=$3:end=$6+10;print $1,start,end,sample} else{($2<$5-10)?start=$2:start=$5-10; ($3+10>$6)?end=$3+10:end=$6; print $1,start,end,sample}}' ${sample_id}.mate1First.bedpe| awk -v OFS="\t" '$3-$2==147{print $1,$2,$3,$4}' >> $fragment_bed
done

python ${project_dir}/src/20230926_dinucleotide_slidingwindow.py -i ${fragment_bed} -o ${fragment_bed}.window2bp || exit 1;


fragment_windows = pd.read_table(
    input_path,
    header=None)

# 65706750 rows
with open(
        output_path,
        "ab") as f:
    for idx, row in fragment_windows.iterrows():
        #f.write(b"\n")
        np.savetxt(f,
                   np.array([[row[0]] * 250,
                             np.arange(row[1] - 51, row[2] + 52),
                             np.arange(row[1] - 49, row[2] + 54),
                             np.arange(250), [row[3]] * 250]).T,
                   delimiter='\t',
                   fmt="%s")


declare -a dinuc_list=("AA" "AT" "TA" "TT" "GG" "GC" "CG" "GC")
# Iterate the string array using for loop
for pattern in ${dinuc_list[@]}; do
bedtools nuc -pattern $pattern -C -fi $hg38_ref -bed ${fragment_bed}.window2bp > ${dinucleotide_freq_output_dir}/fragments_147bp_freq_${pattern}.txt || exit 1;
done


"""
