import pandas as pd
import time, sys, os, json, argparse, subprocess, re, glob, os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Union

pd.set_option("display.max_columns", None)  # Show all columns
pd.set_option("display.expand_frame_repr", False)  # Prevent line breaking
pd.set_option("display.colheader_justify", "center")  # Center the headers
# from power_analysis import *

steps = {
    1: "trimming(trim_galore)",
    2: "alignment and sorting(bwameth and Samtools)",
    3: "mark duplicates(sambamba/samblaster/Picard)",
    4: "methylation ratio calling(MethylDackel)",
    5: "nucleosome occupancy calculation(DANPOS3)",
    6: "window protection score calculation",
}

# Pipeline step definitions
PROCESSING_STEPS = {
    1: "Read trimming (trim_galore)",
    2: "Alignment and sorting (bwameth and Samtools)",
    3: "Mark duplicates (sambamba/samblaster/Picard)",
    4: "Methylation ratio calling (MethylDackel)",
    5: "Nucleosome occupancy calculation (DANPOS3)",
    6: "Window protection score calculation",
}

message = ""
command = ""


def disp(txt):
    print("@%s \t%s" % (time.asctime(), txt), file=sys.stderr)


class ProcessingError(Exception):
    """Custom exception for processing errors."""

    pass


def run_command(command: str, check_success: bool = True) -> int:
    """
    Execute shell command with proper error handling.

    Args:
        command: Shell command to execute
        check_success: Whether to check return code

    Returns:
        Return code from command execution

    Raises:
        ProcessingError: If command fails and check_success is True
    """
    disp(f"Running command:\n{command}")

    try:
        result = subprocess.run(
            command, shell=True, check=check_success, capture_output=False, text=True
        )
        return result.returncode
    except subprocess.CalledProcessError as e:
        error_msg = f"Command failed with return code {e.returncode}: {command}"
        disp(f"ERROR: {error_msg}")
        if check_success:
            raise ProcessingError(error_msg) from e
        return e.returncode


def Merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res


def is_number(s):
    return bool(re.match(r"^-?\d+(?:\.\d+)?$", s))


def create_directory(directory_path: str) -> bool:
    """
    Create directory if it doesn't exist.

    Args:
        directory_path: Path to directory to create

    Returns:
        True if directory was created or already exists, False on failure
    """
    try:
        Path(directory_path).mkdir(parents=True, exist_ok=True)
        disp(f"Directory ready: {directory_path}")
        return True
    except OSError as e:
        disp(f"Failed to create directory {directory_path}: {e}")
        return False


def validate_input_files(file_paths: List[str]) -> bool:
    """
    Validate that input files exist and are readable.

    Args:
        file_paths: List of file paths to validate

    Returns:
        True if all files are valid, False otherwise
    """
    for file_path in file_paths:
        if not os.path.isfile(file_path):
            disp(f"ERROR: Input file not found: {file_path}")
            return False
        if not os.access(file_path, os.R_OK):
            disp(f"ERROR: Input file not readable: {file_path}")
            return False
    return True


def determine_prefix(
    input_files: List[str], specified_prefix: Optional[str] = None
) -> str:
    """
    Determine output prefix from input files or use specified prefix.

    Args:
        input_files: List of input file paths
        specified_prefix: User-specified prefix (optional)

    Returns:
        Determined prefix for output files
    """
    if specified_prefix:
        return specified_prefix

    # Extract prefix from first input file
    first_file = os.path.basename(input_files[0])

    # Handle common naming patterns
    patterns = [
        r"_R1_001",
        r"\.R1_001",
        r"\.fa",
        r"_R1\.",
        r"_r1\.",
    ]

    for pattern in patterns:
        if re.search(pattern, first_file, re.IGNORECASE):
            prefix = re.split(pattern, first_file, flags=re.IGNORECASE)[0]
            disp(f"Auto-determined prefix: {prefix}")
            return prefix

    # Default to full filename without extension
    prefix = os.path.splitext(first_file)[0]
    disp(f"Using filename as prefix: {prefix}")
    return prefix


def get_trimmed_file_paths(args, extension: str = "fq") -> tuple:
    """
    Get paths for trimmed files based on input file extensions.

    Args:
        args: Arguments object containing paths and prefix
        extension: File extension ('fq' or 'fq.gz')

    Returns:
        Tuple of (r1_path, r2_path)
    """
    output_dir = args.trimgalore_output_dir.rstrip("/")

    if extension == "fq.gz":
        r1_path = f"{output_dir}/{args.prefix}_val_1.fq.gz"
        r2_path = f"{output_dir}/{args.prefix}_val_2.fq.gz"
    else:
        r1_path = f"{output_dir}/{args.prefix}_val_1.fq"
        r2_path = f"{output_dir}/{args.prefix}_val_2.fq"

    return r1_path, r2_path


def execute_trimming(args) -> None:
    """
    Execute read trimming step using trim_galore.

    Args:
        args: Arguments object containing processing parameters
    """
    disp(f"Start: {PROCESSING_STEPS[1]}")

    # Create output directory
    if not create_directory(args.trimgalore_output_dir):
        raise ProcessingError("Failed to create trimming output directory")

    disp("Output files:")
    print(f"{args.prefix}_val_1.fq(.gz)\n{args.prefix}_val_2.fq(.gz)", file=sys.stderr)

    # Build command
    base_cmd = (
        f"trim_galore --paired --2colour 20 --cores {args.cores} "
        f"-o {args.trimgalore_output_dir} --basename {args.prefix}"
    )

    if hasattr(args, "trimming_args") and args.trimming_args:
        cmd = f"{base_cmd} {args.trimming_args} {' '.join(args.infile)}"
    else:
        cmd = f"{base_cmd} {' '.join(args.infile)}"

    cmd += " || exit 1"

    run_command(cmd)
    disp(f"Complete: {PROCESSING_STEPS[1]}")


def execute_alignment(args) -> None:
    """
    Execute alignment and sorting step using bwameth and sambamba.

    Args:
        args: Arguments object containing processing parameters
    """
    disp(f"Start: {PROCESSING_STEPS[2]}")

    # Create output directory
    if not create_directory(args.bwameth_output_dir):
        raise ProcessingError("Failed to create alignment output directory")

    disp("Output files:")
    print(f"{args.prefix}.bam", file=sys.stderr)

    # Determine input files
    if 1 in args.step:
        # Use trimmed files
        is_gzipped = all(f.endswith(".gz") for f in args.infile)
        extension = "fq.gz" if is_gzipped else "fq"
        r1_input, r2_input = get_trimmed_file_paths(args, extension)
    else:
        # Use original input files
        disp("Using original input files:")
        print(args.infile, file=sys.stderr)
        r1_input, r2_input = args.infile[:2]

    # Build bwameth command
    bwameth_cmd = f"bwameth.py --reference {args.ref} -t {args.cores}"
    if hasattr(args, "bwameth_args") and args.bwameth_args:
        bwameth_cmd += f" {args.bwameth_args}"
    bwameth_cmd += f" {r1_input} {r2_input}"

    # Build full pipeline command
    filter_cmd = (
        f"sambamba view -t {args.cores} "
        "-F 'not secondary_alignment and not failed_quality_control "
        "and not supplementary and proper_pair and mapping_quality > 0' "
        "-f bam -S -l 0 /dev/stdin"
    )

    sort_cmd = f"sambamba sort -t {args.cores} -o {args.bwameth_output_dir}/{args.prefix}.bam /dev/stdin"
    index_cmd = (
        f"samtools index -@ {args.cores} {args.bwameth_output_dir}/{args.prefix}.bam"
    )

    full_cmd = (
        f"{bwameth_cmd} | {filter_cmd} | {sort_cmd} || exit 1; {index_cmd} || exit 1"
    )

    run_command(full_cmd)
    disp(f"Complete: {PROCESSING_STEPS[2]}")


def execute_mark_duplicates(args) -> None:
    """
    Execute duplicate marking step using specified tool.

    Args:
        args: Arguments object containing processing parameters
    """
    disp(f"Start: {PROCESSING_STEPS[3]}")

    # Create output directory
    if not create_directory(args.markdup_output_dir):
        raise ProcessingError("Failed to create mark duplicates output directory")

    disp(f"Output files: {args.prefix}.markdup.bam")

    # Determine input BAM file
    if 2 in args.step:
        bam_input = f"{args.bwameth_output_dir}/{args.prefix}.bam"
    else:
        disp(f"Using provided BAM file: {args.infile}")
        bam_input = args.infile[0]

    # Build command based on tool
    tool = getattr(args, "markdup_tool", "sambamba")
    extra_args = getattr(args, "markdup_args", "")

    if tool == "sambamba":
        cmd = (
            f"sambamba markdup -t {args.cores} {extra_args} "
            f"{bam_input} {args.markdup_output_dir}/{args.prefix}.markdup.bam || exit 1"
        )

    elif tool == "samblaster":
        cmd = (
            f"samblaster {extra_args} --addMateTags "
            f"--splitFile {args.markdup_output_dir}/{args.prefix}.markdup.split.bam "
            f"--outputFile {args.markdup_output_dir}/{args.prefix}.markdup.bam "
            f"< {bam_input} || exit 1"
        )

    elif tool == "picard":
        jar_path = getattr(args, "picard_jar_path", "picard")
        cmd = (
            f"{jar_path} MarkDuplicates "
            f"I={bam_input} "
            f"O={args.markdup_output_dir}/{args.prefix}.markdup.bam "
            f"R={args.ref} "
            f"M={args.markdup_output_dir}/{args.prefix}.markdup_metrics.txt "
            "SORTING_COLLECTION_SIZE_RATIO=0.15 "
            "ASSUME_SORT_ORDER=coordinate "
            "OPTICAL_DUPLICATE_PIXEL_DISTANCE=2500 "
            f"MAX_RECORDS_IN_RAM=1000 {extra_args} || exit 1; "
            f"samtools index -@ {args.cores} "
            f"{args.markdup_output_dir}/{args.prefix}.markdup.bam || exit 1"
        )
    else:
        raise ProcessingError(f"Unknown duplicate marking tool: {tool}")

    cmd = f"{cmd} samtools index -@ {args.cores} {args.markdup_output_dir}/{args.prefix}.markdup.bam|| exit 1;"
    run_command(cmd)
    disp(f"Complete: {PROCESSING_STEPS[3]}")


def execute_methylation_calling(args) -> None:
    """
    Execute methylation calling step using MethylDackel.

    Args:
        args: Arguments object containing processing parameters
    """
    disp(f"Start: {PROCESSING_STEPS[4]}")

    # Create output directory
    if not create_directory(args.methyldackel_output_dir):
        raise ProcessingError("Failed to create MethylDackel output directory")

    disp(f"Output files: {args.prefix}_CpG.bedGraph")

    # Determine input BAM file
    if 3 in args.step:
        bam_input = f"{args.markdup_output_dir}/{args.prefix}.markdup.bam"
    elif 2 in args.step:
        bam_input = f"{args.bwameth_output_dir}/{args.prefix}.bam"
    else:
        disp(f"Using provided BAM file: {args.infile}")
        bam_input = args.infile[0]

    # Build MethylDackel commands
    mbias_cmd = (
        f"MethylDackel mbias -@ {args.cores} {args.ref} {bam_input} "
        f"{args.methyldackel_output_dir}/{args.prefix} "
        f"&> {args.methyldackel_output_dir}/{args.prefix}_mbias_OT_OB.temp || exit 1"
    )

    extract_cmd = (
        f"MethylDackel extract --minDepth 10 --maxVariantFrac 0.25 "
        f"-@ {args.cores} "
        f"--OT $(cat {args.methyldackel_output_dir}/{args.prefix}_mbias_OT_OB.temp | "
        "grep -oP '(?<=--OT )[^ ]+') "
        f"--OB $(cat {args.methyldackel_output_dir}/{args.prefix}_mbias_OT_OB.temp | "
        "grep -oP '(?<=--OB )[^ ]+') "
        f"-o {args.methyldackel_output_dir}/{args.prefix} "
    )

    if hasattr(args, "methyldackel_args") and args.methyldackel_args:
        extract_cmd += f"{args.methyldackel_args} "

    extract_cmd += f"{args.ref} {bam_input} || exit 1"

    full_cmd = f"{mbias_cmd}; {extract_cmd}"

    run_command(full_cmd)
    disp(f"Complete: {PROCESSING_STEPS[4]}")


def execute_nucleosome_occupancy(args) -> None:
    """
    Execute nucleosome occupancy calculation using DANPOS3.

    Args:
        args: Arguments object containing processing parameters
    """
    disp(f"Start: {PROCESSING_STEPS[5]}")

    # Create output directory
    if not create_directory(args.danpos_output_dir):
        raise ProcessingError("Failed to create DANPOS output directory")

    disp("Output files:")
    print(f"{args.prefix}.occupancy.tsv", file=sys.stderr)

    # Determine input BAM file and output paths
    if 3 in args.step:
        bam_input = f"{args.markdup_output_dir}/{args.prefix}.markdup.bam"
        base_name = f"{args.prefix}.markdup"
    else:
        disp("Using provided BAM file:")
        print(args.infile, file=sys.stderr)
        bam_input = args.infile[0]
        base_name = os.path.splitext(os.path.basename(bam_input))[0]

    # Define output file paths
    wig_output = f"{args.danpos_output_dir}/pooled/{base_name}.Fnor.smooth.wig"
    bw_output = f"{args.danpos_output_dir}/pooled/{base_name}.bw"
    occupancy_output = f"{args.danpos_output_dir}/pooled/{base_name}.occupancy.tsv"

    # Get required files
    src_path = os.path.dirname(__file__)
    chrom_sizes = os.path.join(os.path.dirname(src_path), "hg38.chrom.sizes")
    region_file = getattr(
        args,
        "region",
        os.path.join(
            os.path.dirname(src_path), "hg38_annotated_collapsed_TSS_PAS_1kb.bed"
        ),
    )

    # Build DANPOS command
    danpos_cmd = (
        f"python {args.danpos_path} dpos {bam_input} "
        f"--paired 1 -u 0 -c 1000000 -o {args.danpos_output_dir}"
    )

    if hasattr(args, "danpos_args") and args.danpos_args:
        danpos_cmd += f" {args.danpos_args}"

    # Build full pipeline command
    full_cmd = (
        f"{danpos_cmd} && "
        f"wigToBigWig -clip {wig_output} {chrom_sizes} {bw_output} && "
        f"bigWigAverageOverBed {bw_output} {region_file} {occupancy_output} || exit 1"
    )

    run_command(full_cmd)
    disp(f"Complete: {PROCESSING_STEPS[5]}")


def execute_wps_calculation(args) -> None:
    """
    Execute window protection score calculation.

    Args:
        args: Arguments object containing processing parameters
    """
    disp(f"Start: {PROCESSING_STEPS[6]}")

    # Create output directory
    if not create_directory(args.wps_output_dir):
        raise ProcessingError("Failed to create WPS output directory")

    # Determine input BAM file and output path
    if 3 in args.step:
        bam_input = f"{args.markdup_output_dir}/{args.prefix}.markdup.bam"
        output_file = f"{args.wps_output_dir}/{args.prefix}.wps.txt"
    else:
        disp(f"Using provided BAM file: {args.infile}")
        bam_input = args.infile[0]
        base_name = os.path.splitext(os.path.basename(bam_input))[0]
        output_file = f"{args.wps_output_dir}/{base_name}.wps.txt"

    disp("Output files:")
    print(os.path.basename(output_file), file=sys.stderr)

    # Get region file
    src_path = os.path.dirname(__file__)
    region_file = getattr(
        args,
        "region",
        os.path.join(
            os.path.dirname(src_path), "hg38_annotated_collapsed_TSS_PAS_1kb.bed"
        ),
    )

    # Build WPS command
    wps_script = os.path.join(src_path, "WPS_region.py")
    cmd = (
        f"python {wps_script} -b {bam_input} -r {region_file} "
        f"-t {args.cores} -o {output_file} --mean"
    )

    if hasattr(args, "wps_args") and args.wps_args:
        cmd += f" {args.wps_args}"

    cmd += " || exit 1"

    run_command(cmd)
    disp(f"Complete: {PROCESSING_STEPS[6]}")


def process(args) -> int:
    """
    Main processing function that orchestrates the entire pipeline.

    Args:
        args: Arguments object containing all processing parameters

    Returns:
        0 for success, 1 for error
    """
    try:
        # Check for initialization file
        if not os.path.exists("./twist_init.json"):
            disp("Initialization information not found. Initialization required.")
            return 1

        # Load initialization data and merge with args
        try:
            with open("./twist_init.json", "r") as f:
                init_data = json.load(f)

            # Merge initialization data with args (args takes precedence)
            for key, value in init_data.items():
                if not hasattr(args, key) or getattr(args, key) is None:
                    setattr(args, key, value)

        except (json.JSONDecodeError, IOError) as e:
            disp(f"Error loading initialization file: {e}")
            return 1

        # Determine output prefix
        args.prefix = determine_prefix(args.infile, getattr(args, "prefix", None))

        # Validate input files
        if not args.infile or not validate_input_files(args.infile):
            disp("Input file validation failed!")
            disp(f"Current input files: {args.infile}")
            return 1

        # Display processing information
        disp(f"Processing sample(s): {args.infile}")

        step_message = "\nAnalysis steps:\n"
        for step_num in args.step:
            if step_num in PROCESSING_STEPS:
                step_message += f"{step_num}. {PROCESSING_STEPS[step_num]}\n"
        disp(step_message)

        # Execute processing steps
        step_functions = {
            1: execute_trimming,
            2: execute_alignment,
            3: execute_mark_duplicates,
            4: execute_methylation_calling,
            5: execute_nucleosome_occupancy,
            6: execute_wps_calculation,
        }

        for step_num in args.step:
            if step_num in step_functions:
                step_functions[step_num](args)
            else:
                disp(f"WARNING: Unknown step number: {step_num}")

        disp("All processing steps completed successfully.")

        # Save updated configuration
        try:
            with open("./twist_init.json", "w") as f:
                # Convert args to dict, handling special objects
                args_dict = {}
                for key, value in vars(args).items():
                    try:
                        json.dumps(value)  # Test if value is JSON serializable
                        args_dict[key] = value
                    except (TypeError, ValueError):
                        args_dict[key] = str(
                            value
                        )  # Convert to string if not serializable
                json.dump(args_dict, f, indent=2)

        except IOError as e:
            disp(f"Warning: Failed to update initialization file: {e}")

        return 0

    except ProcessingError as e:
        disp(f"Processing error: {e}")
        return 1
    except KeyboardInterrupt:
        disp("Processing interrupted by user.")
        return 1
    except Exception as e:
        disp(f"Unexpected error during processing: {e}")
        return 1
