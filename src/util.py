import pandas as pd
import numpy as np
import time, sys, os, json, argparse, subprocess, re

steps = {
    1: "trimming(trim_galore)",
    2: "alignment and sorting(bwameth and Samtools)",
    3: "mark duplicates(Picard)",
    4: "methylation ratio calling(MethylDackel)",
    5: "nucleosome occupancy calculation(DANPOS2)",
}
message = ""
command = ""


def disp(txt):
    print("@%s \t%s" % (time.asctime(), txt), file=sys.stderr)


def init(args):
    with open("./twist_init.json", "w") as f:
        json.dump(args.__dict__, f, indent=2)
    if args.output_dir == os.getcwd():
        disp(
            "Output directory is not sepecified. Using current directory: %s"
            % args.output_dir
        )


def Merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res


def process(args):
    # check initilization
    if not os.path.exists("./twist_init.json"):
        disp("Initialization infomation is not found. Initilization required.")
        return 1
    # store json informaiton
    if not args.prefix:
        args.prefix = re.split("_R1_001|.R1_001", args.infile[0], flags=re.IGNORECASE)[
            0
        ]
        disp(
            "Prefix for output is not specified. Using input files' name as prefix: %s\n"
            % args.prefix,
        )
    args.__dict__ = Merge(json.load(open("./twist_init.json", "r")), args.__dict__)
    with open("./twist_init.json", "w") as f:
        json.dump(args.__dict__, f, indent=2)
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
    if 1 in args.step:
        disp("Start: %s" % steps[1])
        output_dir = args.trimgalore_output_dir
        if os.path.exists(args.trimgalore_output_dir):
            disp("Outputting to: %s" % args.trimgalore_output_dir)
        else:
            disp("%s doesn't exist. Creating it for you." % args.trimgalore_output_dir)
            os.mkdir(args.trimgalore_output_dir)
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
            disp("Running: %s\n" % command)
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
            disp("Running: %s\n" % command)
            os.system(command)
        disp("Complete: %s" % steps[1])

    if 2 in args.step:
        disp("Start: %s" % steps[2])
        output_dir = args.bwameth_output_dir
        if os.path.exists(args.bwameth_output_dir):
            disp("Outputting to: %s" % args.bwameth_output_dir)
        else:
            disp("%s doesn't exist. Creating it for you." % args.bwameth_output_dir)
            os.mkdir(args.bwameth_output_dir)
        disp("Output:\n")
        print(
            "%s.bam" % args.prefix,
            file=sys.stderr,
        )
        if 1 in args.step:
            r1_input = (
                str(args.trimgalore_output_dir).strip() + "/%s_val_1.fq" % args.prefix
            )
            r2_input = (
                str(args.trimgalore_output_dir).strip() + "/%s_val_2.fq" % args.prefix
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
        disp("Running: %s\n" % command)
        os.system(command)
        disp("Complete: %s" % steps[2])

    if 3 in args.step:
        disp("Start: %s" % steps[3])
        output_dir = args.picard_output_dir
        if os.path.exists(args.picard_output_dir):
            disp("Outputting to: %s" % args.picard_output_dir)
        else:
            disp("%s doesn't exist. Creating it for you." % args.picard_output_dir)
            os.mkdir(args.picard_output_dir)
        disp("Output:\n")
        print(
            "%s.markup.bam" % args.prefix,
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
                "%s MarkDuplicates I=%s O=%s/%s.markup.bam R=%s M=%s/%s.markdup_raw_metrics \
                SORTING_COLLECTION_SIZE_RATIO=0.15 ASSUME_SORT_ORDER=coordinate \
                OPTICAL_DUPLICATE_PIXEL_DISTANCE=2500 MAX_RECORDS_IN_RAM=1000 %s || exit 1"
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
            )
        else:
            command = (
                "%s MarkDuplicates I=%s O=%s/%s.markup.bam R=%s M=%s/%s.markdup_raw_metrics \
                SORTING_COLLECTION_SIZE_RATIO=0.15 ASSUME_SORT_ORDER=coordinate \
                OPTICAL_DUPLICATE_PIXEL_DISTANCE=2500 MAX_RECORDS_IN_RAM=1000 || exit 1"
                % (
                    args.picard_jar_path,
                    bam_input,
                    args.picard_output_dir,
                    args.prefix,
                    args.ref,
                    args.picard_output_dir,
                    args.prefix,
                )
            )
        disp("Running: %s\n" % command)
        os.system(command)
        disp("Complete: %s" % steps[3])

    if 4 in args.step:
        disp("Start: %s" % steps[4])
        output_dir = args.methyldackel_output_dir
        if os.path.exists(args.methyldackel_output_dir):
            disp("Outputting to: %s" % args.methyldackel_output_dir)
        else:
            disp(
                "%s doesn't exist. Creating it for you." % args.methyldackel_output_dir
            )
            os.mkdir(args.methyldackel_output_dir)
        disp("Output:\n")
        print(
            "%s_CpG.bedGraph" % args.prefix,
            file=sys.stderr,
        )
        if 3 in args.step:
            bam_input = (
                str(args.picard_output_dir).strip() + "/%s.markup.bam" % args.prefix
            )
        else:
            message = "Processing sample(s):\n"
            print(message, file=sys.stderr)
            print(args.infile, file=sys.stderr)
            bam_input = args.infile[0]
        if args.methyldackel_args:
            command = (
                "MethylDackel mbias -@ %s %s %s/%s.markup.bam %s/%s &> %s/%s_mbias_OT_OB.temp || exit 1;"
                % (
                    args.cores,
                    args.ref,
                    args.picard_output_dir,
                    args.prefix,
                    args.methyldackel_output_dir,
                    args.prefix,
                    args.methyldackel_output_dir,
                    args.prefix,
                )
                + "MethylDackel extract --minDepth 10 --maxVariantFrac 0.25 -@ %s --OT $(cat %s/%s_mbias_OT_OB.temp | \
                grep -oP '(?<=--OT )[^ ]+') --OB $(cat %s/%s_mbias_OT_OB.temp | \
                grep -oP '(?<=--OB )[^ ]+') -o %s/%s --mergeContext %s \
                %s %s/%s.markup.bam || exit 1;"
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
                    args.methyldackel_output_dir,
                    args.prefix,
                )
            )
        else:
            command = (
                "MethylDackel mbias -@ %s %s %s/%s.markup.bam %s/%s &> %s/%s_mbias_OT_OB.temp || exit 1;"
                % (
                    args.cores,
                    args.ref,
                    args.picard_output_dir,
                    args.prefix,
                    args.methyldackel_output_dir,
                    args.prefix,
                    args.methyldackel_output_dir,
                    args.prefix,
                )
                + "MethylDackel extract --minDepth 10 --maxVariantFrac 0.25 -@ %s --OT $(cat %s/%s_mbias_OT_OB.temp | \
                grep -oP '(?<=--OT )[^ ]+') --OB $(cat %s/%s_mbias_OT_OB.temp | \
                grep -oP '(?<=--OB )[^ ]+') -o %s/%s --mergeContext \
                %s %s/%s.markup.bam || exit 1;"
                % (
                    args.cores,
                    args.methyldackel_output_dir,
                    args.prefix,
                    args.methyldackel_output_dir,
                    args.prefix,
                    args.methyldackel_output_dir,
                    args.prefix,
                    args.ref,
                    args.methyldackel_output_dir,
                    args.prefix,
                )
            )
        disp("Running: %s\n" % command)
        os.system(command)
        disp("Complete: %s" % steps[4])
        
    if 5 in args.step:
        disp("Start: %s" % steps[4])
        output_dir = args.danpos_output_dir
        if os.path.exists(args.danpos_output_dir):
            disp("Outputting to: %s" % args.danpos_output_dir)
        else:
            disp(
                "%s doesn't exist. Creating it for you." % args.danpos_output_dir
            )
            os.mkdir(args.danpos_output_dir)
        disp("Output:\n")
        print(
            "%s_CpG.bedGraph" % args.prefix,
            file=sys.stderr,
        )
        if 3 in args.step:
            bam_input = (
                str(args.picard_output_dir).strip() + "/%s.markup.bam" % args.prefix
            )
        else:
            message = "Processing sample(s):\n"
            print(message, file=sys.stderr)
            print(args.infile, file=sys.stderr)
            bam_input = args.infile[0]
        disp("Completing all processes.")


"""
sambamba_filter='not failed_quality_control and not supplementary and proper_pair and mapping_quality > 0'
r1_input=${trim_galore_output_dir}/${sample_id}/Apps_1063_UCI_Li_Test_${sample_id}_R1_001_val_1.fq.gz
r2_input=${trim_galore_output_dir}/${sample_id}/Apps_1063_UCI_Li_Test_${sample_id}_R2_001_val_2.fq.gz

bwameth.py --reference ${hg38_ref} \
-t 40 \
$r1_input $r2_input |
sambamba view -t 40 -F 'not secondary_alignment and not failed_quality_control and not supplementary and proper_pair and mapping_quality > 0' -f bam -S -l 0 /dev/stdin |
sambamba sort -t 40 -m 200G -o ${bwameth_output_dir}/${sample_id}_sorted.bam /dev/stdin || exit 1;

samtools index -@ 40 ${bwameth_output_dir}/${sample_id}_sorted.bam;

echo ${sample_id} aligning and sorting completed;
date;



picard MarkDuplicates \
I=${bwameth_output_dir}/${sample_id}_sorted.bam \
O=${bwameth_output_dir}/${sample_id}_sorted.markdup.bam \
R=${hg38_ref} \
M=${bwameth_output_dir}/${sample_id}_sorted.picard_markdup_raw_metrics \
SORTING_COLLECTION_SIZE_RATIO=0.15 \
ASSUME_SORT_ORDER=coordinate \
OPTICAL_DUPLICATE_PIXEL_DISTANCE=2500 \
MAX_RECORDS_IN_RAM=1000 || exit 1;
#--REMOVE_DUPLICATES=true

samtools index -@ 40 ${bwameth_output_dir}/${sample_id}_sorted.markdup.bam;

MethylDackel mbias -@ 40 ${hg38_ref} ${bwameth_output_dir}/${sample_id}_sorted.markdup.bam ${methyldackel_output_dir}/${sample_id}_methylome &> ${methyldackel_output_dir}/${sample_id}_mbias_OT_OB.temp || exit 1;

OT=$(cat ${methyldackel_output_dir}/${sample_id}_mbias_OT_OB.temp | grep -oP '(?<=--OT )[^ ]+');
OB=$(cat ${methyldackel_output_dir}/${sample_id}_mbias_OT_OB.temp | grep -oP '(?<=--OB )[^ ]+');

MethylDackel extract 
--minDepth 10 \
--maxVariantFrac 0.25 \
-@ 40 \
--OT $OT \
--OB $OB \
-o ${methyldackel_output_dir}/${sample_id}_methylome \
--mergeContext \
${hg38_ref} \
${bwameth_output_dir}/${sample_id}_sorted.markdup.bam || exit 1;



files=$(find /dfs4/weil21-lab1/chaoronc/project/2023-Prostate_cancer_recurrence/result/4.methyldackel -type f -name "*_methylome_processed.bedgraph")
for file in $files
do
  bedtools intersect -b /dfs4/weil21-lab1/chaoronc/project/2023-Prostate_cancer_recurrence/data/TWIST/covered_targets_Twist_Methylome_hg38_annotated_collapsed.bed -a ${file} | awk -v OFS="\t" '{print $1"_"$2"_"$3, $4}' > ${file}.overlap
done


module load bedtools2

bedtools intersect -b ~/project/2023-Prostate_cancer_recurrence/data/TWIST/covered_targets_Twist_Methylome_hg38_annotated_collapsed.bed -a /dfs4/weil21-lab1/chaoronc/project/2023-Prostate_cancer_recurrence/result/4.methyldackel/A11_S1_methylome_processed.bedgraph > /dfs4/weil21-lab1/chaoronc/project/2023-Prostate_cancer_recurrence/result/4.methyldackel/A11_S1_methylome_processed.overlapped.bedgraph


danpos_dir='/dfs5/weil21-lab2/chaoronc/tool/DANPOS3'

python ${danpos_dir}/danpos.py dpos ${bwameth_output_dir}/${sample_id}_sorted.markdup.bam \
--paired 1 \
-u 0 \
-c 1000000 \
-o ${danpos_output_dir} || exit 1;

wigToBigWig -clip ${danpos_output_dir}/pooled/${sample_id}*wig /dfs5/weil21-lab2/chaoronc/reference_genome/hg38/hg38.chrom.sizes ${danpos_output_dir}/pooled/${sample_id}.bw || exit 1;

region_file=/dfs4/weil21-lab1/chaoronc/project/2023-Prostate_cancer_recurrence/data/TWIST/covered_targets_Twist_Methylome_hg38_annotated_collapsed_TSS_PAS_1kb_.bed

bigWigAverageOverBed ${danpos_output_dir}/pooled/${sample_id}.bw $region_file ${danpos_output_dir}/${sample_id}.occupancy.tsv || exit 1;

"""

# trim_galore --paired --fastqc --fastqc_args "-t 40" --2colour 20 --clip_R1 10 --clip_R2 10 --three_prime_clip_R1 5 --three_prime_clip_R2 5 --cores 40 -o ${trim_galore_output_dir}/${sample_id} R1_001.fastq.gz R2_001.fastq.gz || exit 1;
# trim_galore --paired --2colour 20 --clip_R1 10 --clip_R2 10 --three_prime_clip_R1 5 --three_prime_clip_R2 5 --cores 40 -o ${trim_galore_output_dir}/${sample_id} R1_001.fastq.gz R2_001.fastq.gz || exit 1;
