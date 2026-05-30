#!/usr/bin/env Rscript
# DMR annotation using annotatr (hg38).
# Usage: Rscript dmr_annotation.r <dmr_raw.bed> <dmr_annotated.bed>

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
    stop("Usage: Rscript dmr_annotation.r <input.bed> <output.bed>")
}

input_bed  <- args[1]
output_bed <- args[2]

suppressPackageStartupMessages({
    library(annotatr)
    library(TxDb.Hsapiens.UCSC.hg38.knownGene)
    library(GenomicRanges)
})

# read metilene output: chr start end q-value mean_diff #CpGs p_MWU p_2DKS mean_g1 mean_g2
dmr <- read.table(input_bed, header = FALSE, sep = "\t", stringsAsFactors = FALSE)
col_names <- c("chr","start","end","q_value","mean_diff","n_CpGs",
               "p_MWU","p_2DKS","mean_g1","mean_g2")
colnames(dmr) <- col_names[seq_len(ncol(dmr))]

dm_regions <- makeGRangesFromDataFrame(
    dmr,
    keep.extra.columns = TRUE,
    ignore.strand       = TRUE,
    seqnames.field      = "chr",
    start.field         = "start",
    end.field           = "end"
)

annots <- c(
    "hg38_cpgs",
    "hg38_basicgenes",
    "hg38_genes_5UTRs",
    "hg38_genes_3UTRs",
    "hg38_genes_exons",
    "hg38_genes_intronexonboundaries",
    "hg38_genes_1to5kb",
    "hg38_genes_promoters",
    "hg38_cpg_islands",
    "hg38_genes_cds",
    "hg38_genes_introns",
    "hg38_cpg_shores",
    "hg38_cpg_shelves",
    "hg38_enhancers_fantom"
)

annotations   <- build_annotations(genome = "hg38", annotations = annots)
dm_annotated  <- annotate_regions(regions = dm_regions,
                                   annotations = annotations,
                                   ignore.strand = TRUE,
                                   quiet = FALSE)

write.table(dm_annotated, output_bed, sep = "\t", quote = FALSE, row.names = FALSE)
cat(sprintf("[done] annotated DMRs written to: %s\n", output_bed))
