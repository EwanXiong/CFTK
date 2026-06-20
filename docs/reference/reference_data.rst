Reference Data
==============

CFTK needs both bundled region resources and user-supplied genome resources.

Bundled Files
-------------

The repository keeps these reference assets:

- ``data/hg38.chrom.sizes``
- ``data/hg38_annotated_collapsed_TSS_PAS_1kb.bed``
- ``data/covered_targets_Twist_Methylome_hg38_annotated_collapsed.bed``
- ``data/manifest.json`` and companion ``data/*.npy`` / ``data/*.npz`` arrays
  for model-level power analysis.

The example config still uses placeholder paths. Replace the placeholders with
paths that are valid in your environment.

User-Supplied Files
-------------------

Most full workflows require additional references:

- ``genome_fa``: hg38 FASTA used by alignment, methylation extraction, and some
  QC steps.
- ``genome_2bit``: 2bit genome used by some fragmentomics workflows.
- ``ctcf_bed``: CTCF regions for cleavage.
- ``blacklist`` and ``gap``: excluded regions for DELFI-style features.
- ``bins``: genomic bins for DELFI-style features.
- ``cpg_std``: CpG standard deviation table for power analysis.

Coordinate Safety
-----------------

Do not mix genome builds or chromosome naming conventions. Keep FASTA,
chromosome sizes, BED files, bins, blacklist, and gap files on the same build
and naming scheme.
