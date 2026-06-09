Getting Started
===============

CFTK is organized around one project configuration file and a command-line
dispatcher. Edit ``cftk_init.json`` for your cohort, then run the commands that
match the analysis stage you need.

1. Validate The Config
----------------------

.. code-block:: bash

   python src/cftk.py --config cftk_init.json init

This checks required top-level sections, sample definitions, comparison group
names, and required reference fields.

2. Inspect Commands
-------------------

.. code-block:: bash

   python src/cftk.py --help

The major commands are:

- ``init``: validate and summarize ``cftk_init.json``.
- ``process``: run raw processing steps 1 through 4.
- ``qc``: run methylation, fragment length, or dinucleotide QC.
- ``power``: run statistical power analysis.
- ``diff``: run PCA, differential testing, and summary plots.
- ``dmr``: run DMR preparation, metilene, annotation, and plotting.
- ``frag``: run occupancy, WPS, DELFI, end motif, and cleavage workflows.
- ``mesa``: run modality performance and multimodal MESA modeling.
- ``merge``: build feature matrices from user-provided files.
- ``vis``: regenerate plots from existing results.
- ``report``: generate a self-contained HTML report.
- ``run-all``: run the configured end-to-end workflow.

3. Run Raw Processing
---------------------

After replacing the example paths in ``cftk_init.json``:

.. code-block:: bash

   python src/cftk.py --config cftk_init.json process -s 1 2 3 4

The process command creates standard subdirectories under
``<output_dir>/results/1_process`` and merges per-sample CpG calls into
``cpg_matrix.tsv`` after successful methylation calling.

4. Run Downstream Workflows
---------------------------

Examples:

.. code-block:: bash

   python src/cftk.py --config cftk_init.json qc -s 1 2 3
   python src/cftk.py --config cftk_init.json diff
   python src/cftk.py --config cftk_init.json frag --wps
   python src/cftk.py --config cftk_init.json mesa --performance --mesa-model --loocv
   python src/cftk.py --config cftk_init.json report

Use ``run-all`` only after individual stages and tool paths have been validated
on representative data.
