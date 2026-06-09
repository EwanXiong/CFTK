Configuration
=============

CFTK reads project settings from ``cftk_init.json``. Pass a different config
with ``--config``:

.. code-block:: bash

   python src/cftk.py --config path/to/cftk_init.json init

Top-Level Sections
------------------

``project_name``
   Short project label used in summaries and reports.

``output_dir``
   Work directory where CFTK creates the ``results`` tree.

``comparison``
   Two-group comparison name formatted as ``GroupA_vs_GroupB``. The group names
   must match keys under ``samples``.

``samples``
   Two groups of samples. Each sample needs ``name`` and ``input_type``. FASTQ
   samples need ``r1`` and ``r2``. BAM samples need ``bam``.

``reference_data``
   Reference genome and annotation paths used by processing and fragmentomics.

``process``
   Tool and parameter settings for raw processing steps.

``analysis``
   Parameters for QC, power, differential analysis, DMR, fragmentomics, and
   MESA workflows.

Sample Definitions
------------------

.. code-block:: json

   {
     "samples": {
       "Control": [
         {
           "name": "control_1",
           "input_type": "fastq",
           "r1": "/data/control_1_R1.fq.gz",
           "r2": "/data/control_1_R2.fq.gz"
         }
       ],
       "sALS": [
         {
           "name": "sALS_1",
           "input_type": "bam",
           "bam": "/data/sALS_1.markdup.bam"
         }
       ]
     }
   }

Sample names should use letters, digits, hyphens, underscores, or periods. CFTK
uses sample names to construct output filenames and merged matrix columns.

Reference Data
--------------

Required reference keys validated by ``init``:

- ``genome_fa``
- ``genome_2bit``
- ``chrom_sizes``

Common optional keys used by downstream commands:

- ``tss_pas_bed`` for WPS and occupancy region summaries.
- ``ctcf_bed`` for cleavage workflows.
- ``blacklist``, ``gap``, and ``bins`` for DELFI-style features.
- ``cpg_std`` for power analysis.

Processing Tools
----------------

The raw processing block configures four steps:

.. code-block:: json

   {
     "process": {
       "step1_trimming": {"tool": "trim_galore", "params": {"cores": 20}},
       "step2_alignment": {"tool": "bwameth", "params": {"cores": 20}},
       "step3_markdup": {"tool": "sambamba", "params": {"cores": 20}},
       "step4_methylation": {"tool": "methyldackel", "params": {"cores": 20}}
     }
   }

Supported tools are defined in ``src/process.py``:

- Step 1: ``trim_galore`` or ``fastp``.
- Step 2: ``bwameth`` or ``bismark``.
- Step 3: ``sambamba``, ``picard``, or ``samblaster``.
- Step 4: ``methyldackel`` or ``bismark_extractor``.
