Configuration Schema
====================

The example config is ``cftk_init.json`` at the repository root. It is JSON and
does not support comments.

Required Top-Level Keys
-----------------------

- ``project_name``
- ``output_dir``
- ``comparison``
- ``samples``
- ``reference_data``
- ``process``
- ``analysis``

Required Reference Keys
-----------------------

- ``genome_fa``
- ``genome_2bit``
- ``chrom_sizes``

Comparison Format
-----------------

``comparison`` must use ``GroupA_vs_GroupB``. Both group names must exist under
``samples``.

.. code-block:: json

   {
     "comparison": "Control_vs_sALS",
     "samples": {
       "Control": [],
       "sALS": []
     }
   }

Output Paths
------------

CFTK derives output paths from ``output_dir``. The helper function
``get_work_paths`` in ``src/init.py`` defines the canonical layout.

Analysis Blocks
---------------

The ``analysis`` section controls downstream workflows:

- ``qc.params`` for QC settings.
- ``power.params`` for sample size, effect size, depth, and power plots.
- ``diff.params`` for modalities, colors, and heatmap feature counts.
- ``dmr`` for metilene and DMR annotation settings.
- ``frag`` for occupancy, WPS, DELFI, end motif, and cleavage settings.
- ``mesa.params`` for modalities, classifiers, feature size, subset, and repeat.

Example File
------------

The full example file is included in the repository as ``cftk_init.json``.
