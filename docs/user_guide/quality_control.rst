Quality Control
===============

The ``qc`` command runs configured quality-control analyses.

.. code-block:: bash

   python src/cftk.py --config cftk_init.json qc -s 1 2 3

QC Steps
--------

1. Methylation distribution
   Uses the merged CpG matrix from raw processing.

2. Fragment length distribution
   Uses BAM files resolved from mark-duplicate outputs, alignment outputs, or
   direct BAM paths in the sample config.

3. Dinucleotide frequency
   Requires ``reference_data.genome_fa`` and configured fragment settings.

Visualization
-------------

QC commands call the visualization layer after analysis. To regenerate plots
from existing outputs without rerunning QC:

.. code-block:: bash

   python src/cftk.py --config cftk_init.json vis --mode qc

Expected Output Location
------------------------

QC outputs are written under:

.. code-block:: text

   <output_dir>/results/2_qc/
