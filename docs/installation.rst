Installation
============

CFTK is still under active package development. The current repository is most
reliable when run directly from a source checkout.

Clone The Repository
--------------------

.. code-block:: bash

   git clone https://github.com/ChaorongC/CFTK.git
   cd CFTK

Create An Environment
---------------------

Use a Python environment that matches the target platform for your analysis.
The project metadata currently declares Python 3.9 or newer.

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip

Install Python dependencies according to the workflows you plan to run. The
core analysis modules use packages such as ``numpy``, ``pandas``, ``scipy``,
``scikit-learn``, ``matplotlib``, ``seaborn``, ``pysam``, ``pyBigWig``,
``bx-python``, ``statsmodels``, ``xgboost``, and ``finaletoolkit``.

Run From Source
---------------

Until the package metadata and console entry point are finalized, run CFTK from
the checkout:

.. code-block:: bash

   python src/cftk.py --help
   python src/cftk.py --config cftk_init.json init

External Tools
--------------

Many workflows call command-line tools that Python packaging does not install:

- ``trim_galore`` or ``fastp``
- ``bwameth.py`` or ``bismark``
- ``sambamba``, ``samtools``, ``picard``, or ``samblaster``
- ``MethylDackel`` or ``bismark_methylation_extractor``
- ``bedtools``
- ``multiqc``
- ``DANPOS``
- UCSC tools such as ``wigToBigWig`` and ``bigWigAverageOverBed``
- ``metilene`` for DMR analysis
- R packages used by DMR annotation, including ``annotatr`` and hg38 annotation
  packages

Install and validate these tools separately in the compute environment where
the pipeline will run.

Build The Documentation
-----------------------

.. code-block:: bash

   python -m pip install -r docs/requirements.txt
   python -m sphinx -b html docs docs/_build/html

The local HTML entry point is ``docs/_build/html/index.html``.
