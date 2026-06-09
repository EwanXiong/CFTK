CFTK Documentation
==================

.. rst-class:: cftk-hero

CFTK is a cfDNA multimodal epigenetic analysis toolkit for raw processing,
quality control, differential methylation, fragmentomics, multimodal modeling,
visualization, and report generation.

The current workflow is driven by ``cftk_init.json`` and the ``cftk`` command
implemented in ``src/cftk.py``. During active development, the most reliable
way to run the command from a checkout is:

.. code-block:: bash

   python src/cftk.py --help

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Get Started
      :link: getting_started
      :link-type: doc

      Set up a checkout, validate the config, and run the first commands.

   .. grid-item-card:: Configure A Project
      :link: user_guide/configuration
      :link-type: doc

      Understand samples, references, tools, analysis settings, and outputs.

   .. grid-item-card:: Run Workflows
      :link: user_guide/index
      :link-type: doc

      Process raw data, run QC, fragmentomics, differential analysis, MESA,
      visualization, and report generation.

   .. grid-item-card:: Command Reference
      :link: reference/cli
      :link-type: doc

      See the available ``cftk`` commands and common command patterns.

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Start

   installation
   getting_started

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: User Guide

   user_guide/index

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Reference

   reference/index

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Project

   development
