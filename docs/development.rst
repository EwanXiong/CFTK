Development
===========

Documentation Development
-------------------------

Install documentation dependencies:

.. code-block:: bash

   python -m pip install -r docs/requirements.txt

Build HTML locally:

.. code-block:: bash

   python -m sphinx -b html docs docs/_build/html

Treat warnings as failures when preparing docs for release:

.. code-block:: bash

   python -m sphinx -W -b html docs docs/_build/html

Read The Docs
-------------

Read the Docs uses ``.readthedocs.yaml`` and builds from ``docs/conf.py``.
The docs dependency set is intentionally small and independent from the full
analysis stack so the website can build before package installation is
finalized.

Code Development Notes
----------------------

- Keep command behavior and config keys stable unless a migration is planned.
- Call out changes to coordinate conventions, filtering, sorting, NA handling,
  random seeds, or model defaults.
- Validate pipeline changes with command exit status and expected artifacts, not
  only successful job submission.
- Avoid committing generated reports, BAM/FASTQ files, docs build outputs, or
  local result directories.
