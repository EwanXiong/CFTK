Modeling, Visualization, And Reports
====================================

MESA Modeling
-------------

CFTK includes MESA-style multimodal modeling commands.

Run modality performance screening:

.. code-block:: bash

   python src/cftk.py --config cftk_init.json mesa --performance

Run model construction:

.. code-block:: bash

   python src/cftk.py --config cftk_init.json mesa --mesa-model

Run leave-one-out cross-validation and plots:

.. code-block:: bash

   python src/cftk.py --config cftk_init.json mesa --loocv

Commonly used together:

.. code-block:: bash

   python src/cftk.py --config cftk_init.json mesa --performance --mesa-model --loocv

Visualization Regeneration
--------------------------

Use ``vis`` to regenerate figures from existing result files:

.. code-block:: bash

   python src/cftk.py --config cftk_init.json vis --mode all
   python src/cftk.py --config cftk_init.json vis --mode power qc diff

Report Generation
-----------------

Generate a self-contained HTML report:

.. code-block:: bash

   python src/cftk.py --config cftk_init.json report

Reports are written under:

.. code-block:: text

   <output_dir>/results/report/

End-To-End Runs
---------------

The ``run-all`` command runs the configured pipeline end to end:

.. code-block:: bash

   python src/cftk.py --config cftk_init.json run-all

Because ``run-all`` continues after some failures, review logs and expected
artifacts before treating a run as complete.
