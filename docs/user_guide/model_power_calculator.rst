Model-Development Power Calculator
==================================

The CFTK model-development power calculator asks a study-design question:
for a proposed liquid-biopsy biomarker-discovery cohort, what is the
probability that the internal cross-validated model reaches a target AUC?

The reported primary endpoint is:

.. code-block:: text

   power = P(out-of-fold cross-validated AUC >= target AUC)

This is an internal model-development adequacy calculation. It does not
estimate external cohort performance, and independent validation is still
required before claims of generalizability.

Discovery Workflow
------------------

Each simulated study uses the total cohort sample size entered in the app.
The samples are split with stratified cross-validation. Within every training
fold, CFTK applies the full biomarker-discovery workflow:

1. training-fold-only missingness filtering;
2. training-fold-only median imputation;
3. zero-variance filtering;
4. training-fold-only ANOVA F ranking;
5. fixed top-k CpG selection;
6. training-fold-only standardization;
7. logistic-regression fitting;
8. held-out-fold prediction;
9. combined out-of-fold AUC calculation.

All preprocessing, feature ranking, feature selection, scaling, and model
fitting are repeated inside each training fold. The public calculator uses
logistic regression as a fixed model-development pipeline.

User-Adjustable Assumptions
---------------------------

The app exposes study-design settings for total sample sizes, case-to-control
ratio, mean sequencing depth, and target cross-validated AUC. Biomarker
assumptions include the number of candidate CpGs, number of true signal CpGs,
mean absolute methylation difference, and selected CpGs per fold.

Advanced settings control effect-size variability, effect direction,
within-block correlation, cross-validation folds, minimum observed fraction,
the specificity operating point for descriptive sensitivity reporting, and
the empirical SD uncertainty scenario. Lower SD estimates are generally more
optimistic, while upper SD estimates are generally more conservative.

Sequencing depth enters through empirical CpG variance and missingness
patterns derived from the bundled aggregate reference arrays. The calculator
does not use or expose patient-level data.

Interpreting Outputs
--------------------

``power`` is the proportion of simulated studies whose out-of-fold
cross-validated AUC reaches the selected target. ``mean_cv_auc`` summarizes
the average internal CV AUC across simulated studies.

Sensitivity at the selected specificity is reported as a secondary descriptive
operating-point metric. It does not enter the primary power definition.

Feature recall and precision are simulation diagnostics because the simulated
true signal CpGs are known. Selection Jaccard summarizes fold-to-fold
feature-set overlap and can help identify unstable feature selection in small
or noisy designs.

Interactive App
---------------

.. Replace this placeholder after the Streamlit Community Cloud app is deployed.

.. raw:: html

   <iframe
      src="https://REPLACE-WITH-CFTK-APP.streamlit.app"
      width="100%"
      height="1300px"
      style="border: 0;"
      title="CFTK Model-Development Power Calculator">
   </iframe>

Standalone app:
`https://REPLACE-WITH-CFTK-APP.streamlit.app <https://REPLACE-WITH-CFTK-APP.streamlit.app>`__

Reproducibility
---------------

The app provides downloads for the power curve, optional replicate-level
results, and a settings JSON file. The settings export includes the user
inputs, selected precision mode, deterministic seeds, and fixed computation
settings needed to reproduce the calculation.
