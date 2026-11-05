# Research protocol and interpretation

No actual research dataset was supplied. The application is ready to collect measurements; the test fixtures establish software behavior, not a scientific performance claim.

1. Define the prediction question, unit of observation, inference-time feature availability and target using domain knowledge.
2. Reserve a genuinely untouched confirmation dataset before exploratory diagnosis where possible. Full-dataset profiling can inform human choices and therefore weakens confirmatory claims on that same dataset.
3. Select random, entity-group, or chronological splitting as appropriate. Do not place repeated entities in independent partitions. Random splits also protect normalized identical-feature groups; approved exclusions are considered when forming groups.
4. Pre-specify the repair plan, algorithm/hyperparameters, primary metric, seeds and acceptable information loss. Review warnings and approve operations explicitly.
5. For each seed, run all three arms with shared row assignments. Only training rows fit preprocessing, clipping and models. Exact deduplication can remove C training rows, but never evaluation rows or conflicting-label records. Record changed feature sets and training counts.
6. Use validation outcomes to investigate repair choices. Do not select or tune using the displayed final test metrics. The application does not automatically search strategies or rank models using test results.
7. Export the complete report for every run, including unchanged and worse outcomes. Compare macro F1 for classification or RMSE for regression; choose additional metrics based on the use case. ROC-AUC may be unavailable if evaluation class coverage is inadequate.
8. Repeat across pre-specified seeds and datasets. History supports inspecting those runs but does not compute a cross-validation confidence interval. Distribution of seed results is descriptive and not automatically a valid independent-sample confidence interval.
9. Discuss computation, removed training repetitions, discarded feature information, sensitive tail behavior, uncalibrated probabilities, and the risks of over-cleaning. Confirm on independent data before claiming a general benefit.

## Implemented method boundaries

- Missingness: counts and recurring null patterns; no MCAR/MAR/MNAR diagnosis is claimed.
- Duplicates: exact and conservative string-normalized equality. Semantic fuzzy matching and entity discovery are not implemented.
- Anomalies: IQR fences and Isolation Forest (up to 40 numeric features, first 10,000 rows for exploratory fit). Scores indicate unusualness, not incorrectness.
- Label review: training-only stratified out-of-fold logistic probabilities, 3–5 folds when eligible; flags disagreement with maximum probability ≥0.8. Grouped/chronological and large training sets skip this analysis rather than use invalid folds. No labels are changed.
- Leakage: identifiers, exact target copies, training-only extreme regression correlation, and partition duplicate-overlap checks. These do not establish future-information availability or exclude important features automatically.
- Distribution: imbalance ratios, skewness, high cardinality, histograms and train–validation KS tests. P-values are exploratory and uncorrected for multiple comparisons.
- Repairs: explicit rule-based recommendations, no learned ranking or reinforcement learning. Export imputations are distinct from train-fitted model preprocessing.
- Quality score: 70% completeness and 30% exact-record uniqueness; other checks remain separate review evidence. It does not estimate predictive accuracy.
- Explainability: operation rationale, model configuration, permutation importance and evaluation diagnostics; no causal explanation or calibrated uncertainty is claimed.

## Example report interpretation template

Report the dataset provenance and task, selected features, missing-target exclusions, split strategy and row counts, approved repairs, baseline/standard/repaired metrics for validation and test separately, absolute and relative changes, changed training row counts, and software versions. Fill these from downloaded reports only. If C equals B, report no measured effect. If C is worse, report the reduction and possible mechanisms without inventing an explanation.
