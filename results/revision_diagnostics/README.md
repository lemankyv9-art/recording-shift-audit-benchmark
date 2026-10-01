# Sensitivity-extension diagnostics

These files are secondary analyses added after the v3.2 primary protocol. They do not overwrite the fixed matrices or primary predictions.

- `calibration_summary.csv`: calibration-in-the-large, calibration slope, severe AUROC and severe AP summaries for the fixed composition-shift selection rows.
- `threshold_independent_metrics.csv` and `composition_shift_summary.csv`: AUROC, average precision, balanced accuracy, skill scores and top-decile coverage for the fixed release.
- `factorial_results.csv`: 21,600 rows from 72 crossed sensitivity cells and 30 replicates per cell.
- `factorial_effects.csv`: seed-level paired contrasts; intervals use 30 independent seeds.
- `selection_summary.csv` and `selection_paired.csv`: fixed, development-CV and independent-validation selection comparisons.
- `oracle_summary.csv`: data-rich references and wide finite Monte Carlo conditional-pattern intervals. These are not analytic Bayes bounds.

The sensitivity runners and generation utilities remain in the internal construction workspace; this delivery contains fixed inputs and compact outputs only.

- `data_rich_class_recall.csv`: high/major argmax recall recomputed from all five 200,000-row logistic fits and their 50,000-row evaluation predictions; the source prediction hashes identify the separate construction-workspace files.
- `primary_cost_metrics.csv`, `source_adjustment_comparison.csv` and `reporting_audit.json`: directly recomputed decision and paired-comparison diagnostics used in the revised text.
