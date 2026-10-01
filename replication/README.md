# v3.2 replication

Run from the package root. The scripts use `data/datasets` and write to `results` by default; the supplied rows are fixed simulation-study records.

```sh
python replication/verify_integrity.py
python replication/verify_secondary.py
python replication/verify_reporting.py
```

`analysis_pipeline.py` refits the fixed primary protocol when `BENCHMARK_RESULTS` points to a separate output directory. `supplementary_analysis.py` creates the 900 exploratory learning-curve fits. `build_publication_figures.py` regenerates the seven publication figures from fixed matrices and result summaries, without fitting models or generating events. `build_report_figures.py` and `build_report_tables.py` export legacy TeX fragments and numeric summaries under `results/revision_diagnostics/`; neither changes manuscript sources. Use `build_publication_figures.py` for the current vector figures. Generation utilities remain in the separate construction workspace; this release verifies the supplied fixed rows and results.

## Sensitivity-extension protocol

The sensitivity-extension factorial design is specified in `extension_protocol.json` and its fixed outputs are included in the package. `revision_diagnostics.py` reads the fixed release and computes threshold-independent metrics, recording-rate gradients, external domain AUC and selection diagnostics without refitting or regenerating the primary analysis.

`verify_reporting.py` recomputes primary cost-threshold metrics and the source-fixed versus intercept-adjusted comparison, checks the calibration and constant-decision cost tables against the stored replicate metrics, and independently reconstructs factorial contrasts and Monte Carlo intervals from the 30 seed-level means.

## Audit-extension diagnostics

`audit_extension_analysis.py` reads only the stored predictions and released summaries. It writes `results/audit_extension/` (the integer-grid cost-ratio sweep, the method-by-shift matrix, the recording-coverage gradient, severe-risk calibration by scenario and the cross-fitted recalibration sweep) and redraws `manuscript/figures/decision_curve.pdf`. The recalibration sweep uses independent-test labels and is an oracle decomposition, not a primary result.
