# Provenance and interpretation

This release contains generated simulation-study records with arbitrary keys. The simulation mechanism is stylised; the release is not a transcription of official accident investigation reports and not a population-calibrated generator. Six public investigation reports are used only as qualitative mechanism references; they are not rows in the matrices and no population parameters are fitted from them.

The generator samples event context, latent factors, vehicles and exposed people, casualty outcomes and three loss components in that order. The four-level study-defined label is computed after casualties and loss. Recorded factor flags are created by an assessment process that does not directly read casualty or loss fields and is driven by event complexity, context and report quality; the primary classifier uses those recorded flags.

The fixed release has 30 replicates and three paired scenarios. The 2,000-row external cohort is independent of model selection. Monte Carlo intervals describe uncertainty in mean performance under the specified data-generating process, not uncertainty about a real accident population. There are no real dates, report identities, exposure denominators or external validation data in this release.

No real event identity was anonymised: event keys were generated. The public-report ledger is qualitative and source access is uneven; those reports are not used for calibration or validation.
