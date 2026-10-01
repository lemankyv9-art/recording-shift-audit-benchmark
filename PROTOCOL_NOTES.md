# Notes on the frozen v3.2 protocol

The JSON protocol is retained byte-for-byte because the primary result files record its hash. The prose below uses the simulation-study terminology used in the manuscript. These interpretation notes do not change its executable parameters or the fixed data.

- The key `covariate_shift` denotes **composition shift** in the article. It shifts both event context and factor logits. It is not guaranteed to be pure covariate shift given the 20 recorded flags.
- `conditional_shift` denotes **injury-mechanism shift**, with the same contexts, latent factors and recording masks as composition shift.
- “Pre-outcome” in the frozen description means before casualty/loss outcomes. Fire, rollover and exposure enter the recording model; it is not a pre-crash feature set. The phrase “outcome-blind” in the frozen JSON is retained for hash compatibility; the article states that the primary mask does not directly read casualties or loss, while allowing association with severity through shared event complexity. The extension adds a separate label-dependent recording stress test.
- `generator.context_probabilities` is an unused field. v3.2 uses `context_logits`, `region_effects`, latent conditions and the specified scenario offsets.
- The loss invariant means that unrounded components sum to raw loss, and reported loss equals raw loss after the specified mixed rounding. Four-decimal component storage can differ from the eight-decimal stored raw sum by at most 0.00015 million CNY.
- Primary weighting retains all source rows. Development cross-validation chooses model regularisation and TCA parameters without using holdout or independent-cohort labels/features.
- The construction audit and representative held-out perturbation audit are in `results/generation_boundary_audit.json`. The generation utility itself is outside this release. The audit is internal reproducibility evidence, not validation of population realism.
- The protocol was developed iteratively. The final set retains every seed; it is not an independently preregistered confirmatory experiment.
