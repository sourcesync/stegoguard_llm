# StegoGuard-LM Step 11 Implementation

Protocol: `step11_system_analysis_v1`

Step 11 is a read-only, model-free analysis layer over the final frozen Step 10 evidence.

## Added package

`src/stegoguard_lm/analysis/`

- `io.py`: configuration, frozen-input verification, JSON/CSV helpers.
- `statistics.py`: deterministic parent-level bootstrap confidence intervals and paired contrasts.
- `system.py`: round, trajectory, contrast, transition, family, residual, recovery, resilience, history, stratified, clean, and hypothesis metrics.
- `reporting.py`: result package writing, figures, interpretation and QA text.

## Added scripts

- `verify_step11_inputs.py`
- `freeze_step11_analysis_spec.py`
- `verify_step11_analysis_spec.py`
- `smoke_test_step11.py`
- `run_step11_system_analysis.py`
- `validate_step11_results.py`
- `freeze_step11_analysis.py`
- `verify_step11_freeze.py`

## Design guarantees

1. Step 11 refuses to run unless Step 10 is finally frozen and all Step 10 hashes verify.
2. Step 11 never writes into the Step 10 adaptive output tree.
3. Primary comparison definitions are frozen before Step 11 result generation.
4. Bootstrap resampling is by clean parent and uses a deterministic seed.
5. Final result validation checks row counts, figure counts, hypothesis artifacts, and provenance before final freeze.
6. The final Step 11 freeze records hashes of the complete analysis package and all Step 11 implementation files.
