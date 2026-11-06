# StegoGuard-LM Step 12 Implementation

Protocol: `step12_robustness_v1`

Step 12 adds a new read-only package under `src/stegoguard_lm/robustness/` and never writes into frozen Step 8–11 output directories.

## Package

- `io.py` — Step 12 config loading, frozen-input verification, Step 10 row loading, CSV/JSON/hash utilities.
- `statistics.py` — parent aggregation, paired bootstrap intervals, paired sign-flip randomization tests, BH-FDR adjustment, deletion-stability summaries.
- `analysis.py` — all Step 12 robustness analyses and claim-stability assessment.
- `reporting.py` — result tables, figures, interpretation and QA artifacts.

## Scripts

- `verify_step12_inputs.py`
- `freeze_step12_analysis_spec.py`
- `verify_step12_analysis_spec.py`
- `smoke_test_step12.py`
- `run_step12_robustness_analysis.py`
- `validate_step12_results.py`
- `freeze_step12_analysis.py`
- `verify_step12_freeze.py`

## Design guarantees

1. Steps 8, 9, 10, and 11 must all verify as frozen before Step 12 runs.
2. Step 12 writes only under `outputs/pilot_v0/analysis/step12/` plus Step 12 manifests.
3. No model inference, attack generation, mitigation generation, detector retraining, threshold tuning, or adaptive-round extension occurs.
4. The analysis specification is frozen before the robustness result package is produced.
5. Parent remains the inferential unit for paired tests and bootstrap intervals.
6. Multiple-testing correction is supplementary; effect sizes, confidence intervals, influence checks, and alternative outcomes remain central.
7. Final Step 12 freeze hashes both result artifacts and implementation files.
