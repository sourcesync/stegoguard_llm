# StegoGuard-LM Step 12 — Local Running Instructions

Protocol: `step12_robustness_v1`

## What Step 12 is

Step 12 is the **robustness, sensitivity, and boundary-condition audit** of the frozen Step 11 findings. It is model-free and non-generative. It does not load Qwen, Mistral, or Gemma; it does not create new attacks; it does not add adaptive rounds; it does not retrain the detector; and it does not rerun mitigation.

The analysis checks whether the Step 11 findings survive alternative inferential procedures, deletion of individual parents/cells, alternative leakage metrics, the full adaptive R1–R3 window, starting-channel histories, and model/domain/length strata. It also repackages the already-frozen Step 8 detector ablation results for paper use.

## Patch installation

Your authoritative project folder should be:

```text
<repo-root>
```

Before updating it, make a backup copy of the entire folder.

Extract the Step 12 patch ZIP directly into the repository root. The patch contains only new Step 12 files. It should **not ask to overwrite any frozen Step 1–11 scientific files**.

After extraction, these new items should exist:

```text
stegoguard_lm\
├── config_step12.yaml
├── STEP12_PROTOCOL.md
├── STEP12_IMPLEMENTATION.md
├── STEP12_Local_Running_Instructions.md
├── STEP12_TEST_REPORT.md
├── src\stegoguard_lm\robustness\
│   ├── __init__.py
│   ├── io.py
│   ├── statistics.py
│   ├── analysis.py
│   └── reporting.py
├── scripts\
│   ├── verify_step12_inputs.py
│   ├── freeze_step12_analysis_spec.py
│   ├── verify_step12_analysis_spec.py
│   ├── smoke_test_step12.py
│   ├── run_step12_robustness_analysis.py
│   ├── validate_step12_results.py
│   ├── freeze_step12_analysis.py
│   └── verify_step12_freeze.py
└── tests\
    ├── test_step12_statistics.py
    ├── test_step12_analysis.py
    └── test_step12_provenance.py
```

## 1. Open PowerShell and enter the repository

```powershell
cd <repo-root>
$env:PYTHONPATH="src"
$env:PYTHONDONTWRITEBYTECODE="1"
```

Keep this PowerShell window open for the entire Step 12 run.

## 2. Verify Step 11 is still frozen

Run:

```powershell
python scripts/verify_step11_freeze.py --config config_step11.yaml
```

Expected:

```text
STEP 11 FREEZE VERIFICATION PASSED
Protocol: step11_system_analysis_v1
Status: frozen
Frozen artifacts: 26
Frozen implementation files: 13
```

If this fails, **stop**. Do not run Step 12 until the authoritative Step 11 freeze is intact.

## 3. Compile all source, scripts, and tests

```powershell
python -m compileall src scripts tests
```

Compilation should finish without syntax errors.

## 4. Run the complete regression test suite

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

Expected final lines:

```text
Ran 97 tests
OK
```

The 88 prior tests remain intact and Step 12 adds 9 new tests covering sign-flip testing, FDR adjustment, paired-parent handling, deletion stability, contrast construction, role splitting, and provenance hashing.

If any test fails, stop and send the full failure output before proceeding.

## 5. Verify all frozen Step 12 inputs

Run:

```powershell
python scripts/verify_step12_inputs.py --config config_step12.yaml
```

Expected:

```text
STEP 12 INPUT VERIFICATION PASSED
STEP8 protocol: step8_static_v1 (frozen)
STEP9 protocol: step9_mitigation_v1 (frozen)
STEP10 protocol: step10_adaptive_v1 (frozen)
STEP11 protocol: step11_system_analysis_v1 (frozen)
Parents: 48
Trajectories: 288
Stego rows: 1152
Clean rows: 384
```

This check is important because Step 12 reads frozen evidence from Steps 8–11. It refuses to proceed if any frozen hash no longer matches.

## 6. Freeze the Step 12 robustness-analysis specification

Run:

```powershell
python scripts/freeze_step12_analysis_spec.py --config config_step12.yaml
```

Expected:

```text
STEP 12 ANALYSIS-SPEC FREEZE PASSED
Protocol: step12_robustness_v1
Status: frozen_analysis_spec
Frozen spec/code artifacts: 16
```

This creates:

```text
outputs\pilot_v0\manifests\step12_analysis_spec_manifest.json
outputs\pilot_v0\manifests\step12_analysis_spec_hashes.txt
```

Do not use `--overwrite` after this succeeds.

Why this happens before the analysis: Step 11 outcomes are already known, so this is not a preregistration. It is a post-outcome robustness-spec freeze that prevents us from changing the Step 12 tests after seeing their results.

## 7. Verify the Step 12 analysis-spec freeze

```powershell
python scripts/verify_step12_analysis_spec.py --config config_step12.yaml
```

Expected:

```text
STEP 12 ANALYSIS-SPEC VERIFICATION PASSED
Protocol: step12_robustness_v1
Status: frozen_analysis_spec
Frozen spec/code artifacts: 16
```

If this fails, stop. Do not edit a hashed file and continue.

## 8. Run the model-free smoke test

```powershell
python scripts/smoke_test_step12.py --config config_step12.yaml
```

Expected:

```text
STEP 12 MODEL-FREE SMOKE TEST PASSED
Randomization rows: 54
R3 contrasts: 9
Leave-one-parent rows: 432
Leave-one-cell rows: 72
```

This smoke test uses a small randomization budget to exercise the real analysis logic quickly. It does not write scientific Step 12 results.

## 9. Run the full Step 12 robustness analysis

```powershell
python scripts/run_step12_robustness_analysis.py --config config_step12.yaml
```

On the tested project snapshot this completed in only a few seconds and used no LLM inference. Runtime on Windows may differ, but it should still be light compared with Steps 8–10.

Expected general output:

```text
STEP 12 ROBUSTNESS ANALYSIS COMPLETE
Parents: 48
Trajectories: 288
Randomization test rows: 54
R3 endpoint contrasts: 9
Leave-one-parent rows: 432
Leave-one-cell rows: 72
Figures: 5
Wrote: outputs/pilot_v0/analysis/step12/step12_robustness_summary.json
```

The full analysis uses 20,000 paired sign-flip randomization draws for each of the 54 primary outcome tests and 2,000 parent-bootstrap replicates for sensitivity contrasts.

## 10. Validate the complete Step 12 result package

Run:

```powershell
python scripts/validate_step12_results.py --config config_step12.yaml
```

Expected:

```text
STEP 12 RESULT VALIDATION PASSED
{'protocol_version': 'step12_robustness_v1',
 'status': 'validated',
 'parents': 48,
 'trajectories': 288,
 'randomization_test_rows': 54,
 'r3_contrast_count': 9,
 'figure_count': 5,
 'failures': []}
```

### STOP HERE

Do **not** run the Step 12 final freeze yet. Send the terminal output from Steps 9 and 10 to ChatGPT for scientific inspection first, exactly as we did for Steps 10 and 11.

## Step 12 output tree after Step 10 validation

```text
outputs\pilot_v0\analysis\step12\
├── step12_robustness_summary.json
├── step12_randomization_tests.csv
├── step12_r3_influence_summary.csv
├── step12_r3_leave_one_parent.csv
├── step12_r3_leave_one_cell.csv
├── step12_r3_alternative_leakage.csv
├── step12_adaptive_window_contrasts.csv
├── step12_history_conditioned_r3.csv
├── step12_heterogeneity_r3.csv
├── step12_clean_cell_metrics.csv
├── step12_static_ablation_reference.csv
├── step12_cross_stage_reference.csv
├── step12_claim_stability.json
├── step12_robustness_interpretation.md
├── step12_qa_review.txt
├── step12_validation_summary.json
└── figures\
    ├── step12_r3_effect_estimates.png
    ├── step12_leave_one_cell_stability.png
    ├── step12_alternative_leakage_metrics.png
    ├── step12_clean_burden_cells.png
    └── step12_static_ablation_reference.png
```

## What each Step 12 analysis means

### Paired randomization tests

These ask whether the observed parent-paired differences are unusually large relative to a null model in which the direction of each parent's paired difference is exchangeable. BH-FDR q-values are reported separately for successful covert bits and exact payload recovery across the 27 pre-specified Step 11 comparisons. This is a **supplementary sensitivity analysis**; do not replace Step 11 effect sizes and bootstrap confidence intervals with p-values.

### Leave-one-parent-out

Each of the nine R3 endpoint contrasts is recalculated 48 times, dropping one parent at a time. If an effect changes sign after deleting a single parent, the finding is influence-sensitive. If the sign remains stable across all deletions, the endpoint direction is not driven by one parent.

### Leave-one-cell-out

Each R3 contrast is recalculated eight times while dropping one source-model × domain × length cell. This tests whether one balanced experimental cell dominates the aggregate result.

### Alternative leakage metrics

Exact-message survival is not the only security outcome. Step 12 also tests effective bit accuracy and residual capacity. This is especially important for rank, where exact recovery can be zero even when partial information remains.

### Adaptive-window analysis

Instead of using only R3, this analysis aggregates the entire adaptive period R1–R3. It tests whether an endpoint finding is consistent with the broader post-feedback trajectory.

### History-conditioned analysis

The R3 contrasts are rerun separately for trajectories that began with rank and those that began with synonym. This identifies dependence on starting attack family without claiming universal path dependence.

### Heterogeneity analysis

R3 effects are recalculated separately by source model, domain, and nominal length. These are sensitivity/boundary-condition checks, not confirmatory subgroup tests because subgroup sample sizes are small.

### Clean-burden cell audit

For every defender × round × balanced cell, Step 12 reports clean intervention rate and visible-response utility measures. This checks whether aggregate clean burden conceals a particularly poor model/domain/length cell.

### Frozen static-ablation reference

The eight already-frozen Step 8 ablation metrics are copied into a Step 12 paper-ready reference table and figure. No detector is retrained and no static result is altered.

## Final freeze commands — only after scientific inspection

After the Step 12 results have been reviewed and accepted, the final commands will be:

```powershell
python scripts/freeze_step12_analysis.py --config config_step12.yaml
```

Expected general form:

```text
STEP 12 FREEZE PASSED
Protocol: step12_robustness_v1
Status: frozen
Frozen artifacts: 26
Frozen implementation files: 13
```

Then:

```powershell
python scripts/verify_step12_freeze.py --config config_step12.yaml
```

Expected:

```text
STEP 12 FREEZE VERIFICATION PASSED
Protocol: step12_robustness_v1
Status: frozen
Frozen artifacts: 26
Frozen implementation files: 13
```

Do not run these final commands until after the scientific review checkpoint.

## Important non-negotiable rules

- Never edit Steps 1–11 frozen files to make Step 12 results look cleaner.
- Never change the 20,000 randomization draws, 2,000 bootstrap replicates, contrast universe, outcomes, or FDR family after seeing Step 12 results.
- Never use `--overwrite` on a successful Step 12 analysis-spec freeze or final freeze without an explicitly documented protocol amendment.
- Preserve weak, null, contradictory, or sensitivity-dependent results.
- Treat multiple-testing results as supplementary to effect sizes and uncertainty.
- Do not call exact-message suppression “zero leakage” if residual information remains.
- Do not claim equilibrium, emergence, universal robustness, or long-run convergence from four rounds.
