# StegoGuard-LM Step 12 Implementation and Test Report

Protocol: `step12_robustness_v1`

Tested against the updated project snapshot supplied on August 30, 2026.

## Frozen input verification

The uploaded project was independently checked before Step 12 implementation:

```text
STEP 11 FREEZE VERIFICATION PASSED
Protocol: step11_system_analysis_v1
Status: frozen
Frozen artifacts: 26
Frozen implementation files: 13
```

Step 12 input verification passed for all frozen stages it reads directly:

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

## Regression suite

After adding Step 12:

```text
Ran 97 tests
OK
```

All prior tests remained passing. Step 12 adds nine new tests for randomization determinism, zero-difference behavior, BH-FDR adjustment, paired-parent alignment, deletion stability summaries, endpoint contrast construction, stable contrast IDs, record-role separation, and provenance hash verification.

## Analysis-spec freeze test

```text
STEP 12 ANALYSIS-SPEC FREEZE PASSED
Protocol: step12_robustness_v1
Status: frozen_analysis_spec
Frozen spec/code artifacts: 16
```

Independent verification also passed.

## Smoke test

```text
STEP 12 MODEL-FREE SMOKE TEST PASSED
Randomization rows: 54
R3 contrasts: 9
Leave-one-parent rows: 432
Leave-one-cell rows: 72
```

## Full analysis test

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

Observed Linux test-runtime for the full analysis was approximately 6 seconds with peak memory around 170 MB. No LLM inference was used.

## Result validation

```text
STEP 12 RESULT VALIDATION PASSED
{'protocol_version': 'step12_robustness_v1', 'status': 'validated',
 'parents': 48, 'trajectories': 288,
 'randomization_test_rows': 54, 'r3_contrast_count': 9,
 'figure_count': 5, 'failures': []}
```

## Final freeze machinery test

The final freeze commands were also tested in the isolated development copy:

```text
STEP 12 FREEZE PASSED
Protocol: step12_robustness_v1
Status: frozen
Frozen artifacts: 26
Frozen implementation files: 13
```

and:

```text
STEP 12 FREEZE VERIFICATION PASSED
Protocol: step12_robustness_v1
Status: frozen
Frozen artifacts: 26
Frozen implementation files: 13
```

The user should **not** run the final freeze until the locally generated Step 12 result package has been scientifically inspected.

## Development-run scientific sanity check

The isolated test run produced coherent sensitivity behavior rather than simply reproducing Step 11 mechanically. In particular:

- migration-versus-frozen retained positive R3 effect direction under all leave-one-parent and leave-one-cell deletions for both defender policies;
- paired sign-flip p-values for migration-versus-frozen were small before multiplicity correction, but BH-FDR correction across all 27 comparisons per outcome was conservative and did not retain those endpoint tests below q=0.05;
- migration-versus-fixed remained directionally positive but more uncertain;
- fixed-channel adaptation remained weak relative to frozen/no-feedback;
- randomized rewriting did not show a sustained advantage;
- alternative leakage metrics revealed that higher exact/bit recovery under migration can coexist with lower absolute residual-capacity bits because migration shifts activity toward the lower-capacity synonym family.

These are reasons to preserve effect sizes, bootstrap intervals, influence diagnostics, and multiple outcome definitions together rather than reducing Step 12 to a single significance flag.
