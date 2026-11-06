# StegoGuard-LM Step 11 Test Report

The Step 11 implementation was built and tested against the uploaded August 2026 project snapshot in an isolated working copy.

## Uploaded snapshot note

The uploaded `stegoguard_lm.zip` contained the complete Step 10 adaptive result tree and protocol freeze, but it did not contain the final Step 10 freeze manifest/hash files. The existing final Step 10 freeze scripts were therefore exercised in the isolated test copy before Step 11.

## Step 10 prerequisite test

```text
STEP 10 FREEZE PASSED
Protocol: step10_adaptive_v1
Status: frozen
Frozen artifacts: 40

STEP 10 FREEZE VERIFICATION PASSED
Protocol: step10_adaptive_v1
Status: frozen
Frozen artifacts: 40
Frozen implementation files: 29
```

## Compilation and full test suite

All source/scripts/tests compiled successfully.

```text
Ran 88 tests
OK
```

The prior Step 10 package contained 81 tests. Step 11 adds 7 tests.

All eight new Step 11 script `--help` entry points loaded successfully.

## Step 11 input verification

```text
STEP 11 INPUT VERIFICATION PASSED
Step 10 status: frozen
Step 10 protocol: step10_adaptive_v1
Parents: 48
Trajectories: 288
Stego rows: 1152
Clean rows: 384
```

## Analysis-spec freeze and smoke test

```text
STEP 11 ANALYSIS-SPEC FREEZE PASSED
Protocol: step11_system_analysis_v1
Status: frozen_analysis_spec
Frozen spec/code artifacts: 16

STEP 11 ANALYSIS-SPEC VERIFICATION PASSED
Protocol: step11_system_analysis_v1
Status: frozen_analysis_spec
Frozen spec/code artifacts: 16

STEP 11 MODEL-FREE SMOKE TEST PASSED
Parents: 48
Round-condition rows: 24
Trajectory rows: 288
```

## Full Step 11 analysis and validation

```text
STEP 11 SYSTEM-LEVEL ANALYSIS COMPLETE
Parents: 48
Trajectories: 288
Primary pairwise comparison rows: 27
R3 stratified rows: 48
Figures: 5
Wrote: outputs/pilot_v0/analysis/step11/step11_system_analysis.json

STEP 11 RESULT VALIDATION PASSED
{'protocol_version': 'step11_system_analysis_v1', 'status': 'validated', 'parents': 48, 'trajectories': 288, 'stego_rows': 1152, 'clean_rows': 384, 'pairwise_comparison_rows': 27, 'figure_count': 5, 'failures': []}
```

The five figures were rendered and visually inspected.

## Final-freeze script test

The Step 11 final freeze and verification scripts were also tested in the isolated copy:

```text
STEP 11 FREEZE PASSED
Protocol: step11_system_analysis_v1
Status: frozen
Frozen artifacts: 26
Frozen implementation files: 13

STEP 11 FREEZE VERIFICATION PASSED
Protocol: step11_system_analysis_v1
Status: frozen
Frozen artifacts: 26
Frozen implementation files: 13
```

The distributed patch/ready package intentionally does not include generated Step 11 scientific result/freeze outputs. The local run should create those from the user's own frozen Step 10 state.

## Scientific sanity checks from the isolated test run

The model-free Step 11 analysis reproduces the known Step 10 aggregate trajectories and adds paired uncertainty.

At R3, migration versus frozen/no-feedback shows a positive parent-paired throughput difference under both defender policies:

```text
Deterministic: +0.1875 successful bits/parent
95% parent-bootstrap CI: [0.0625, 0.3542]

Randomized: +0.2083 successful bits/parent
95% parent-bootstrap CI: [0.0833, 0.3750]
```

R3 exact-survival differences versus frozen control are also positive with intervals above zero for both migration policies.

Fixed-channel versus frozen control has small point differences whose intervals touch zero, supporting the interpretation that within-family adaptation provides weak/unstable recovery.

Migration versus fixed-channel adaptation has positive R3 point estimates under both policies, but the throughput intervals are not uniformly separated from zero. Step 11 therefore classifies H10.5 as `directionally_supported_mixed_ci` rather than overstating statistical certainty.

Randomized minus deterministic migration at R3 is small and does not show a defensive advantage, preserving the negative H10.4 result.

Rank exact-message survival remains zero, while mean R3 residual capacity remains non-zero (roughly 5.6-12.2 bits depending on condition), preserving the crucial exact-message-versus-partial-information distinction.
