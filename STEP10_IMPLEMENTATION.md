# StegoGuard-LM Step 10 Implementation and Local Running Instructions

## 1. Purpose and frozen status

Step 10 is the adaptive attacker-defender experiment under protocol `step10_adaptive_v1`. It begins the repeated-feedback evaluation that was intentionally excluded from frozen Steps 1 through 9.

The supplied Step 10 package has already completed the model-independent protocol construction and freeze. Do not rebuild or alter the Step 10 protocol after adaptive candidate generation begins.

Frozen Step 10 design:

- Population: 48 balanced common parents, six in each source-model x domain x nominal-length cell.
- Rounds: R0, R1, R2, R3.
- Attacker conditions:
  - `frozen_no_feedback`
  - `fixed_channel_feedback`
  - `channel_migration_feedback`
- Defender policies:
  - `deterministic_rewrite`
  - `randomized_rewrite`
- Primary adaptive attack families:
  - `rank_encoding`
  - `synonym_slot_codebook`
- Unicode remains a structural-control result from the frozen prior stages rather than a primary migration competitor.
- Attacker feedback contains only:
  - detected yes/no
  - intervention route
  - exact payload recovery yes/no
- Maximum candidate budget per family: 10.
- Primary operating point: the frozen Step 8 F1-oriented linguistic detector, reconstructed as an outer-fold ensemble.
- Randomized rewrite seeds are pre-frozen and hidden from the attacker policy.
- Clean controls: the same 48 clean parents, under both defender policies, in every round.
- Primary system outcome: successful covert throughput in exact delivered payload bits, with effective bit accuracy and residual capacity retained separately.

The package contains 638 pre-frozen candidate specifications and 288 adaptive trajectories. The 288 trajectories are:

`48 parents x 3 attacker conditions x 2 defender policies = 288 trajectories`

Every completed round must contain:

- 288 stego trajectory tasks
- 96 clean-control tasks
- 384 total defense/utility/feedback rows
- 288 stego payload-evaluation rows

Across four rounds, the planned experiment therefore contains 1,152 stego round observations and 384 clean-control round observations.

## 2. Files added for Step 10

### Root files

- `config_step10.yaml`
- `STEP10_PROTOCOL.md`
- `STEP10_IMPLEMENTATION.md`

### Adaptive package

`src/stegoguard_lm/adaptive/`

- `__init__.py`
- `io.py`
- `strategy.py`
- `manifests.py`
- `candidates.py`
- `detector.py`
- `rounds.py`

### Step 10 scripts

- `scripts/build_step10_manifest.py`
- `scripts/validate_step10_manifest.py`
- `scripts/freeze_step10_protocol.py`
- `scripts/verify_step10_protocol_freeze.py`
- `scripts/generate_adaptive_candidates.py`
- `scripts/validate_adaptive_candidates.py`
- `scripts/build_adaptive_detection_dataset.py`
- `scripts/extract_adaptive_evaluator_features.py`
- `scripts/extract_adaptive_semantic_features.py`
- `scripts/extract_adaptive_unicode_features.py`
- `scripts/validate_adaptive_detection_features.py`
- `scripts/score_adaptive_candidates.py`
- `scripts/build_adaptive_round.py`
- `scripts/generate_adaptive_defense_outputs.py`
- `scripts/evaluate_adaptive_payloads.py`
- `scripts/extract_adaptive_utility.py`
- `scripts/finalize_adaptive_feedback.py`
- `scripts/validate_adaptive_round.py`
- `scripts/summarize_adaptive_results.py`
- `scripts/freeze_adaptive_experiment.py`
- `scripts/verify_step10_freeze.py`
- `scripts/smoke_test_adaptive.py`

### Tests added

- `tests/test_adaptive_strategy.py`
- `tests/test_adaptive_candidates.py`
- `tests/test_adaptive_protocol.py`

### Pre-frozen Step 10 protocol artifacts

- `outputs/pilot_v0/adaptive/step10/candidate_manifest.jsonl`
- `outputs/pilot_v0/adaptive/step10/trajectory_manifest.jsonl`
- `outputs/pilot_v0/manifests/step10_adaptive_protocol_manifest.json`
- `outputs/pilot_v0/manifests/step10_protocol_hashes.txt`

## 3. Important rules before you run anything

1. Use the supplied `config_step10.yaml` for Step 10. Do not edit frozen `config.yaml`.
2. Do not modify any Step 1 through Step 9 output, manifest, hash file, attack rule, detector feature definition, detector training procedure, mitigation rule, or Step 9 seed.
3. Do not change `STEP10_PROTOCOL.md`, `config_step10.yaml`, `candidate_manifest.jsonl`, or `trajectory_manifest.jsonl` after generation begins.
4. Do not increase the candidate budget because an attack fails.
5. Do not add rounds because R3 looks interesting.
6. Do not expose detector scores, thresholds, feature vectors, or rewrite seeds to the attacker policy.
7. Keep terminal candidate-generation failures as results. Do not regenerate until success.
8. Execute rounds sequentially. R1 must not be built before R0 feedback is finalized and validated. The same rule applies to R2 and R3.
9. If the laptop shuts down during a resumable command, rerun the exact same command with `--resume`.
10. Do not use `--overwrite` after a scientific stage has begun unless we explicitly diagnose a real implementation defect and version the protocol accordingly.

## 4. Step A: Extract the package and open PowerShell

Extract the project folder to your working location. For example:

```powershell
cd <repo-root>
```

Confirm that these files exist:

```powershell
Test-Path config_step10.yaml
Test-Path STEP10_PROTOCOL.md
Test-Path outputs/pilot_v0/manifests/step9_mitigation_freeze_manifest.json
Test-Path outputs/pilot_v0/manifests/step10_adaptive_protocol_manifest.json
```

All four should print:

```text
True
```

## 5. Step B: Activate the Python environment and set PYTHONPATH

Use the same Python environment that successfully completed Steps 1 through 9.

Then run:

```powershell
$env:PYTHONPATH="src"
$env:PYTHONDONTWRITEBYTECODE="1"
```

Confirm Python:

```powershell
python --version
```

If imports that previously worked now fail, restore the same dependencies used for Steps 8 and 9. Do not change model versions in `config_step10.yaml`.

## 6. Step C: Compile and run the full test suite

Run:

```powershell
python -m compileall src scripts tests
```

Then:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

Expected current total:

```text
Ran 81 tests
OK
```

The original 69 tests remain, and 12 Step 10 tests are added.

## 7. Step D: Verify frozen Steps 8, 9, and the Step 10 protocol

The Step 10 manifest builder already verified the frozen Step 8 and Step 9 hashes when the package was prepared. On your machine, verify the Step 10 protocol again:

```powershell
python scripts/validate_step10_manifest.py --config config_step10.yaml
```

Expected:

```text
Step 10 manifest validation passed.
Candidate specifications: 638
Trajectories: 288
Parents: 48
```

Then run:

```powershell
python scripts/verify_step10_protocol_freeze.py --config config_step10.yaml
```

Expected:

```text
STEP 10 PROTOCOL FREEZE VERIFICATION PASSED
Protocol: step10_adaptive_v1
Frozen protocol artifacts: 4
```

Then run the model-free adaptive smoke test:

```powershell
python scripts/smoke_test_adaptive.py --config config_step10.yaml
```

Expected:

```text
STEP 10 MODEL-FREE SMOKE TEST PASSED
Trajectories: 288
Policy transition tests: passed
Seed determinism test: passed
```

### Do not run these two commands now

The package already contains the frozen Step 10 manifests. These commands are documented for provenance only:

```powershell
python scripts/build_step10_manifest.py --config config_step10.yaml --overwrite
python scripts/freeze_step10_protocol.py --config config_step10.yaml --overwrite
```

Do not rerun them once adaptive candidate generation has started.

## 8. Step E: Generate the adaptive candidate catalog

There are 638 frozen candidate specifications. Strategy 0 in each parent/family is anchored to the frozen Step 7 accepted attack. Alternative strategies are bounded Step 10 realizations.

Because of your local hardware constraints, run one attack/model group at a time.

### E1. Synonym candidates for Qwen

```powershell
python scripts/generate_adaptive_candidates.py --config config_step10.yaml --channel synonym --source-model qwen --resume
```

### E2. Synonym candidates for Mistral

```powershell
python scripts/generate_adaptive_candidates.py --config config_step10.yaml --channel synonym --source-model mistral --resume
```

The synonym alternatives use tokenizer-only generation, so they should be much lighter than rank generation.

### E3. Rank candidates for Qwen

```powershell
python scripts/generate_adaptive_candidates.py --config config_step10.yaml --channel rank --source-model qwen --resume
```

### E4. Rank candidates for Mistral

```powershell
python scripts/generate_adaptive_candidates.py --config config_step10.yaml --channel rank --source-model mistral --resume
```

### If the laptop shuts down during E1 through E4

Rerun the exact interrupted command. Keep `--resume`.

The script immediately appends accepted candidates and preserves terminal candidate failures separately.

### E5. Validate candidate accounting

After all four generation commands finish:

```powershell
python scripts/validate_adaptive_candidates.py --config config_step10.yaml --require-complete
```

This must pass before detection-feature extraction. The final number of accepted candidates may be lower than 638 because alternative candidates are allowed to terminate under the frozen bounded rules. All 96 strategy-0 anchors must be accepted.

Do not change the candidate budget to compensate for terminal failures.

## 9. Step F: Build the Step 10 detection dataset

Run:

```powershell
python scripts/build_adaptive_detection_dataset.py --config config_step10.yaml
```

The dataset contains:

- 48 frozen balanced clean controls
- every accepted Step 10 adaptive candidate

If the file already exists from a genuine pre-generation setup attempt and no scientific scoring has occurred, stop and inspect before using `--overwrite`.

## 10. Step G: Extract detection features for the adaptive candidates

The clean-control features are reused from frozen Step 8. Only new candidate features require new model inference.

### G1. Evaluator A distributional features

```powershell
python scripts/extract_adaptive_evaluator_features.py --config config_step10.yaml --evaluator evaluator_a --resume
```

If interrupted, rerun the exact command with `--resume`.

### G2. Evaluator B distributional features

```powershell
python scripts/extract_adaptive_evaluator_features.py --config config_step10.yaml --evaluator evaluator_b --resume
```

If interrupted, rerun with `--resume`.

### G3. Semantic features

```powershell
python scripts/extract_adaptive_semantic_features.py --config config_step10.yaml --resume
```

If interrupted, rerun with `--resume`.

### G4. Unicode structural features

```powershell
python scripts/extract_adaptive_unicode_features.py --config config_step10.yaml
```

### G5. Validate all Step 10 detector features

```powershell
python scripts/validate_adaptive_detection_features.py --config config_step10.yaml
```

This must pass before candidate scoring.

## 11. Step H: Score every accepted candidate with the frozen Step 8 detector

Run:

```powershell
python scripts/score_adaptive_candidates.py --config config_step10.yaml
```

This reconstructs the frozen Step 8 10-fold topic-disjoint full-fusion detector ensemble. Each fold uses its training-only F1 threshold. The Step 10 deployed decision is a majority vote across the reconstructed folds. Continuous scores remain researcher-only and are not passed to the adaptive attacker.

The attacker receives only the three frozen feedback fields after each round.

## 12. Step I: Run Round 0

Round 0 is the adaptive experiment baseline. No prior feedback exists.

### I1. Build R0 tasks

```powershell
python scripts/build_adaptive_round.py --config config_step10.yaml --round 0
```

Expected task count:

```text
384
```

composed of 288 stego trajectories and 96 clean controls.

### I2. Run deterministic defender outputs

```powershell
python scripts/generate_adaptive_defense_outputs.py --config config_step10.yaml --round 0 --policy deterministic_rewrite --resume
```

### I3. Run randomized defender outputs

```powershell
python scripts/generate_adaptive_defense_outputs.py --config config_step10.yaml --round 0 --policy randomized_rewrite --resume
```

If either command is interrupted, rerun the exact command with `--resume`.

### I4. Evaluate synonym payloads

```powershell
python scripts/evaluate_adaptive_payloads.py --config config_step10.yaml --round 0 --channel synonym --resume
```

### I5. Evaluate rank payloads for Qwen

```powershell
python scripts/evaluate_adaptive_payloads.py --config config_step10.yaml --round 0 --channel rank --source-model qwen --resume
```

### I6. Evaluate rank payloads for Mistral

```powershell
python scripts/evaluate_adaptive_payloads.py --config config_step10.yaml --round 0 --channel rank --source-model mistral --resume
```

### I7. Extract visible-response utility

```powershell
python scripts/extract_adaptive_utility.py --config config_step10.yaml --round 0 --resume
```

### I8. Finalize bounded R0 attacker feedback

```powershell
python scripts/finalize_adaptive_feedback.py --config config_step10.yaml --round 0
```

### I9. Validate R0

```powershell
python scripts/validate_adaptive_round.py --config config_step10.yaml --round 0
```

Do not proceed to R1 unless R0 validation passes.

Expected validated R0 accounting:

- total tasks: 384
- stego trajectories: 288
- clean controls: 96
- payload rows: 288
- defense failures: 0

## 13. Step J: Run Round 1

R1 is the first round whose attack choice can react to R0 feedback.

```powershell
python scripts/build_adaptive_round.py --config config_step10.yaml --round 1

python scripts/generate_adaptive_defense_outputs.py --config config_step10.yaml --round 1 --policy deterministic_rewrite --resume
python scripts/generate_adaptive_defense_outputs.py --config config_step10.yaml --round 1 --policy randomized_rewrite --resume

python scripts/evaluate_adaptive_payloads.py --config config_step10.yaml --round 1 --channel synonym --resume
python scripts/evaluate_adaptive_payloads.py --config config_step10.yaml --round 1 --channel rank --source-model qwen --resume
python scripts/evaluate_adaptive_payloads.py --config config_step10.yaml --round 1 --channel rank --source-model mistral --resume

python scripts/extract_adaptive_utility.py --config config_step10.yaml --round 1 --resume

python scripts/finalize_adaptive_feedback.py --config config_step10.yaml --round 1
python scripts/validate_adaptive_round.py --config config_step10.yaml --round 1
```

Do not proceed unless validation passes.

## 14. Step K: Run Round 2

```powershell
python scripts/build_adaptive_round.py --config config_step10.yaml --round 2

python scripts/generate_adaptive_defense_outputs.py --config config_step10.yaml --round 2 --policy deterministic_rewrite --resume
python scripts/generate_adaptive_defense_outputs.py --config config_step10.yaml --round 2 --policy randomized_rewrite --resume

python scripts/evaluate_adaptive_payloads.py --config config_step10.yaml --round 2 --channel synonym --resume
python scripts/evaluate_adaptive_payloads.py --config config_step10.yaml --round 2 --channel rank --source-model qwen --resume
python scripts/evaluate_adaptive_payloads.py --config config_step10.yaml --round 2 --channel rank --source-model mistral --resume

python scripts/extract_adaptive_utility.py --config config_step10.yaml --round 2 --resume

python scripts/finalize_adaptive_feedback.py --config config_step10.yaml --round 2
python scripts/validate_adaptive_round.py --config config_step10.yaml --round 2
```

Do not proceed unless validation passes.

## 15. Step L: Run Round 3

```powershell
python scripts/build_adaptive_round.py --config config_step10.yaml --round 3

python scripts/generate_adaptive_defense_outputs.py --config config_step10.yaml --round 3 --policy deterministic_rewrite --resume
python scripts/generate_adaptive_defense_outputs.py --config config_step10.yaml --round 3 --policy randomized_rewrite --resume

python scripts/evaluate_adaptive_payloads.py --config config_step10.yaml --round 3 --channel synonym --resume
python scripts/evaluate_adaptive_payloads.py --config config_step10.yaml --round 3 --channel rank --source-model qwen --resume
python scripts/evaluate_adaptive_payloads.py --config config_step10.yaml --round 3 --channel rank --source-model mistral --resume

python scripts/extract_adaptive_utility.py --config config_step10.yaml --round 3 --resume

python scripts/finalize_adaptive_feedback.py --config config_step10.yaml --round 3
python scripts/validate_adaptive_round.py --config config_step10.yaml --round 3
```

After R3 validation passes, the repeated adaptive experiment is complete but not yet scientifically frozen.

## 16. Step M: Build the adaptive system-level summary

Run:

```powershell
python scripts/summarize_adaptive_results.py --config config_step10.yaml
```

Primary outputs:

```text
outputs/pilot_v0/adaptive/step10/adaptive_summary.json
outputs/pilot_v0/adaptive/step10/adaptive_qa_review.txt
```

The summary reports, by attacker condition, defender policy, and round:

- exact payload survival rate
- total successful covert bits
- successful covert bits per parent with parent-cluster bootstrap confidence intervals
- mean effective bit accuracy
- residual capacity bits
- intervention rate
- strategy-change rate
- channel-migration rate
- channel counts and shares
- trajectory throughput sum across R0-R3
- R3 minus R1 throughput recovery
- channel transition counts
- clean-control intervention and utility burden

Do not interpret a failed exact payload as zero information leakage. Inspect bit accuracy and residual capacity too.

## 17. Step N: Scientific inspection before final freeze

Before freezing, open and inspect:

```text
outputs/pilot_v0/adaptive/step10/adaptive_summary.json
outputs/pilot_v0/adaptive/step10/adaptive_qa_review.txt
outputs/pilot_v0/adaptive/step10/round_0/validation_summary.json
outputs/pilot_v0/adaptive/step10/round_1/validation_summary.json
outputs/pilot_v0/adaptive/step10/round_2/validation_summary.json
outputs/pilot_v0/adaptive/step10/round_3/validation_summary.json
```

Do not modify the protocol because a hypothesis did not behave as expected. Negative findings are valid findings.

In particular, do not claim:

- adaptation unless feedback-enabled trajectories differ meaningfully from frozen controls
- channel migration unless observed transition counts support it
- resilience from one-shot suppression alone
- emergence unless a genuinely unforced system-level pattern is observed
- zero leakage merely because exact recovery is zero

## 18. Step O: Freeze Step 10 only after scientific inspection

Run:

```powershell
python scripts/freeze_adaptive_experiment.py --config config_step10.yaml
```

Expected terminal form:

```text
STEP 10 FREEZE PASSED
Protocol: step10_adaptive_v1
Status: frozen
Frozen artifacts: <count>
```

Then verify:

```powershell
python scripts/verify_step10_freeze.py --config config_step10.yaml
```

Expected:

```text
STEP 10 FREEZE VERIFICATION PASSED
Protocol: step10_adaptive_v1
Status: frozen
```

Final freeze files:

```text
outputs/pilot_v0/manifests/step10_adaptive_freeze_manifest.json
outputs/pilot_v0/manifests/step10_final_hashes.txt
```

Once this verification passes, Step 10 evidence is immutable and ready for Step 11 trajectory interpretation and system-level resilience analysis.

## 19. Resume guide after shutdown

The following expensive commands are resumable. If interrupted, rerun the exact command with `--resume`:

- `generate_adaptive_candidates.py`
- `extract_adaptive_evaluator_features.py`
- `extract_adaptive_semantic_features.py`
- `generate_adaptive_defense_outputs.py`
- `evaluate_adaptive_payloads.py`
- `extract_adaptive_utility.py`

Do not delete partially written JSONL files simply because the run was interrupted.

The following commands are deterministic construction/validation commands and should normally be rerun only after checking why an output exists:

- `build_adaptive_detection_dataset.py`
- `extract_adaptive_unicode_features.py`
- `score_adaptive_candidates.py`
- `build_adaptive_round.py`
- `finalize_adaptive_feedback.py`
- `summarize_adaptive_results.py`

Do not use `--overwrite` casually. It is provided for diagnosed setup errors before freezing, not for trying alternative outcomes.

## 20. What was tested before delivery

The supplied package was checked against the latest frozen Steps 1 through 9 project snapshot.

Model-independent checks completed successfully:

- all 246 original project files matched their pre-Step-10 SHA-256 hashes after implementation
- all frozen Step 8 and Step 9 manifests verified during Step 10 manifest construction
- Step 10 candidate manifest built successfully
- candidate specifications: 638
- adaptive trajectories: 288
- balanced parents: 48
- Step 10 manifest validation passed
- Step 10 protocol freeze passed
- Step 10 protocol freeze verification passed
- Step 10 model-free adaptive smoke test passed
- frozen Step 8 detector reconstruction produced all 10 outer-fold detectors and successfully scored a held-out-format sample
- all 81 unit/integration tests passed
- all Step 10 Python files compiled successfully
- all Step 10 command-line parsers opened successfully with `--help`

The GPU/model-dependent full Step 10 experiment was intentionally not executed in the packaging environment because the local Hugging Face model stack and your downloaded model weights are not available there. The expensive model-dependent stages must run on the same local environment that completed Steps 5 through 9. This is why the package freezes the protocol and validates the complete orchestration before local scientific execution rather than claiming unperformed results.

## 21. What to send back after each major stage

To minimize debugging time, send the terminal output after these checkpoints:

1. Step D protocol verification and smoke test.
2. Step E5 complete candidate validation.
3. Step G5 feature validation plus Step H scoring completion.
4. R0 validation.
5. R1 validation.
6. R2 validation.
7. R3 validation.
8. Step M summary output.
9. Step O final freeze verification.

If any validation fails, stop at that point and preserve all current files. Do not delete outputs or change budgets before inspection.
