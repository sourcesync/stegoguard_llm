# StegoGuard-LM Step 11 — Local Running Instructions

Protocol: `step11_system_analysis_v1`

## What Step 11 does

Step 11 is the **system-level analysis phase** for the already-completed adaptive experiment. Step 10 generated and validated the repeated attacker-defender trajectories. Step 11 does not generate new attacks or new LLM outputs. Instead, it turns the frozen R0-R3 evidence into the statistical and system-level results needed for the paper.

Step 11 formalizes:

1. feedback-enabled conditions versus the frozen/no-feedback control;
2. fixed-channel adaptation versus channel migration;
3. deterministic versus randomized defense under adaptation;
4. cumulative and final covert throughput;
5. exact-message survival versus residual rank information;
6. channel-transition/churn behavior;
7. history-conditioned outcomes by initial channel;
8. descriptive model/domain/length robustness slices;
9. legitimate-user intervention and utility burden;
10. H10.1-H10.5 assessment and interpretation guardrails.

It is **model-free**. No Hugging Face model downloads, generation, evaluator inference, or GPU work are required.

---

# Important prerequisite: verify the final Step 10 freeze

The ZIP uploaded to this chat contains all completed Step 10 result files but, despite the local Step 10 being described as frozen, the uploaded ZIP itself does **not** contain:

```text
outputs\pilot_v0\manifests\step10_adaptive_freeze_manifest.json
outputs\pilot_v0\manifests\step10_final_hashes.txt
```

I tested the existing Step 10 freeze scripts against an isolated copy of that uploaded snapshot. They passed with:

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

If your **local** project already contains the Step 10 final freeze manifest, do **not** run the freeze again. Run only the verification command below. Never use `--overwrite`.

---

# A. Update your local project

## Recommended option: patch ZIP

Use the patch if your local folder is the authoritative frozen Step 10 repository.

1. Make a backup copy of your current folder first.
2. Open:

```text
<repo-root>
```

3. Extract `StegoGuard_LM_Step11_System_Analysis_Patch_August_2026.zip` **into the repository root**.
4. Allow it to create the new Step 11 files/directories. The patch contains new Step 11 files only; it does not replace Step 1-10 code or results.

After extraction, the new structure should include:

```text
stegoguard_lm/
├── config_step11.yaml
├── STEP11_PROTOCOL.md
├── STEP11_IMPLEMENTATION.md
├── STEP11_Local_Running_Instructions.md
├── STEP11_TEST_REPORT.md
├── src/
│   └── stegoguard_lm/
│       └── analysis/
│           ├── __init__.py
│           ├── io.py
│           ├── statistics.py
│           ├── system.py
│           └── reporting.py
├── scripts/
│   ├── verify_step11_inputs.py
│   ├── freeze_step11_analysis_spec.py
│   ├── verify_step11_analysis_spec.py
│   ├── smoke_test_step11.py
│   ├── run_step11_system_analysis.py
│   ├── validate_step11_results.py
│   ├── freeze_step11_analysis.py
│   └── verify_step11_freeze.py
└── tests/
    ├── test_step11_statistics.py
    ├── test_step11_system_metrics.py
    └── test_step11_provenance.py
```

## Alternative option: complete ready project ZIP

`StegoGuard_LM_Step11_System_Analysis_Ready_August_2026.zip` is a complete project copy based on the uploaded snapshot. It includes the Step 10 final freeze manifest/hash generated and verified in the isolated test copy plus the Step 11 implementation. Use this as a clean reference or as a replacement working copy only if desired.

For a local repository that you already froze yourself, the **patch is safer and preferred**.

---

# B. Open PowerShell and set the environment

```powershell
cd <repo-root>
$env:PYTHONPATH="src"
$env:PYTHONDONTWRITEBYTECODE="1"
```

You can confirm Python is running:

```powershell
python --version
```

Step 11 uses the existing environment. No model inference is required.

---

# C. Confirm the Step 10 final freeze

Check whether this file exists:

```text
outputs\pilot_v0\manifests\step10_adaptive_freeze_manifest.json
```

### If it already exists

Run only:

```powershell
python scripts/verify_step10_freeze.py --config config_step10.yaml
```

Expected:

```text
STEP 10 FREEZE VERIFICATION PASSED
Protocol: step10_adaptive_v1
Status: frozen
Frozen artifacts: 40
Frozen implementation files: 29
```

### If it does not exist

Stop and first confirm that the folder really is your completed R0-R3/summary state. If it is the same completed state, run:

```powershell
python scripts/freeze_adaptive_experiment.py --config config_step10.yaml
```

Expected:

```text
STEP 10 FREEZE PASSED
Protocol: step10_adaptive_v1
Status: frozen
Frozen artifacts: 40
```

Then immediately verify:

```powershell
python scripts/verify_step10_freeze.py --config config_step10.yaml
```

Do not use `--overwrite`. If either command fails, stop and send the error before continuing.

---

# D. Compile all source, scripts, and tests

```powershell
python -m compileall src scripts tests
```

This should finish without syntax errors.

---

# E. Run the complete regression/unit test suite

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

Expected final lines:

```text
Ran 88 tests
OK
```

The previous Step 10 package had 81 tests. Step 11 adds 7 tests covering parent-level bootstrap behavior, paired comparisons, system metrics, and provenance hashing.

If any test fails, stop before continuing.

---

# F. Verify the Step 11 inputs

Run:

```powershell
python scripts/verify_step11_inputs.py --config config_step11.yaml
```

Expected:

```text
STEP 11 INPUT VERIFICATION PASSED
Step 10 status: frozen
Step 10 protocol: step10_adaptive_v1
Parents: 48
Trajectories: 288
Stego rows: 1152
Clean rows: 384
```

This step re-hashes the frozen Step 10 artifacts. Step 11 therefore refuses to analyze a silently changed Step 10 state.

---

# G. Freeze the Step 11 analysis specification

This step matters because the Step 10 outcomes are already known. Therefore this is **not a preregistration**. It is a post-outcome analysis-specification freeze that prevents us from changing selected contrasts, outcomes, bootstrap rules, or interpretation guardrails after seeing the Step 11 inferential results.

Run once:

```powershell
python scripts/freeze_step11_analysis_spec.py --config config_step11.yaml
```

Expected:

```text
STEP 11 ANALYSIS-SPEC FREEZE PASSED
Protocol: step11_system_analysis_v1
Status: frozen_analysis_spec
Frozen spec/code artifacts: 16
```

Then verify it:

```powershell
python scripts/verify_step11_analysis_spec.py --config config_step11.yaml
```

Expected:

```text
STEP 11 ANALYSIS-SPEC VERIFICATION PASSED
Protocol: step11_system_analysis_v1
Status: frozen_analysis_spec
Frozen spec/code artifacts: 16
```

Do not use `--overwrite` after this point.

---

# H. Run the model-free Step 11 smoke test

```powershell
python scripts/smoke_test_step11.py --config config_step11.yaml
```

Expected:

```text
STEP 11 MODEL-FREE SMOKE TEST PASSED
Parents: 48
Round-condition rows: 24
Trajectory rows: 288
```

This checks trajectory assembly and the 3 attacker × 2 defender × 4 round structure without writing the full scientific result package.

---

# I. Run the full Step 11 system analysis

```powershell
python scripts/run_step11_system_analysis.py --config config_step11.yaml
```

Expected:

```text
STEP 11 SYSTEM-LEVEL ANALYSIS COMPLETE
Parents: 48
Trajectories: 288
Primary pairwise comparison rows: 27
R3 stratified rows: 48
Figures: 5
Wrote: outputs/pilot_v0/analysis/step11/step11_system_analysis.json
```

This is model-free and should run much faster than Step 10.

The command writes:

```text
outputs/pilot_v0/analysis/step11/
├── step11_system_analysis.json
├── step11_round_metrics.csv
├── step11_pairwise_comparisons.csv
├── step11_trajectory_metrics.csv
├── step11_transition_metrics.csv
├── step11_attack_family_metrics.csv
├── step11_rank_residual_capacity.csv
├── step11_recovery_metrics.csv
├── step11_resilience_profiles.csv
├── step11_history_conditioned_metrics.csv
├── step11_stratified_metrics.csv
├── step11_clean_metrics.csv
├── step11_hypothesis_assessment.json
├── step11_system_interpretation.md
├── step11_qa_review.txt
└── figures/
    ├── step11_throughput_by_condition.png
    ├── step11_exact_survival_by_condition.png
    ├── step11_migration_synonym_share.png
    ├── step11_rank_residual_capacity_r3.png
    └── step11_clean_semantic_utility.png
```

### What the 27 pairwise rows are

For R1-R3, separately for deterministic and randomized defense:

```text
fixed-channel - frozen/no-feedback
migration - frozen/no-feedback
migration - fixed-channel
```

That produces 18 rows.

For R1-R3, separately for each of the three attacker conditions:

```text
randomized rewrite - deterministic rewrite
```

That adds 9 rows.

Total = 27 parent-paired comparison rows.

Each row contains a paired difference and 95% parent-bootstrap CI for successful covert bits and exact payload survival.

---

# J. Validate the Step 11 result package

Run:

```powershell
python scripts/validate_step11_results.py --config config_step11.yaml
```

Expected:

```text
STEP 11 RESULT VALIDATION PASSED
{'protocol_version': 'step11_system_analysis_v1', 'status': 'validated', 'parents': 48, 'trajectories': 288, 'stego_rows': 1152, 'clean_rows': 384, 'pairwise_comparison_rows': 27, 'figure_count': 5, 'failures': []}
```

## Stop here for scientific review

After Steps I and J pass, send me both terminal outputs. Do **not** freeze Step 11 yet. We should inspect the inferential results together first.

The most important review files are:

```text
outputs/pilot_v0/analysis/step11/step11_system_interpretation.md
outputs/pilot_v0/analysis/step11/step11_pairwise_comparisons.csv
outputs/pilot_v0/analysis/step11/step11_resilience_profiles.csv
outputs/pilot_v0/analysis/step11/step11_rank_residual_capacity.csv
outputs/pilot_v0/analysis/step11/step11_transition_metrics.csv
outputs/pilot_v0/analysis/step11/step11_hypothesis_assessment.json
outputs/pilot_v0/analysis/step11/step11_qa_review.txt
```

---

# K. Final Step 11 freeze — only after scientific review

After we review and accept the Step 11 analysis without changing the frozen specification:

```powershell
python scripts/freeze_step11_analysis.py --config config_step11.yaml
```

Expected:

```text
STEP 11 FREEZE PASSED
Protocol: step11_system_analysis_v1
Status: frozen
Frozen artifacts: 26
Frozen implementation files: 13
```

Then verify:

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

Never use `--overwrite` after final freeze.

Final provenance files:

```text
outputs/pilot_v0/manifests/step11_system_analysis_freeze_manifest.json
outputs/pilot_v0/manifests/step11_final_hashes.txt
```

---

# What Step 11 means scientifically

Step 10 established that bounded feedback can change attacker behavior and that cross-family migration is the dominant observed recovery mechanism. Step 11 is designed to make those claims statistically defensible rather than relying on aggregate totals alone.

The main inferential questions are:

- Does migration outperform the frozen/no-feedback counterfactual at the parent level?
- Is fixed-channel search reliably better than frozen control?
- Does migration reliably outperform fixed-channel adaptation, or are the aggregate differences larger than the paired uncertainty permits us to claim?
- Does randomized rewriting reduce adaptive recovery?
- Does rank retain partial information even when exact rank delivery is zero?
- Can high trajectory-level switching coexist with stable aggregate channel shares?
- How does the initial channel condition later outcomes?
- Does the legitimate-user intervention/utility burden remain stable while the adversarial system adapts?

Step 11 intentionally reports a **resilience profile** rather than inventing a single composite resilience score.
