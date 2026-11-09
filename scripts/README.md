# Scripts

Run every script from the repository root after `pip install -e ".[models]"`.
Most take `--config` and default to the config for their step. Scripts are kept
in one flat folder because the freeze manifests record their paths.

The order below follows each step's implementation document. For Steps 1–7,
which have no running instructions, the order is inferred from the script
inputs and outputs.

## Steps 1–5: prompts and clean corpus (`config.yaml`)

1. `render_prompts.py`
2. `validate_rendered_prompts.py`
3. `generate_clean_samples.py` (uses standard prompts only; conflict prompts are filtered out)
4. `validate_clean_samples.py`

## Step 6: attack infrastructure (`config.yaml`)

1. `audit_synonym_codebook.py`, `audit_synonym_inplace_capacity.py`
2. `smoke_test_attacks.py`
3. `validate_attack_smoke.py`
4. `reset_attack_smoke_channel.py` (maintenance: clears one channel's smoke results)

## Step 7: full attack corpus (`config.yaml`)

1. `build_attack_task_manifest.py`
2. `validate_attack_task_manifest.py`
3. `generate_stego_samples.py`
4. `validate_stego_samples.py`
5. `summarize_attack_corpus.py`

## Step 8: static detection (`config.yaml`)

1. `build_detection_dataset.py`, `validate_detection_dataset.py`
2. `build_detection_splits.py`, `validate_detection_splits.py`
3. `extract_unicode_features.py`, `extract_evaluator_features.py`, `extract_semantic_features.py`
4. `validate_detection_features.py`
5. `run_static_detection.py`
6. `summarize_static_detection.py`
7. `freeze_static_detection.py`

## Step 9: mitigation and utility (`config.yaml`)

1. `build_mitigation_manifest.py`, `validate_mitigation_manifest.py`
2. `generate_mitigated_outputs.py`, `validate_mitigated_outputs.py`
3. `evaluate_mitigation_payloads.py`
4. `extract_mitigation_utility.py`
5. `validate_mitigation_results.py`
6. `run_gated_mitigation.py`
7. `summarize_mitigation.py`
8. `freeze_mitigation.py`

## Step 10: adaptive replay (`config_step10.yaml`)

Protocol freeze:

1. `build_step10_manifest.py`, `validate_step10_manifest.py`
2. `freeze_step10_protocol.py`, `verify_step10_protocol_freeze.py`

Candidate catalogue (built once, before any round):

3. `generate_adaptive_candidates.py`, `validate_adaptive_candidates.py`
4. `build_adaptive_detection_dataset.py`
5. `extract_adaptive_evaluator_features.py`, `extract_adaptive_semantic_features.py`, `extract_adaptive_unicode_features.py`
6. `validate_adaptive_detection_features.py`
7. `score_adaptive_candidates.py`

Each round R0–R3:

8. `build_adaptive_round.py`
9. `generate_adaptive_defense_outputs.py`
10. `evaluate_adaptive_payloads.py`
11. `extract_adaptive_utility.py`
12. `finalize_adaptive_feedback.py`
13. `validate_adaptive_round.py`

After R3:

14. `summarize_adaptive_results.py`
15. `freeze_adaptive_experiment.py`, `verify_step10_freeze.py`

`smoke_test_adaptive.py` runs a small end-to-end check.

## Step 11: system analysis (`config_step11.yaml`)

1. `verify_step10_freeze.py`, `verify_step11_inputs.py`
2. `freeze_step11_analysis_spec.py`, `verify_step11_analysis_spec.py`
3. `smoke_test_step11.py`
4. `run_step11_system_analysis.py`
5. `validate_step11_results.py`
6. `freeze_step11_analysis.py`, `verify_step11_freeze.py`

## Step 12: robustness analysis (`config_step12.yaml`)

1. `verify_step11_freeze.py`, `verify_step12_inputs.py`
2. `freeze_step12_analysis_spec.py`, `verify_step12_analysis_spec.py`
3. `smoke_test_step12.py`
4. `run_step12_robustness_analysis.py`
5. `validate_step12_results.py`
6. `freeze_step12_analysis.py`, `verify_step12_freeze.py`

## Repository utilities

- `verify_code_hashes.py`: checks the working tree against all freeze manifests, skipping files that are absent (such as the untracked data). Added after the experiments; it is not itself hashed.
- `export_explorer_bundle.py`: writes `outputs/explorer/explorer_bundle.json`, a compact JSON of detector predictions and all adaptive trajectories (see the main README, "Results bundle"). Needs the data bundle. Added after the experiments; not hashed.
- `build_explorer_page.py`: embeds that bundle in `explorer/template.html` and writes `outputs/explorer/index.html`, a self-contained results explorer. Added after the experiments; not hashed.
