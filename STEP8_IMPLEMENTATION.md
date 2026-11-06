# StegoGuard-LM Step 8: Static Detection Implementation and Local Run Guide

## What Step 8 does

Step 8 is the first direct evaluation of the StegoGuard-LM defense. It uses the frozen Step 5 clean corpus and the accepted Step 7 steganographic corpus. It does not regenerate attacks.

The protocol is `step8_static_v1`.

The detector is evaluated using:

1. Distributional features from two independent evaluator LMs.
2. Cross-model disagreement features derived using clean-training-fold normalization only.
3. Semantic consistency features.
4. A separate deterministic Unicode hygiene layer.
5. Regularized logistic-regression fusion with leave-one-topic-out outer evaluation.
6. Nested leave-one-topic-out threshold calibration inside each outer training fold.
7. Parent-clustered bootstrap confidence intervals.

The primary fair comparison set is the 104-parent common set in which clean, rank, synonym, and Unicode versions all exist, giving 416 records. A 48-parent balanced common subset gives 192 records for robustness analysis.

## Important Step 8 evaluator freeze

Before any Step 8 evaluator scores were generated, the evaluator IDs were frozen as instruction-tuned variants:

- evaluator_a: `google/gemma-2-2b-it`
- evaluator_b: `Qwen/Qwen2.5-3B-Instruct`

This is intentional because Step 8 conditions scoring on the visible user prompt and the semantic features require deterministic instruction-following behavior. No Step 8 scientific model scores existed before this correction.

If Hugging Face reports that a model requires authentication or license acceptance, resolve that access in your Hugging Face account and authenticate locally. Do not put an access token into `config.yaml`.

## New Step 8 code

### Package

`src/stegoguard_lm/detection/`

- `__init__.py`
- `io_utils.py`
- `datasets.py`
- `splits.py`
- `unicode_features.py`
- `evaluator_scoring.py`
- `semantic.py`
- `features.py`
- `modeling.py`
- `metrics.py`

### Scripts

- `scripts/build_detection_dataset.py`
- `scripts/validate_detection_dataset.py`
- `scripts/build_detection_splits.py`
- `scripts/validate_detection_splits.py`
- `scripts/extract_unicode_features.py`
- `scripts/extract_evaluator_features.py`
- `scripts/extract_semantic_features.py`
- `scripts/validate_detection_features.py`
- `scripts/run_static_detection.py`
- `scripts/summarize_static_detection.py`
- `scripts/freeze_static_detection.py`

### New tests

- `tests/test_detection_dataset.py`
- `tests/test_detection_splits.py`
- `tests/test_unicode_detection_features.py`
- `tests/test_evaluator_feature_math.py`
- `tests/test_detection_features.py`
- `tests/test_semantic_detection_features.py`
- `tests/test_detection_metrics.py`
- `tests/test_detection_modeling.py`

## Model-free checks already completed in the packaged project

The packaged project was tested without external model weights:

- Python compilation: PASS
- Full unit and synthetic integration suite: 50/50 PASS
- Actual Step 7 corpus -> Step 8 detection dataset build: PASS
- Detection dataset: 583 records = 160 clean + 423 accepted stego
- Common all-three parents: 104
- Balanced common parents: 48
- Leave-one-topic-out folds: 10
- Sample leakage: none detected
- Parent-lineage leakage: none detected
- Topic leakage: none detected
- Unicode feature extraction on all 583 records: PASS
- Suspicious Unicode records: 160, matching the 160 accepted Unicode attack records
- Synthetic end-to-end detector pipeline, including all ablations, nested threshold calibration, channel-specific runs, metrics, and result summarization: PASS

The actual evaluator-LM distributional and semantic feature extraction cannot be run in the packaging environment because the Hugging Face `transformers` package and your local model cache are not available there. Those are the local-machine integration steps below.

# Exact local execution sequence

Run all commands from your repository root, for example:

```powershell
cd <repo-root>
$env:PYTHONPATH="src"
```

## A. Compile and run all tests

```powershell
python -m compileall src scripts tests
python -m unittest discover -s tests -p "test_*.py" -v
```

Expected ending:

```text
Ran 50 tests
OK
```

If Python reports that `sklearn`, `numpy`, or `scipy` is missing, install only the missing scientific packages, for example:

```powershell
pip install numpy scipy scikit-learn
```

Do not upgrade PyTorch or Transformers just because of Step 8 unless an actual dependency error requires it.

## B. Rebuild the deterministic detection dataset locally

The complete project already includes the model-free Step 8 dataset and splits, but rebuild them once locally before model scoring so the hashes reflect your exact project copy.

```powershell
python scripts/build_detection_dataset.py --config config.yaml --overwrite
```

Required output:

```text
Records: 583
Clean: 160
Stego: 423
Common all-three parents: 104
Balanced common parents: 48
```

Validate:

```powershell
python scripts/validate_detection_dataset.py --config config.yaml
```

Required high-level counts:

```text
Total records: 583
Clean: 160
Stego: 423
rank_encoding: 155
synonym_slot_codebook: 108
unicode_hidden_channel: 160
Common all-three parents: 104
Balanced common parents: 48
```

## C. Build and validate leakage-safe topic folds

```powershell
python scripts/build_detection_splits.py --config config.yaml --overwrite
python scripts/validate_detection_splits.py --config config.yaml
```

Required result:

```text
Strategy: leave_one_topic_out
Folds: 10
No sample, parent-lineage, or topic leakage detected.
```

Do not replace this with a random row split. The topic-disjoint protocol is intentional.

## D. Extract deterministic Unicode hygiene features

```powershell
python scripts/extract_unicode_features.py --config config.yaml --overwrite
```

Expected:

```text
Records: 583
Records with suspicious codepoints: 160
```

This feature family is deterministic and is reported separately from the core linguistic detector.

## E. Smoke-test evaluator A distributional scoring

Evaluator A is:

```text
google/gemma-2-2b-it
```

Run three records first:

```powershell
python scripts/extract_evaluator_features.py --config config.yaml --evaluator evaluator_a --resume --limit 3
```

You should see three completed sample IDs and a summary with approximately:

```text
record_count: 3
expected_record_count: 583
complete: false
```

If this succeeds, continue the same file without regenerating those records:

```powershell
python scripts/extract_evaluator_features.py --config config.yaml --evaluator evaluator_a --resume
```

Final required state:

```text
record_count: 583
expected_record_count: 583
complete: true
```

Every accepted feature row is appended and fsync'd immediately. If the process is interrupted, rerun the same `--resume` command.

## F. Smoke-test evaluator B distributional scoring

Evaluator B is:

```text
Qwen/Qwen2.5-3B-Instruct
```

First three:

```powershell
python scripts/extract_evaluator_features.py --config config.yaml --evaluator evaluator_b --resume --limit 3
```

Then full resume:

```powershell
python scripts/extract_evaluator_features.py --config config.yaml --evaluator evaluator_b --resume
```

Required final state:

```text
record_count: 583
complete: true
```

## G. Smoke-test semantic feature extraction

The semantic stage currently uses evaluator A and computes:

1. `visible_instruction_retention`, a deterministic YES-vs-NO conditional-likelihood judgment of whether the candidate response addresses the visible prompt.
2. `paraphrase_embedding_similarity`, using a deterministic evaluator-generated paraphrase and the evaluator's hidden-state representation.

These are candidate-only features. The clean parent, attack family, payload, and labels are never supplied to the semantic evaluator.

Start with two records:

```powershell
python scripts/extract_semantic_features.py --config config.yaml --resume --limit 2
```

Inspect the two terminal scores. Both feature values must be finite. The retention value must be in `[0,1]` and cosine similarity in `[-1,1]`.

Then continue all remaining records:

```powershell
python scripts/extract_semantic_features.py --config config.yaml --resume
```

Required final state:

```text
record_count: 583
complete: true
```

This is likely the slowest Step 8 extraction phase because it performs deterministic paraphrasing. It is resumable. Do not use `--overwrite` after successful records exist unless we deliberately restart the entire semantic protocol.

## H. Validate all feature families before fitting a detector

Only after evaluator A, evaluator B, semantic, and Unicode feature files each contain all 583 sample IDs, run:

```powershell
python scripts/validate_detection_features.py --config config.yaml
```

Required result:

```text
Step 8 feature validation passed.
Expected records: 583
Complete records across all feature families: 583
Complete: True
```

The validator also creates:

```text
outputs/pilot_v0/detection/step8/features/feature_manifest.json
```

The model-fitting script refuses to run against feature files whose hashes differ from this manifest.

## I. Run the full static detector evaluation

```powershell
python scripts/run_static_detection.py --config config.yaml --overwrite
```

This does not load the 7B source models. It operates on the frozen feature tables using scikit-learn.

It runs:

- distributional only
- cross-model only
- semantic only
- distributional + cross-model
- distributional + semantic
- cross-model + semantic
- full linguistic fusion
- balanced-common full-fusion robustness check
- channel-specific full-fusion models for rank, synonym, and Unicode
- deterministic Unicode hygiene evaluation
- a separately reported deployed score that adds Unicode hygiene as an override to the core linguistic fusion

For the primary 104-parent matched population, outer evaluation is leave-one-topic-out. Thresholds for F1, 5% FPR, and 1% FPR are calibrated using nested leave-one-topic-out predictions from the outer training topics only.

The cross-model features are also leakage-safe. The 5th-percentile tail threshold and evaluator normalization statistics are computed only from clean records inside the relevant training fold.

Expected output files:

```text
outputs/pilot_v0/detection/step8/predictions.jsonl
outputs/pilot_v0/detection/step8/metrics.json
```

Do not interpret results from the terminal alone. Keep the full metrics file.

## J. Summarize results and generate QA cases

```powershell
python scripts/summarize_static_detection.py --config config.yaml
```

This creates:

```text
outputs/pilot_v0/detection/step8/static_detection_summary.json
outputs/pilot_v0/detection/step8/false_positive_review.txt
outputs/pilot_v0/detection/step8/false_negative_review.txt
```

The review files contain the highest-scoring clean responses and lowest-scoring stego responses under the primary full-fusion detector.

## K. Stop and review the results together

At this point, send the following terminal output and preferably the updated ZIP back into the conversation:

1. Final evaluator A feature summary.
2. Final evaluator B feature summary.
3. Final semantic feature summary.
4. `validate_detection_features.py` output.
5. `run_static_detection.py` primary full-fusion metrics.
6. `summarize_static_detection.py` output.

Do not change detector features, evaluator models, regularization, folds, or thresholds after seeing the result without versioning a new protocol.

## L. Freeze Step 8 only after scientific review

The freeze script is included, but do not run it until the Step 8 results and QA cases have been reviewed together.

When approved:

```powershell
python scripts/freeze_static_detection.py --config config.yaml
```

It will create:

```text
outputs/pilot_v0/manifests/step8_static_detection_manifest.json
outputs/pilot_v0/manifests/step8_final_hashes.txt
```

# Important methodological protections implemented

## Detector input leakage

The detector feature extractors receive only:

```text
prompt
candidate text
```

Ground-truth fields such as attack name, payload, source model, domain, and topic are analysis metadata only.

## Topic leakage

The primary outer split is leave-one-topic-out, not a random row split.

## Parent leakage

All descendants of a clean parent stay in the same topic fold.

## Cross-model normalization leakage

Evaluator means, standard deviations, and the clean 5th-percentile token-logp threshold are fitted inside each training fold only.

## Threshold leakage

The outer test fold is not used to select F1 or FPR thresholds. Thresholds come from inner training-topic OOF predictions.

## Unicode inflation

The deterministic Unicode scanner is reported separately. The headline core linguistic StegoGuard result is distributional + cross-model + semantic. A second deployed score adds the Unicode hygiene override.

## False-positive operating points

Primary operating points are 5% and 1% FPR. The older 0.1% target is retained only as an exploratory concept because 160 clean controls are insufficient for stable direct empirical estimation at 0.1% FPR.

## Confidence intervals

AUROC and AUPRC confidence intervals use parent-cluster bootstrap resampling so clean/stego descendants of the same parent are resampled together.

# Step 8 completion gate

Step 8 is ready to freeze only when:

- 583 detection rows validate.
- 10 topic-disjoint folds validate.
- Both evaluator feature tables contain 583 rows.
- Semantic feature table contains 583 rows.
- Unicode feature table contains 583 rows.
- Feature hash manifest validates.
- Static detection produces out-of-topic predictions for the full primary population.
- Ablation results are available.
- Rank, synonym, and Unicode performance is reported separately.
- Source model, domain, and nominal-length stratification is available.
- False-positive and false-negative QA cases have been inspected.
- No protocol changes are made post-result without a new version.
