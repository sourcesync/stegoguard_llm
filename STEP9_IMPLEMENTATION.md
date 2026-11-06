# StegoGuard-LM Step 9: Mitigation and Utility Evaluation

## Purpose

Step 9 evaluates whether channel-appropriate mitigation destroys recoverable covert payloads while preserving the legitimate usefulness of the visible response. It separates two questions:

1. **Oracle mitigation:** If a mitigation is applied to a known stego sample, does it disrupt the payload?
2. **Detector-gated deployment:** Given the frozen Step 8 detector, does the end-to-end defense actually route suspicious outputs to the right mitigation and reduce payload survival without imposing excessive cost on clean outputs?

Step 8 is frozen before Step 9. The Step 9 implementation preserves a byte-for-byte copy of the pre-Step-9 `config.yaml` at:

`outputs/pilot_v0/manifests/step8_frozen_config.yaml`

The Step 9 manifest verifies that this file matches the config SHA-256 stored in the frozen Step 8 manifest.

## Frozen Step 9 protocol

### Original records

Step 9 starts from the 583 frozen Step 8 detection records:

- 160 clean
- 155 rank stego
- 108 synonym stego
- 160 Unicode stego

The primary matched population is the 104-parent common set:

- 104 clean
- 104 rank
- 104 synonym
- 104 Unicode
- 416 original records total

### Mitigation conditions

Each original record has five fixed mitigation tasks:

1. Unicode sanitization
2. One deterministic rewrite
3. Randomized rewrite with seed 9101
4. Randomized rewrite with seed 9102
5. Randomized rewrite with seed 9103

Therefore the frozen full task universe is:

- 583 original records x 5 mitigation realizations = **2,915 tasks**
- 416 primary matched records x 5 = **2,080 primary tasks**

### Rewriter

The frozen rewriter is the Step 8 evaluator-A model:

`google/gemma-2-2b-it`

Deterministic and randomized rewriting use the same model so the comparison isolates the decoding policy rather than model family.

Randomized rewriting uses exactly three predetermined seeds. Do not regenerate until a payload happens to break.

### Unicode sanitizer

The sanitizer:

1. applies NFC normalization;
2. removes the four reserved Step 7 Unicode-channel symbols;
3. removes Unicode category `Cf` formatting controls.

It is applied to clean, rank, synonym, and Unicode records so the intervention cost is measurable rather than label-dependent.

## Metrics

### Security

For every stego mitigation output:

- decoder success
- exact payload recovery
- effective bit accuracy
- effective bit error rate
- residual recoverable capacity

The original frozen attack decoder is used. Decoder failure counts as zero effective recovery for missing bits, while decoder success is reported separately.

### Utility

Using the frozen Step 8 evaluator-A model:

- semantic similarity between original and mitigated response
- visible instruction retention before and after
- instruction-retention delta
- evaluator mean log probability before and after
- perplexity before and after
- response token-count change
- length ratio

### Randomized rewriting

For each original stego sample across the three fixed seeds:

- payload survival frequency
- any-seed survival
- mean effective bit accuracy

### Gated deployment

The primary 416-record population is replayed through the frozen Step 8 `full_fusion` predictions at two operating points:

- Step 8 best-F1 threshold
- Step 8 approximately 1%-FPR-calibrated threshold

Suspicious Unicode codepoints always route to deterministic Unicode sanitization. Other flagged records route to deterministic or randomized rewriting. Unflagged records pass through unchanged.

## Exact local execution sequence

Run all commands from the project root.

### A. Compile and test

```powershell
cd <repo-root>
$env:PYTHONPATH="src"
python -m compileall src scripts tests
python -m unittest discover -s tests -p "test_*.py" -v
```

Expected final unit-test result for this release:

```text
Ran 69 tests
OK
```

### B. Build and validate the frozen Step 9 task universe

```powershell
python scripts/build_mitigation_manifest.py --config config.yaml --overwrite
python scripts/validate_mitigation_manifest.py --config config.yaml
```

Required counts:

```text
Detection records: 583
Mitigation tasks: 2915
Primary matched tasks: 2080
unicode_sanitization: 583
deterministic_rewrite: 583
randomized_rewrite: 1749
```

After the first mitigation output is generated, do not rebuild the task manifest.

### C. Run Unicode sanitization across all 583 records

```powershell
python scripts/generate_mitigated_outputs.py --config config.yaml --condition unicode_sanitization --resume
python scripts/validate_mitigated_outputs.py --config config.yaml --allow-pending
```

At this stage, the expected accounting is approximately:

```text
accepted_tasks: 583
terminal_failures: 0
pending_tasks: 2332
```

### D. Evaluate the sanitizer as a channel-specific sanity check

Unicode should be destroyed:

```powershell
python scripts/evaluate_mitigation_payloads.py --config config.yaml --channel unicode --condition unicode_sanitization --resume
```

Synonym should survive sanitization because ordinary lexical choices are unaffected:

```powershell
python scripts/evaluate_mitigation_payloads.py --config config.yaml --channel synonym --condition unicode_sanitization --resume
```

Rank should also normally survive sanitization because its visible token sequence is not a Unicode-format channel:

```powershell
python scripts/evaluate_mitigation_payloads.py --config config.yaml --channel rank --condition unicode_sanitization --resume
```

These are intentionally different outcomes. Sanitization is supposed to be channel-specific, not a universal language rewrite.

### E. Smoke-test deterministic rewriting on three primary records

```powershell
python scripts/generate_mitigated_outputs.py --config config.yaml --condition deterministic_rewrite --population primary_common_all --limit 3 --resume
```

Inspect the last three accepted rows in:

`outputs/pilot_v0/mitigation/step9/mitigated_outputs.jsonl`

Check that the rewrite:

- returns only the answer;
- still addresses the visible prompt;
- does not mention steganography, detection, or rewriting;
- is non-empty;
- may change length naturally.

An unchanged rewrite is not automatically rejected. It is a valid mitigation outcome and will be measured as weak disruption.

### F. Run deterministic rewriting over the full primary matched set

```powershell
python scripts/generate_mitigated_outputs.py --config config.yaml --condition deterministic_rewrite --population primary_common_all --resume
```

The primary deterministic condition contains 416 outputs.

### G. Decode deterministic rewrites

Run the cheap channels first:

```powershell
python scripts/evaluate_mitigation_payloads.py --config config.yaml --channel unicode --condition deterministic_rewrite --population primary_common_all --resume
python scripts/evaluate_mitigation_payloads.py --config config.yaml --channel synonym --condition deterministic_rewrite --population primary_common_all --resume
```

Then run rank decoding, which loads the 7B source models:

```powershell
python scripts/evaluate_mitigation_payloads.py --config config.yaml --channel rank --condition deterministic_rewrite --population primary_common_all --resume
```

### H. Smoke-test utility extraction

```powershell
python scripts/extract_mitigation_utility.py --config config.yaml --condition deterministic_rewrite --population primary_common_all --limit 2 --resume
```

If both records succeed, continue:

```powershell
python scripts/extract_mitigation_utility.py --config config.yaml --condition deterministic_rewrite --population primary_common_all --resume
```

Also quantify normalization utility on the primary set:

```powershell
python scripts/extract_mitigation_utility.py --config config.yaml --condition unicode_sanitization --population primary_common_all --resume
```

### I. Run randomized rewriting one frozen seed at a time

Seed 9101 smoke test:

```powershell
python scripts/generate_mitigated_outputs.py --config config.yaml --condition randomized_rewrite --population primary_common_all --seed 9101 --limit 3 --resume
```

If the outputs look valid, complete seed 9101:

```powershell
python scripts/generate_mitigated_outputs.py --config config.yaml --condition randomized_rewrite --population primary_common_all --seed 9101 --resume
```

Then run the other two fixed seeds:

```powershell
python scripts/generate_mitigated_outputs.py --config config.yaml --condition randomized_rewrite --population primary_common_all --seed 9102 --resume
python scripts/generate_mitigated_outputs.py --config config.yaml --condition randomized_rewrite --population primary_common_all --seed 9103 --resume
```

Do not introduce additional seeds after seeing whether the first three succeed or fail.

### J. Decode randomized rewrites

```powershell
python scripts/evaluate_mitigation_payloads.py --config config.yaml --channel unicode --condition randomized_rewrite --population primary_common_all --resume
python scripts/evaluate_mitigation_payloads.py --config config.yaml --channel synonym --condition randomized_rewrite --population primary_common_all --resume
python scripts/evaluate_mitigation_payloads.py --config config.yaml --channel rank --condition randomized_rewrite --population primary_common_all --resume
```

### K. Extract randomized utility

You can run all remaining randomized primary utility rows with one command:

```powershell
python scripts/extract_mitigation_utility.py --config config.yaml --condition randomized_rewrite --population primary_common_all --resume
```

The utility script is resumable. If interrupted, rerun the same command.

### L. Validate complete primary oracle-mitigation coverage

```powershell
python scripts/validate_mitigation_results.py --config config.yaml --population primary_common_all
```

If no rewriter infrastructure failure occurred, the ideal primary counts are:

```text
Selected tasks: 2080
Accepted outputs: 2080
Terminal generation failures: 0
Stego payload evaluations: 1560
Utility evaluations: 2080
```

Why 1,560 payload evaluations?

- 312 primary stego records
- 5 mitigation realizations each
- 312 x 5 = 1,560

### M. Simulate the frozen Step 8 detector-gated deployment

Only after the primary oracle results are complete:

```powershell
python scripts/run_gated_mitigation.py --config config.yaml --overwrite
```

Expected row count:

```text
3328
```

This comes from:

- 416 primary records
- 2 Step 8 operating points
- 1 deterministic policy + 3 randomized-seed policies
- 416 x 2 x 4 = 3,328

Validate again:

```powershell
python scripts/validate_mitigation_results.py --config config.yaml --population primary_common_all --require-gated
```

### N. Build the primary Step 9 summary and QA review

```powershell
python scripts/summarize_mitigation.py --config config.yaml
```

This writes:

- `outputs/pilot_v0/mitigation/step9/mitigation_summary.json`
- `outputs/pilot_v0/mitigation/step9/mitigation_qa_review.txt`

The QA file highlights the strongest surviving payloads, the lowest semantic-preservation cases, and the clean controls with the largest utility degradation.

**Stop here and review the primary matched experiment before extending to all 583 originals.**

### O. Extend rewriting to the full feasible inventory after primary review

The full manifest is already frozen. `--resume` processes only the 167 original records outside the primary 416-record population.

Deterministic:

```powershell
python scripts/generate_mitigated_outputs.py --config config.yaml --condition deterministic_rewrite --population all --resume
```

Randomized:

```powershell
python scripts/generate_mitigated_outputs.py --config config.yaml --condition randomized_rewrite --population all --seed 9101 --resume
python scripts/generate_mitigated_outputs.py --config config.yaml --condition randomized_rewrite --population all --seed 9102 --resume
python scripts/generate_mitigated_outputs.py --config config.yaml --condition randomized_rewrite --population all --seed 9103 --resume
```

Decode all remaining stego outputs:

```powershell
python scripts/evaluate_mitigation_payloads.py --config config.yaml --channel unicode --population all --resume
python scripts/evaluate_mitigation_payloads.py --config config.yaml --channel synonym --population all --resume
python scripts/evaluate_mitigation_payloads.py --config config.yaml --channel rank --population all --resume
```

Extract all remaining utility rows:

```powershell
python scripts/extract_mitigation_utility.py --config config.yaml --population all --resume
```

Validate the full task universe:

```powershell
python scripts/validate_mitigated_outputs.py --config config.yaml
python scripts/validate_mitigation_results.py --config config.yaml --population all
```

Ideal full counts if all rewriting tasks produce a usable output or fallback:

```text
Selected tasks: 2915
Accepted outputs: 2915
Stego payload evaluations: 2115
Utility evaluations: 2915
```

The 2,115 stego payload rows are:

- 423 accepted stego inputs
- 5 mitigation realizations each
- 423 x 5 = 2,115

Rebuild the final summary:

```powershell
python scripts/summarize_mitigation.py --config config.yaml
```

### P. Freeze Step 9 only after scientific review

Do not freeze immediately after generation. First review:

- exact payload recovery by attack and mitigation;
- bit accuracy and residual capacity;
- randomized survival frequency;
- clean utility loss;
- deterministic versus randomized tradeoffs;
- F1-threshold versus low-FPR detector-gated deployment;
- QA cases where payload survives with high utility.

Once the experiment is accepted:

```powershell
python scripts/freeze_mitigation.py --config config.yaml
```

This creates:

- `outputs/pilot_v0/manifests/step9_mitigation_freeze_manifest.json`
- `outputs/pilot_v0/manifests/step9_final_hashes.txt`

## Important stop rules

1. Do not change Step 8 predictions or thresholds during Step 9.
2. Do not add random rewrite seeds after seeing results.
3. Do not regenerate a rewrite until the payload happens to break.
4. Do not discard utility-degraded rewrites. They are mitigation costs.
5. Do not discard payload-surviving rewrites. They are security failures.
6. Do not silently treat decoder failure as exact payload recovery. Decoder success, exact recovery, and bit recovery are separate fields.
7. If an infrastructure error occurs, stop and resume after fixing the environment. Do not classify a CUDA or memory failure as scientific mitigation failure.
8. Do not run `freeze_mitigation.py` until the primary and, if retained, full-domain results have been reviewed.
