# StegoGuard-LM Step 6 Implementation

This project version adds the Step 6 attack infrastructure while preserving the frozen 160-record Step 5 clean corpus.

## Files added

```text
src/stegoguard_lm/attacks/
├── __init__.py
├── base.py
├── payloads.py
├── rank_channel.py
├── records.py
├── synonym_channel.py
├── unicode_channel.py
└── validation.py

data/attacks/synonym_codebook.yaml

scripts/
├── audit_synonym_codebook.py
├── smoke_test_attacks.py
└── validate_attack_smoke.py

tests/
├── __init__.py
├── test_payloads.py
├── test_rank_channel.py
├── test_rank_partition.py
├── test_synonym_decoder.py
├── test_synonym_encoder.py
└── test_unicode_channel.py
```

`config.yaml` was extended with deterministic payload settings, attack parameters, the four-record core smoke matrix, and Step 6 output paths.

## Frozen Step 5 corpus

The frozen corpus manifest is:

```text
outputs/pilot_v0/manifests/step5_clean_corpus_manifest.json
```

The recorded SHA-256 is:

```text
5765dbf1271d90af39ab61e87dcbf29aca2d54481d937a7ab246162d82940365
```

Do not edit `outputs/pilot_v0/clean/clean_samples_full.jsonl`.

## First local commands

Open PowerShell in the repository root:

```powershell
$env:PYTHONPATH="src"
python -m compileall src scripts tests
python -m unittest discover -s tests -p "test_*.py" -v
```

Audit the codebook with both source tokenizers:

```powershell
python scripts/audit_synonym_codebook.py --config config.yaml
```

The repository was structurally audited without tokenizers during construction. The full tokenizer audit must run locally because it loads the Qwen and Mistral tokenizers.

## Integration sequence

### 1. Unicode channel on one parent

```powershell
python scripts/smoke_test_attacks.py --config config.yaml --channel unicode --sample-id clean_conv_001_64_qwen_000 --overwrite

python scripts/validate_attack_smoke.py --config config.yaml --samples outputs/pilot_v0/attacks/smoke/attack_smoke_samples.jsonl --clean-samples outputs/pilot_v0/clean/clean_samples_full.jsonl --expected-count 1
```

### 2. Add the synonym channel

```powershell
python scripts/smoke_test_attacks.py --config config.yaml --channel synonym --sample-id clean_conv_001_64_qwen_000 --resume

python scripts/validate_attack_smoke.py --config config.yaml --samples outputs/pilot_v0/attacks/smoke/attack_smoke_samples.jsonl --clean-samples outputs/pilot_v0/clean/clean_samples_full.jsonl --expected-count 2
```

### 3. Add the rank channel

```powershell
python scripts/smoke_test_attacks.py --config config.yaml --channel rank --sample-id clean_conv_001_64_qwen_000 --resume

python scripts/validate_attack_smoke.py --config config.yaml --samples outputs/pilot_v0/attacks/smoke/attack_smoke_samples.jsonl --clean-samples outputs/pilot_v0/clean/clean_samples_full.jsonl --expected-count 3 --verify-rank-with-model
```

### 4. Run the final 12-record core matrix

This intentionally replaces the three-record development smoke output.

```powershell
python scripts/smoke_test_attacks.py --config config.yaml --channel all --matrix core --overwrite

python scripts/validate_attack_smoke.py --config config.yaml --samples outputs/pilot_v0/attacks/smoke/attack_smoke_samples.jsonl --clean-samples outputs/pilot_v0/clean/clean_samples_full.jsonl --expected-count 12 --verify-rank-with-model
```

The core matrix covers Qwen and Mistral, conversational and technical prompts, and short and long nominal conditions.

## Output files

```text
outputs/pilot_v0/attacks/smoke/attack_smoke_samples.jsonl
outputs/pilot_v0/attacks/smoke/attack_smoke_failures.jsonl
outputs/pilot_v0/attacks/smoke/attack_smoke_preview.txt
outputs/pilot_v0/attacks/smoke/attack_smoke_summary.json
```

Accepted records are appended and synchronized to disk immediately. Use `--resume` after interruption. Use `--overwrite` only for a deliberate fresh start.

## Channel invariants

Token white-box and token black-box records must equal the clean parent's `actual_token_length` under the source tokenizer.

Unicode records use a different invariant: removing the framed hidden codepoints must reproduce the parent text exactly. Raw Unicode stego token length is recorded rather than assumed to remain unchanged.
