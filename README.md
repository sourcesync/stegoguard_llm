# StegoGuard-LM

Code and frozen artifacts for **StegoGuard-LM: Evaluating End-to-End Protection Against LLM Covert Channels**.

StegoGuard-LM is a controlled framework for evaluating defenses against hidden messages in language-model outputs. It follows each response through attack construction, detection, intervention (sanitization or rewriting), and payload recovery, and measures what the same defense does to clean responses. It covers token-rank and synonym-slot channels, a Unicode concealment control, static and detector-gated mitigation, and a four-round adaptive replay in which an attacker switches strategies or channels based on recovery feedback.

> **Anonymity.** The paper is under anonymous review. Before sharing this repository with reviewers, publish it through an anonymizing service and check that commit metadata, file contents, and paths identify no author (see [Release checklist](#release-checklist)).

## Repository layout

```
stegoguard-lm/
├── README.md
├── pyproject.toml              package metadata and dependencies
├── config.yaml                 base configuration (Steps 1–9)
├── config_step10.yaml          adaptive replay protocol
├── config_step11.yaml          system analysis
├── config_step12.yaml          robustness analysis
├── STEP*_PROTOCOL.md           frozen experiment protocols
├── STEP*_IMPLEMENTATION.md     implementation notes per step
├── STEP*_TEST_REPORT.md        test reports
├── STEP*_Local_Running_Instructions.md
├── src/stegoguard_lm/          importable package
│   ├── attacks/                rank, synonym and Unicode channels; payloads; validation
│   ├── detection/              features, splits, modeling, metrics
│   ├── mitigation/             normalization, rewriting, gated routing, utility
│   ├── adaptive/               candidate catalogue, strategies, rounds, ensemble detector
│   ├── analysis/               Step 11 system analysis
│   ├── robustness/             Step 12 robustness analysis
│   ├── clean_generation.py
│   └── prompt_rendering.py
├── scripts/                    one entry point per pipeline stage (see scripts/README.md)
├── explorer/                   source for the interactive results explorer page
├── tests/                      unit tests
├── data/                       small inputs: topics, prompt templates, synonym codebook
├── outputs/pilot_v0/           frozen results: CSVs, figures, QA reviews, hash manifests
└── docs/                       step-document index, diagrams, planning notes, figures
```

The protocol documents, configs, scripts, and source modules stay at the paths where they were frozen. Each step's freeze manifest records file paths relative to the repository root, so moving these files would break verification. See [Frozen artifacts](#frozen-artifacts-and-verification).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .                   # analysis and model-free code
pip install -e ".[models]"         # adds torch and transformers for generation, encoding, features, rewriting
```

Installing the package replaces the `PYTHONPATH=src` setting used in the step running instructions. All scripts are run from the repository root.

## Pipeline

Each step consumes frozen outputs from the step before it. Stage-by-stage commands are in [`scripts/README.md`](scripts/README.md) and in each step's implementation document.

| Step | Stage | Paper | Outputs |
|---|---|---|---|
| 1–5 | Prompt rendering and clean corpus (160 parents) | §3.3 | `outputs/pilot_v0/prompts/`, `clean/` |
| 6–7 | Attack construction and validation (423 accepted attacks) | §3.3, Table 2–3 | `outputs/pilot_v0/attacks/` |
| 8 | Static detection, leave-one-topic-out | §3.4, §4.1 | `outputs/pilot_v0/detection/step8/` |
| 9 | Oracle and detector-gated mitigation, utility | §3.5, §4.2 | `outputs/pilot_v0/mitigation/step9/` |
| 10 | Adaptive replay, R0–R3 | §3.6, §4.3 | `outputs/pilot_v0/adaptive/step10/` |
| 11 | System analysis | §3.7–3.8, §4.4–4.6 | `outputs/pilot_v0/analysis/step11/` |
| 12 | Robustness and sensitivity analysis | §3.8, Appendix B | `outputs/pilot_v0/analysis/step12/` |

## Frozen artifacts and verification

Steps 7–12 each end with a freeze script that writes SHA-256 hashes of every input, config, protocol, script, module, and output the step depended on (`outputs/pilot_v0/manifests/*_hashes.txt`). To check a working tree against them:

```bash
python scripts/verify_code_hashes.py              # summary
python scripts/verify_code_hashes.py --show-missing
```

On a fresh clone without the data bundle, every code, config, and document file should match. The remaining entries are the large JSON/JSONL outputs, which show as missing until the data bundle is unpacked. The per-step `verify_step*_freeze.py` scripts perform the full checks and require the data.

Line-ending normalization is disabled (`.gitattributes`) because many frozen files have CRLF endings and must stay byte-identical.

## Data

Large outputs (JSON/JSONL corpora, features, defense outputs, payload results) are not tracked in git. See [`data/README.md`](data/README.md) for what they contain and how to obtain them.

## Results bundle

`scripts/export_explorer_bundle.py` collects the frozen results into one compact JSON file (about 1.2 MB) for quick analysis or an interactive explorer:

```bash
python scripts/export_explorer_bundle.py          # writes outputs/explorer/explorer_bundle.json
```

It requires the data bundle to be unpacked. The output contains:

- `op`: out-of-fold detector predictions for the 104 matched parents (416 records) for the deployed detector and each feature-family variant, with each fold's F1, 5% FPR and 1% FPR decisions.
- `parents`: the 48 adaptive-replay parents, with prompt, clean answer, candidate counts, and how StegoGuard routed the clean answer each round.
- `trajs`: all 288 adaptive trajectories, R0–R3, with the chosen candidate, selection reason, detector votes, sent and delivered text, payload length, exact recovery, decoded bits and slot details.
- `texts`: a deduplicated text table referenced by index.

The output is derived data and is not tracked in git. Recomputing from it reproduces the pooled AUROC (0.9037), the per-channel AUROCs, the gate counts in Table 5, and the Table 6 totals.

## Results explorer

`explorer/template.html` is a single-page viewer for the bundle. The **Detector** tab recomputes flagged rates and AUROC per channel for any feature set and threshold, with a strip plot of the out-of-fold scores. The **Replay** tab shows the Table 6 totals and lets you step through any of the 288 trajectories, round by round, with the detector votes, the sent and delivered text, and the decode result.

```bash
python scripts/export_explorer_bundle.py     # needs the data bundle
python scripts/build_explorer_page.py        # writes outputs/explorer/index.html
```

`index.html` is self-contained (the data is embedded), so open it directly in a browser or host it as a static file. It is derived data and is not tracked in git. Fonts load from Google Fonts when online and fall back to system fonts otherwise. If the explorer is published with the review copy, it contains the full model outputs, including the stego texts.

## Review notes

[`docs/review_notes.md`](docs/review_notes.md) records findings from re-checking the draft against the frozen records, each recomputed from the results bundle. In short:

| Finding | Evidence |
|---|---|
| Headline numbers reproduce | AUROC 0.9037; rank/synonym/Unicode 0.9859 / 0.7251 / 1.0000; Table 6 totals 40/44/67 and 41/47/70 |
| The linguistic classifier already detects Unicode attacks | Without the override: Unicode AUROC 0.9795, all 104 flagged at the F1 gate, 45 at 1% FPR |
| Throughput counts repeated deliveries | Deterministic migration: 67 bits counted, 19 distinct; frozen: 40 counted, 10 distinct |
| Most synonym messages are one bit | Of the 48 adaptive parents: 28 one-bit, 15 two-bit, 4 three-bit, 1 four-bit |
| Randomization's edge is mostly decoder coincidence | All six recoveries after a rewrite are randomized synonym rounds; in four the slot word appears in a different sentence. They account for the whole randomized vs deterministic difference |
| StegoGuard rewrites clean answers in the replay | 16 of 48 clean answers flagged in R0 (deterministic) |

## Tests

```bash
python -m unittest discover -s tests
```

67 tests run without model dependencies. Another 9 (rank channel, generation, mitigation and adaptive candidate tests) need the `models` extra.

## Environment

The frozen outputs were produced with Python 3.13 on Windows with CUDA. Models: Qwen2.5-7B-Instruct and Mistral-7B-Instruct-v0.3 (generation), Gemma-2-2B-IT and Qwen2.5-3B-Instruct (evaluators), Gemma-2-2B-IT (rewriter). Exact package versions and model revisions should be recorded here before release.

## Release scope

Consistent with the paper's ethics statement, this repository contains the evaluated channels, the fixed attacker policies, defensive evaluation code, analysis scripts, configurations, and the records needed to reproduce the reported results. It does not include tools for open-ended strategy discovery or attacks on live systems.

## Release checklist

- [ ] Pin package versions and model revisions (Environment).
- [ ] Choose and add a `LICENSE`.
- [ ] Publish the data bundle and fill in `data/README.md`.
- [ ] For review: anonymize. `outputs/pilot_v0/manifests/step7_final_hashes.txt` records absolute local paths containing a user name; it is a frozen artifact, so redact it only in the anonymized copy.
- [ ] Decide whether `data/topics/conflict_templates.csv` and the conflict-prompt code path belong in the release (they are not used by the reported experiments).
- [ ] Decide whether `docs/review_notes.md` stays in the public release (internal working notes).
- [ ] Add `CITATION.cff` after the review period.

## Citation

Anonymous ACL submission. Citation details will be added after review.
