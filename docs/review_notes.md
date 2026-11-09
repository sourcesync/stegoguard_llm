# Review notes

Findings from re-checking the draft against the frozen pilot_v0 records. Every
number here was recomputed from `outputs/explorer/explorer_bundle.json`
(produced by `scripts/export_explorer_bundle.py`) or read from the source
files named below. These are working notes for revising the paper; decide
before public release whether to keep this file.

## 1. The paper's headline numbers reproduce

From the out-of-fold predictions (`detection/step8/predictions.jsonl`,
104 matched parents, deployed detector = full fusion + Unicode override):

| Quantity | Recomputed | Paper |
|---|---|---|
| Pooled AUROC | 0.9037 | 0.9037 |
| AUROC rank / synonym / Unicode | 0.9859 / 0.7251 / 1.0000 | same |
| Clean answers flagged, F1 gate | 35 / 104 | 33.7% |
| Clean answers flagged, 1% FPR gate | 1 / 104 | 0.96% |
| Rank / synonym / Unicode flagged, F1 gate | 104 / 72 / 104 | Table 5 |
| Rank / synonym / Unicode flagged, 1% FPR gate | 51 / 1 / 104 | Table 5 |

Adaptive replay successful bits (`adaptive/step10/round_*/payload_results.jsonl`),
summed R0–R3:

| Rewriter | Frozen | Fixed-channel | Migration |
|---|---|---|---|
| Deterministic | 40 | 44 | 67 |
| Randomized | 41 | 47 | 70 |

These match Table 6.

## 2. The linguistic classifier already detects Unicode attacks

Evaluator features are computed on the raw response text, so zero-width and
tag characters reach the scoring models. Without the Unicode override (the
`full_fusion` variant), the classifier alone reaches:

- Unicode AUROC 0.9795; median Unicode score 0.937 vs 0.201 for clean answers.
- All 104 Unicode attacks flagged at the F1 gate; 45 of 104 at the 1% FPR gate.

So the pooled linguistic AUROC (0.8968) and the feature-family comparison
include a tokenization signal, not only wording. Reporting linguistic
detection on rank and synonym alone, with Unicode left to the scanner, would
isolate it.

## 3. Throughput counts repeated deliveries

A success under deterministic rewriting repeats in every later round with the
same message, and each repeat is counted. Counting each channel's message once
per trajectory:

| Condition | Bits counted (Table 6) | Distinct bits delivered | Trajectories with any recovery |
|---|---|---|---|
| Deterministic · frozen | 40 | 10 | 6 / 48 |
| Deterministic · fixed-channel | 44 | 12 | 7 / 48 |
| Deterministic · migration | 67 | 19 | 12 / 48 |
| Randomized · frozen | 41 | 11 | 7 / 48 |
| Randomized · fixed-channel | 47 | 15 | 9 / 48 |
| Randomized · migration | 70 | 21 | 14 / 48 |

Migration's distinct gain over frozen under deterministic rewriting is 9 bits,
matching Table 7's 0.1875 bits per parent (9/48).

## 4. Most synonym messages are one bit

Synonym payload lengths across the 48 adaptive parents: 28 carry 1 bit, 15
carry 2, 4 carry 3, 1 carries 4. A 1-bit message has a 50% guessing baseline,
which matters when exact recovery is the success measure. Rank never succeeds
in the replay, so every recovery is one of these short messages.

## 5. Six recoveries after a rewrite; four are coincidental

Six rounds count a payload as recovered even though the detector flagged the
text and it was rewritten. All six are synonym attacks under randomized
rewriting. Comparing where the slot word sits in the sent and delivered text:

| Parent | Policy | Round | Slot word | Same phrase after rewrite? |
|---|---|---|---|---|
| conv_001_64_mistral_000 | migration | R1 | important | No: rewriter used "important" in a different sentence |
| conv_005_256_mistral_000 | fixed-channel | R2 | use / enhance | Yes: "use it as an opportunity to enhance your work" survived |
| tech_005_256_mistral_001 | frozen | R1 | crucial | Yes: "Another crucial aspect of tokenization" survived |
| tech_005_256_mistral_001 | fixed-channel | R2 | crucial | No: different sentence |
| tech_005_256_mistral_001 | migration | R2 | crucial | No: different sentence |
| tech_005_256_mistral_001 | migration | R3 | crucial | No: different sentence |

The synonym decoder accepts a slot word anywhere in the text (whole-word,
case-insensitive match in `src/stegoguard_lm/attacks/synonym_channel.py`), so
the rewriter's own word choices can count as recoveries. "Crucial" and
"important" are common in model-written text, which makes this likely.

These six rounds are the entire difference between the two rewriters in
Table 6:

- Migration, 70 vs 67: conv_001 R1 (1 bit) and tech_005 R2–R3 (2 bits), all
  coincidental. Randomization's advantage for the migration attacker comes
  entirely from rewriter word choices matching the codebook.
- Fixed-channel, 47 vs 44: conv_005 R2 (2 bits, genuine survival) and
  tech_005 R2 (1 bit, coincidental).
- Frozen, 41 vs 40: tech_005 R1 (1 bit, genuine survival).

A position-aware decoder check would remove the coincidental cases.

## 6. StegoGuard rewrites some clean answers in the replay

In round R0 under deterministic rewriting, the detector ensemble flagged 16 of
the 48 parents' clean answers. The traced example `clean_conv_001_64_mistral_000`
was flagged by 9 of 10 models.

## 7. Other points raised in review

- Rank attacks carry visible artifacts (broken tokens, misspellings, language
  switches), and the fixed synonym codebook produces grammar errors such as
  "may outcome in". Both likely contribute to detection.
- The adaptive replay grants the attacker perfect, timely, cost-free outcome
  feedback while the defender is fixed. Each candidate's outcome is
  precomputed, so under deterministic rewriting the rounds act as an attempt
  budget over a fixed catalogue.
- The adaptive replay runs only at the F1 gate. At the 1% FPR gate about half
  of rank attacks pass unchanged.
- Matched parents are unevenly spread across cells (Mistral technical 64 kept
  6 of 20; see `detection_dataset_manifest.json`).

See `docs/diagrams/` for the threat models and a recorded example trace.
