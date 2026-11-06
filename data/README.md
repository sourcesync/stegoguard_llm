# Data

## Tracked inputs

| File | Contents |
|---|---|
| `topics/topics.csv` | The 10 topics (5 conversational, 5 technical) used to render prompts |
| `topics/conflict_templates.csv` | Conflict-prompt templates from an earlier design; rendered but filtered out before clean generation, so not used in the reported experiments |
| `attacks/synonym_codebook.yaml` | The eight binary synonym slots used by the synonym channel |

## Large outputs (not tracked)

The experiment's JSON and JSONL records are distributed separately as a data
bundle. Unpack it at the repository root so the files land under
`outputs/pilot_v0/` at the paths listed in the freeze manifests
(`outputs/pilot_v0/manifests/*_hashes.txt`), then run:

```bash
python scripts/verify_code_hashes.py
```

All entries should then report `match`.

Main files in the bundle:

| Path under `outputs/pilot_v0/` | Contents |
|---|---|
| `prompts/rendered_prompts.jsonl` | Rendered prompts |
| `clean/clean_samples_full.jsonl` | 160 clean parents |
| `attacks/full/stego_samples_full.jsonl` | 423 accepted attacks with payloads, keys and decodes |
| `detection/step8/` | Detection dataset, splits, features, out-of-fold predictions |
| `mitigation/step9/` | Oracle and gated mitigation outputs, payload results, utility |
| `adaptive/step10/` | Candidate catalogue, trajectory manifest, per-round defense outputs and payload results |

**Bundle location:** _to be added (e.g. Zenodo DOI)._
