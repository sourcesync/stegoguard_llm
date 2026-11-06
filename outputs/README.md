# Outputs

`pilot_v0/` holds the frozen results of the reported study.

Tracked in git: CSV result tables, figures, QA review notes, interpretation
notes, frozen config copies, and the SHA-256 freeze manifests
(`pilot_v0/manifests/`).

Not tracked: the JSON/JSONL records. See `../data/README.md`.

Do not edit files under `pilot_v0/`. They are hashed by the freeze manifests,
and any change will show as a `MISMATCH` in `scripts/verify_code_hashes.py`.
New runs should write to a new directory (for example `pilot_v1/`) by setting
`output_dir` in the config.
