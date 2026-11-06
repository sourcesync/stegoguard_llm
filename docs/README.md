# Documentation index

The step documents live at the repository root because several are covered by
the freeze manifests (moving them would break hash verification). This page
indexes them by pipeline step.

| Step | Stage | Protocol | Implementation | Running instructions | Test report |
|---|---|---|---|---|---|
| 6 | Attack infrastructure | | [STEP6_IMPLEMENTATION](../STEP6_IMPLEMENTATION.md) | | |
| 8 | Static detection | | [STEP8_IMPLEMENTATION](../STEP8_IMPLEMENTATION.md) | (in implementation doc) | |
| 9 | Mitigation and utility | | [STEP9_IMPLEMENTATION](../STEP9_IMPLEMENTATION.md) | (in implementation doc) | |
| 10 | Adaptive replay | [STEP10_PROTOCOL](../STEP10_PROTOCOL.md) | [STEP10_IMPLEMENTATION](../STEP10_IMPLEMENTATION.md) | (in implementation doc) | |
| 11 | System analysis | [STEP11_PROTOCOL](../STEP11_PROTOCOL.md) | [STEP11_IMPLEMENTATION](../STEP11_IMPLEMENTATION.md) | [STEP11_Local_Running_Instructions](../STEP11_Local_Running_Instructions.md) | [STEP11_TEST_REPORT](../STEP11_TEST_REPORT.md) |
| 12 | Robustness analysis | [STEP12_PROTOCOL](../STEP12_PROTOCOL.md) | [STEP12_IMPLEMENTATION](../STEP12_IMPLEMENTATION.md) | [STEP12_Local_Running_Instructions](../STEP12_Local_Running_Instructions.md) | [STEP12_TEST_REPORT](../STEP12_TEST_REPORT.md) |

Steps 1–5 (prompt rendering, clean corpus) and Step 7 (full attack corpus) have
no separate documents; their configuration is in `config.yaml` and their
outputs and hashes are under `outputs/pilot_v0/`.

The running instructions were written for Windows PowerShell with
`$env:PYTHONPATH="src"`. After `pip install -e .` that setting is unnecessary;
`<repo-root>` stands for the local checkout path.

## Other material

- `planning/new_mcs_inclusive_plan.docx`: experiment plan (July 2026).
- `planning/old_experiment_steps.docx`: earlier step plan (March 2026).
- `figures/taxonomy_early_draft.png`: early taxonomy figure of LM steganography and defenses, not used in the current draft.
