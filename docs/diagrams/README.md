# Diagrams

Explanatory diagrams of the adaptive replay. They were made while reviewing the
paper and are not part of the frozen pilot_v0 artifacts.

| File | Contents |
|---|---|
| `threat_models_adaptive_replay.pdf` | Two threat models for the adaptive replay. Page 1: output interception with a model copy (attacker intercepts the clean answer and substitutes a stego answer). Page 2: compromised LLM decoder (attacker controls token selection inside the serving stack). Both show the StegoGuard defense (detector ensemble at the F1 threshold, deterministic or randomized rewriter) and the colluding receiver linked to the attacker out of band. |
| `adaptive_replay_example.md` | Recorded four-round trace for parent `clean_conv_001_64_mistral_000`, migration policy, deterministic rewriting. Scenario table, Mermaid sequence diagram with the one-time setup (prompt and clean answer recorded once, catalogue built and pre-scored) followed by catalogue picks per round, and an appendix of every raw text. **Current version.** |
| `adaptive_replay_example.pdf` | Earlier rendering of the same trace as a swimlane PDF. It shows per-round prompts and live encoding rather than the one-time setup and catalogue picks in the Markdown version. |

The trace texts come from `outputs/pilot_v0/attacks/full/stego_samples_full.jsonl`
and `outputs/pilot_v0/adaptive/step10/round_{0..3}/` (trajectory
`traj::clean_conv_001_64_mistral_000::channel_migration_feedback::deterministic_rewrite`).
