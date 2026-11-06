# StegoGuard-LM Step 12 robustness interpretation

Protocol: `step12_robustness_v1`

Step 12 is a **model-free robustness, sensitivity, and boundary-condition audit** over already-frozen evidence. It does not generate attacks, add adaptive rounds, retrain the detector, change thresholds, or rerun mitigation.

## Main claim-stability results

- **Migration vs frozen/no-feedback:** supported_with_sensitivity_caveats. The migration-enabled arm remains the clearest adaptive recovery mechanism after paired randomization testing and deletion sensitivity checks.
- **Migration vs fixed-channel adaptation:** directionally_supported_mixed_uncertainty. The direction favors migration, but superiority to fixed-channel adaptation is less uniform than superiority to frozen/no-feedback.
- **Fixed-channel adaptation vs frozen/no-feedback:** weak_or_mixed. Within-family search produces relatively small or unstable gains.
- **Randomized vs deterministic rewriting:** not_supported. Do not claim a general moving-target advantage under the tested four-round protocol.
- **Rank exact suppression vs residual information:** supported. Exact rank-message failure must not be interpreted as zero leakage.

## Why Step 12 matters

Step 11 supplied the primary parent-paired analysis. Step 12 asks whether those findings are artifacts of one inferential procedure, one endpoint, a small subset of influential parents/cells, one leakage metric, or one starting-channel history. The robustness package therefore adds paired sign-flip tests with BH-FDR correction, leave-one-parent-out and leave-one-cell-out influence analysis, alternative leakage outcomes, adaptive-window contrasts, initial-channel conditioning, descriptive model/domain/length heterogeneity, and clean-burden cell checks.

## Guardrails

This audit strengthens or qualifies evidence **within the frozen pilot**. It does not establish long-run equilibrium, universal channel migration, arbitrary-channel coverage, causal generalization beyond the pre-frozen attacker policy, or zero information leakage. Stratified results are sensitivity analyses, not high-powered subgroup claims.
