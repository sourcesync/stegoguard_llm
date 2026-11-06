# StegoGuard-LM Step 12 Protocol — Robustness, Sensitivity, and Boundary-Condition Audit

Protocol: `step12_robustness_v1`

## Purpose

Step 12 stress-tests the already-frozen Step 11 conclusions without creating a new attack experiment. It is deliberately **model-free, non-generative, and read-only with respect to Steps 1–11**. No attacks are regenerated, no adaptive rounds are added, the detector is not retrained, thresholds are not changed, and mitigators are not rerun.

The goal is to determine whether the main adaptive findings depend too heavily on one endpoint, one inferential procedure, a small number of parents or balanced cells, one leakage metric, or one starting-channel history.

## Frozen inputs

Step 12 runs only if the final freezes for Steps 8, 9, 10, and 11 verify successfully. The main adaptive data remain the 48-parent, 288-trajectory Step 10 system analyzed in Step 11: 1,152 stego rows and 384 clean-control rows across R0–R3.

## Robustness analyses

1. **Paired randomization tests.** Re-evaluate all 27 Step 11 primary contrasts for successful covert bits and exact payload survival using deterministic parent-paired sign-flip tests (20,000 Monte Carlo draws). Report Benjamini–Hochberg FDR-adjusted q-values separately for the two outcome families. P-values are supplementary to effect sizes and bootstrap intervals, not replacements for them.
2. **Leave-one-parent-out influence.** Recompute all nine R3 endpoint contrasts 48 times, each time removing one clean parent. Report effect ranges, sign stability, and maximum deviation from the full effect.
3. **Leave-one-cell-out sensitivity.** Repeat the nine R3 contrasts while omitting each of the eight source-model × domain × nominal-length cells. This tests whether the endpoint findings are dominated by one balanced cell.
4. **Alternative leakage outcomes.** Recompute R3 contrasts for effective bit accuracy and residual capacity, because exact message failure is not equivalent to zero leakage.
5. **Adaptive-window sensitivity.** Compare conditions using cumulative R1–R3 successful bits and exact survival counts rather than only the R3 endpoint.
6. **History conditioning.** Recompute R3 contrasts separately for rank-start and synonym-start trajectories.
7. **Descriptive heterogeneity.** Recompute R3 contrasts by source model, domain, and nominal length. These are sensitivity analyses, not confirmatory high-powered subgroup tests.
8. **Clean-burden cell audit.** Report intervention, semantic similarity, instruction-retention delta, and length-ratio outcomes for each defender × round × balanced cell.
9. **Frozen static-ablation reference.** Repackage the eight Step 8 ablation rows as a paper-ready reference table; no detector is retrained and no Step 8 metric is re-estimated.
10. **Cross-stage reference.** Preserve selected frozen Step 9 oracle mitigation and Step 11 R3 results in one compact consistency table for later paper synthesis.

## Primary claim-stability questions

- Does migration-enabled feedback remain stronger than frozen/no-feedback under both defender policies after sensitivity checks?
- Is migration clearly superior to fixed-channel adaptation, or only directionally so?
- Does fixed-channel adaptation remain weak/unstable relative to frozen/no-feedback?
- Does randomized rewriting show a sustained adaptive resilience advantage?
- Does exact suppression of rank coexist with residual recoverable information?

## Interpretation guardrails

- Step 12 is a robustness audit, not a new experiment and not an independent replication.
- Statistical significance is not used without effect sizes and uncertainty.
- Leave-one-out analyses diagnose influence; they do not multiply the sample size.
- Stratified results are descriptive sensitivity checks because subgroup n is small.
- Exact payload failure must never be called zero leakage when residual information remains.
- Four rounds do not establish equilibrium, emergence, tipping points, or long-run convergence.
- Results apply only to the tested source models, domains, lengths, attacker policy, rank/synonym adaptive families, and defender policies.

## Analysis-spec freeze

Step 12 is performed after Step 11 outcomes are known. Therefore this is not a preregistration. Before producing Step 12 results, the Step 12 configuration, code, metrics, and interpretation rules are frozen as a post-outcome robustness-analysis specification to prevent result-driven changes during the audit.
