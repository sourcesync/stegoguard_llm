# StegoGuard-LM Step 11 Protocol — System-Level Adaptive Analysis

Protocol: `step11_system_analysis_v1`

## Purpose

Step 11 converts the already-frozen Step 10 adaptive trajectories into a publication-ready system-level analysis. It is deliberately **model-free and non-generative**: it does not create new attacks, add rounds, retrain the detector, change thresholds, alter mitigation, or load source/evaluator LLMs.

The analysis asks whether the frozen adaptive evidence supports feedback-driven recovery, whether recovery occurs through within-family search or cross-family migration, whether randomized rewriting changes adaptive resilience, how exact-message suppression relates to residual information, and what legitimate-user burden accompanies the security outcome.

## Frozen input

Step 11 may run only after the final Step 10 freeze verifies successfully under `step10_adaptive_v1`. The required input consists of 48 parents, 288 trajectories, 1,152 stego round rows, and 384 clean-control round rows across R0-R3.

## Primary contrasts

All primary inferential comparisons are parent-paired and use 2,000 parent bootstrap replicates.

For R1, R2, and R3, separately under deterministic and randomized defense:

1. fixed-channel feedback minus frozen/no-feedback;
2. channel-migration feedback minus frozen/no-feedback;
3. channel-migration feedback minus fixed-channel feedback.

For R1, R2, and R3, separately for each attacker condition:

4. randomized rewrite minus deterministic rewrite.

This yields 27 pre-specified pairwise rows. Each row reports paired differences and 95% bootstrap intervals for successful covert bits and exact payload survival.

## System-level analyses

Step 11 additionally reports:

- round-by-condition security metrics;
- cumulative/trajectory throughput and R3-R0/R3-R1 recovery;
- attack-family exact survival and residual information;
- channel-transition counts and gross switching;
- migration-arm channel shares;
- rank residual capacity despite zero exact delivery;
- initial-channel/history-conditioned behavior;
- R3 descriptive model × domain × nominal-length strata;
- clean intervention and semantic/task-utility burden;
- explicit H10.1-H10.5 assessment.

## Resilience interpretation

Step 11 uses a **resilience profile**, not a single invented resilience score. The profile jointly reports exact-message suppression, surviving covert throughput, residual information, adaptation recovery, channel migration, clean intervention, and clean visible-response utility.

## Interpretation guardrails

- Feedback-enabled arms must be compared against the frozen/no-feedback control before claiming adaptation effects.
- Exact payload failure is not equivalent to zero information leakage.
- Four rounds do not establish equilibrium, emergence, tipping points, or long-run convergence.
- Initial-channel differences are described as history conditioning within this bounded protocol, not universal path dependence.
- Randomization is not called superior unless paired evidence supports a sustained advantage.
- Channel migration is limited to the tested rank/synonym families and must not be generalized to arbitrary covert channels.
- Rank and synonym have different payload capacities; exact survival, throughput, and residual information must be interpreted jointly.

## Analysis-spec freeze

Step 10 outcomes are already known, so this Step 11 specification is **not a preregistration**. It is a post-outcome analysis-specification freeze that prevents further result-driven changes to contrasts, metrics, bootstrap procedure, and interpretation rules before Step 11 results are produced.
