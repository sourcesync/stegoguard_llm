# StegoGuard-LM Step 10 Adaptive Protocol

Protocol version: `step10_adaptive_v1`

## Scientific purpose

Step 10 is the first repeated attacker-defender experiment. It tests whether bounded defender feedback changes attacker behavior, whether covert throughput recovers after suppression, whether attack-family use migrates, and whether randomized rewriting changes the stability of adaptation. Static Steps 1-9 remain immutable historical evidence.

## Frozen population

Use the Step 8 balanced common population: 48 clean parents, six in each source-model x domain x nominal-length cell. Every selected parent has accepted Step 7 rank, synonym, and Unicode attacks. The primary adaptive competition uses rank and synonym. The implemented Unicode channel remains a structural-control result because always-on Unicode hygiene deterministically closes that implemented channel in Step 9.

## Frozen repeated design

- Rounds: R0, R1, R2, R3. No additional rounds may be added after outcomes are observed.
- Attacker conditions:
  1. `frozen_no_feedback`: keep the initial family and strategy 0 in every round.
  2. `fixed_channel_feedback`: family is fixed; after payload survival stay, after payload loss advance to the next available strategy cyclically.
  3. `channel_migration_feedback`: after payload survival stay; after payload loss switch attack family. When a family is revisited after another loss, advance its strategy cyclically.
- Defender policies:
  1. `deterministic_rewrite`
  2. `randomized_rewrite`
- Primary detector operating point: F1.
- Detector: ensemble reconstructed from the frozen Step 8 10 outer topic-disjoint full-fusion models. Each fold retains its frozen training-only F1 threshold. A new candidate is flagged when at least half of the fold policies flag it. The mean fold probability is retained for researcher analysis but is not shown to the attacker.

## Initial channel assignment

Within each of the eight balanced model x domain x length cells, sort the six parent IDs deterministically. The first three start with rank, the last three start with synonym. This yields 24 rank-start and 24 synonym-start parents per attacker-condition/defender-policy block.

## Candidate strategy budget

Maximum 10 strategies per parent and family.

- Rank strategy 0 is the frozen Step 7 attack. Strategies 1-9 keep the same prompt, clean parent, payload bitstring, token length, top-k, and attack family, while changing only a pre-frozen keyed attack-seed realization. Failed candidate generation remains a terminal candidate failure and is not retried by increasing the budget.
- Synonym strategy 0 is the frozen Step 7 attack. Additional strategies keep the same clean cover, payload bitstring, active slot set, and tokenizer constraints while applying deterministic pre-frozen zero/one mapping flips across active slots. Because a cover with k active slots has at most 2^k unique mapping masks, some parents legitimately have fewer than 10 distinct synonym strategies.

The logical payload root is fixed per parent. Payload bitstrings are fixed within an attack family. Channel migration may change family-specific payload length because rank capacity and synonym capacity are structurally different; this is recorded explicitly and must not be described as a same-capacity comparison.

## Bounded feedback shown to attacker

After each selected attack passes through the defender, the attacker receives only:

- `detected`: yes/no
- `intervention_route`: `pass_through`, `linguistic_rewrite`, or `unicode_sanitization`
- `exact_payload_recovery`: yes/no

The attacker does not receive raw detector probabilities, fold scores, feature vectors, thresholds, evaluator likelihoods, rewriter seeds, or clean references.

## Randomized defender

Randomized rewrite seeds are pre-frozen per trajectory x round using SHA-256 from the protocol version, trajectory ID, round ID, and master seed 101042. These seeds are stored in the trajectory manifest for reproducibility but never passed to the attacker policy.

## Clean controls

The same 48 clean parents are evaluated under both defender policies in every round. Their detector decision is constant across rounds, while randomized rewrite realizations use the corresponding pre-frozen round seed. Clean controls measure repeated false-positive intervention and visible-response utility burden.

## Primary outcomes

For stego trajectory row i at round r:

`successful_covert_bits = payload_bit_length` when the post-defense payload is recovered exactly, otherwise 0.

Report also effective bit accuracy and residual capacity so exact-message failure is not conflated with zero information leakage.

Primary system-level analyses include:

- successful covert throughput by round
- area under the four-round successful-throughput trajectory
- R3 minus R1 recovery after initial defense exposure
- strategy-change rate
- channel-transition counts and migration rate
- attack-family share by round
- intervention rate
- clean intervention burden
- semantic/instruction/fluency utility burden
- deterministic versus randomized defender differences
- initial-channel/path-conditioned trajectories

Uncertainty uses 2,000 parent-cluster bootstrap replicates.

## Frozen hypotheses

- H10.1: defensive intervention creates differential selection pressure among permitted strategies.
- H10.2: low-capacity lexical encoding increases in relative use when high-capacity rank encoding experiences persistent detection and rewriting.
- H10.3: feedback-enabled fixed-channel attackers recover more surviving covert throughput than frozen attackers if exploitable within-family adaptation exists.
- H10.4: randomized rewriting reduces the rate or stability of adaptive recovery relative to deterministic rewriting, even though Step 9 did not establish a universal one-shot randomization advantage.
- H10.5: channel migration preserves more covert throughput than fixed-channel adaptation if the defender creates heterogeneous survival pressure across channels.

A failed hypothesis remains a valid result. No migration, resilience, path dependence, or emergence claim may be made unless the repeated-round evidence supports it.

## Execution discipline

1. Verify Step 8 and Step 9 frozen artifacts before building Step 10 manifests.
2. Build and validate candidate and trajectory manifests.
3. Freeze this protocol and manifests before generating any Step 10 candidate.
4. Generate candidate attacks resumably and preserve terminal failures.
5. Extract candidate detection features and score them with the frozen Step 8 ensemble.
6. Execute rounds sequentially. Round r+1 may only be built after round r feedback is finalized and validated.
7. Preserve pass-through, rewrite, payload, utility, and bounded-feedback records separately.
8. Freeze Step 10 only after all four rounds, clean controls, summary, and validation pass.
