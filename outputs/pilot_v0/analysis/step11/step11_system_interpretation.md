# StegoGuard-LM Step 11 System-Level Interpretation

Protocol: `step11_system_analysis_v1`  
Input: final frozen Step 10 (`step10_adaptive_v1`)

## Central result

The frozen/no-feedback deterministic control remains flat at [10, 10, 10, 10] successful covert bits across R0-R3. Fixed-channel deterministic feedback changes strategies but yields only [10, 10, 12, 12] bits. In contrast, the channel-migration arm rises from 10 to 19 bits after the first feedback opportunity and remains at 19 bits at R3 under deterministic rewriting; randomized migration follows [10, 20, 20, 20].

This supports **mechanism-specific adaptive recovery**: the strongest recovery comes from changing attack family, not merely searching within a fixed family.

## Selection pressure and displacement

Rank exact-message delivery remains suppressed in the tested adaptive trajectories, while synonym encoding supplies the exact surviving messages. In the migration arm, synonym share rises from 50.0% at R0 to 62.5% at R3 under deterministic rewriting and to 62.5% under randomized rewriting. The defensible interpretation is **displacement-limited resilience**: StegoGuard-LM remains highly effective against one high-capacity family, yet its heterogeneous protection changes the relative attractiveness of another family.

## Micro/macro dynamics

The transition table should be read alongside aggregate channel shares. Approximately stable migration-arm synonym shares after R1 coexist with substantial bidirectional trajectory-level switching. This is evidence of persistent micro-level churn beneath a comparatively stable macro composition; it is **not** evidence of equilibrium or emergence.

## Randomization

The pre-frozen hypothesis that randomized rewriting would reduce adaptive recovery is not supported. Randomized and deterministic migration endpoints are close, and the parent-paired R3 comparison must be reported with its confidence interval rather than framed as a universal advantage for either policy.

## Residual information

Zero exact rank-message recovery is not zero leakage. Step 11 therefore preserves rank effective bit accuracy and residual-capacity summaries. The paper should distinguish exact-message delivery from partial recoverable information.

## Legitimate-user burden

Clean intervention remains 33.3% across all round/policy cells, while clean semantic similarity remains high. This is a multi-objective security-service result: the system must control covert throughput while also limiting unnecessary intervention and preserving visible service utility.

## Hypothesis assessment

- **H10.1: supported_within_protocol** — The defense creates differential selection pressure across attack families.
- **H10.2: supported_within_protocol** — Migration shifts relative use toward synonym under persistent rank suppression.
- **H10.3: weak_or_not_supported** — Fixed-channel feedback produces reliable recovery over frozen control.
- **H10.4: not_supported** — Randomized rewriting reduces migration-arm adaptive recovery relative to deterministic rewriting.
- **H10.5: directionally_supported_mixed_ci** — Allowing channel migration preserves more covert throughput than fixed-channel adaptation.


## Guardrails for paper writing

- Do not claim StegoGuard-LM eliminates LLM steganography.
- Do not claim zero rank leakage from zero exact rank recovery.
- Do not call the four-round behavior equilibrium, emergence, convergence, or a tipping point.
- Do not generalize the rank/synonym migration result to arbitrary covert channels.
- Treat model/domain/length strata as descriptive robustness slices; the 6-parent cells are small.
- Preserve the negative randomization result.
- Keep AI security as the novelty anchor; use complex-systems and service-systems theory to interpret repeated interaction, displacement, resilience, legitimate-user burden, and unintended consequences.
