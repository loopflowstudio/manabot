# Project chapter report: Belief-Aware Play — interval unknown (baseline)

Run id: `run_f64cb3504e094318954cce2d82934185`

Reviewed on September 24, 2026, at HEAD `7ad9a41f99cf367240ab380b197226a122398820` (committed at 21:55:58 UTC). This review made no repository or planning changes.

## Project synthesis

**Intended user improvement:** Help manabot developers establish whether explicit viewer-safe beliefs improve playing strength, and help Etude players understand how their beliefs change strategic advice.

**Current user experience:** The repository retains conditional-search results, a curated recommendation flip, exact-range tracking infrastructure, and newer learned-belief demonstrations. It does not establish that beliefs improve actual play at matched compute. The configured advice endpoint still uses fixtures rather than a live posterior, and its source-identity guard now rejects the historical positive fixtures.

**Chapter conclusion:** The Project produced useful instruments and bounded historical results, but its live product integration and primary strength question remain unresolved. Two checked KRs contain serving requirements contradicted by current code. Completed Tasks must not be read as completed scientific outcomes.

**Decisive facts:**

- **July 18:** INT-15 retained a post-hoc curated Has/ Lacks recommendation flip at 512 traversals and 16 worlds per scenario. Its evidence explicitly excludes strength and prospective-stability claims.
- **July 18–19 UTC:** INT-17 stopped after 5,876 wall seconds and three completed commitment updates. Its receipt records **zero curves** and **no scientific result**.
- **July 18 result:** INT-18 retained 720 arena games, but **no exact-range game started**. The requested belief-versus-uniform comparison remains absent.
- **September 24:** A fresh read-only calculation of the production advisor source digest disagreed with both historical fixture registrations. The endpoint’s explicit guard therefore returns `advisor_artifact_mismatch`.

**Measurement fit:** Conditional deltas and a curated flip demonstrate that beliefs can affect an instrument. They do not establish stronger play or useful live advice. The calibration and matched-compute arena KRs address those missing questions more directly, but their requested results were not found.

**Next-chapter implication:** Likely **rewrite** within the proposed consolidated Intelligence Project. Preserve the evidence and identity safeguards; select further belief work according to whether it advances repeatable training and a measured human challenger. The accepted new direction does not automatically carry forward advice integration, broad calibration, or planner migration.

## Objective evidence packet

**Coverage: incomplete.** All seven supplied KRs are included verbatim. The read-only `lf pm show --wave intelligence --project belief-aware-play --json` succeeded and returned the same seven claims; no current drift rows were found.

Repository history, earlier Task designs, experiment records, and Wave memory establish implementation and research history. They did not recover a complete dated sequence of this Project’s earlier, removed, or rewritten KR wording. Provider revision history and a chapter-start ledger remain missing.

**Expected KR rows: 7 · Actual KR rows: 7**  
**Verdict totals: 0 holds · 3 does not hold · 4 unknown**

For every current row, the exact wording is first supported here by the supplied ledger observed at **2026-09-24T21:59:35.585349Z**. Establishment dates are unknown. Earlier experiments are historical evidence, not proof that today’s exact wording was already a promise.

**Fresh verification boundary:** This checkout has no local `_managym` native extension. No endpoint test, game, training run, rebuild, or broad test sweep was performed. The fresh check executed the source-digest function and registered path list extracted from current `etude/advice.py`, without importing the engine or writing files:

| Artifact | Registered source digest | Current comparison |
|---|---|---|
| INT-12 advice fixture | `5ac62748…` | Mismatch |
| INT-15 flip fixture | `90146e3d…` | Mismatch |

Current source digest: `1f6062e9b9a7030837206de059b476ef63851b72a0b365f6fb3a85591a475041`.

The fixture file hashes still match their retained test assertions: `4a3fbeaa…` and `eef99429…`. Historical bytes remain preserved.

## Definition

The definition begins:

> Test whether explicit viewer-safe beliefs make manabot strategy stronger and more legible.

It then distinguishes the historical conditional substrate and curated flip from remaining live-advice, calibration, and exact-range arena work, while deferring supervised beliefs until prerequisites hold.

**Verdict: unknown.**

**Evidence:** The retained experiments demonstrate conditional strategic effects in bounded fixtures. They do not answer the matched-compute strength question or establish useful live advice. Current serving failures establish a concrete product gap, but do not demonstrate that explicit beliefs themselves cannot improve strength or legibility. The newer learned-belief implementation also makes the definition’s deferral language stale without completing its remaining outcome claims.

## KR ledger

### 1. Conditional determinized PUCT over paired inverse-CDF belief worlds returns identity-pinned, aligned strategy evidence for True, Has, Lacks, and paired Q/Not(Q) conditions with reproducible nontrivial policy, Q, and value deltas, frozen as replayable fixtures and served viewer-safe through the versioned advice contract (INT-12/INT-13; max policy delta 0.125, no argmax flip yet).

**Status and active interval:** current · establishment unknown; first exact observation September 24, 2026 → this review.

**Verdict: does not hold.**

**Evidence:** The retained INT-13 fixture contains the five-condition search evidence. The July INT-12 report records identical live-capability/Study response bytes and a maximum policy-probability delta of 0.125, while explicitly limiting interpretation to one engineering fixture. See [INT-12 report](../../../../experiments/int-12-belief-strategy-advisor.md:3).

The serving clause fails at the reviewed HEAD. The fresh digest check found a mismatch for the unchanged INT-12 fixture. [RegisteredAdvisor.verify_artifact](../../../../etude/advice.py:975) rejects that mismatch; the configured [fixture-serving path](../../../../etude/advice.py:1635) returns typed unavailability before serving the positive response. The retained [endpoint regression](../../../../tests/etude/test_belief_advisor.py:104) explicitly expects this behavior.

This is a September 24 source-backed counterexample to the conjunction, not a retraction of the frozen July search result.

### 2. A frozen, exactly replayable fixture shows changing only the typed condition (for example Has(Counterspell) versus Lacks(Counterspell)) flipping the advised top action under paired seeds at a declared offline budget, served through the advice-v1 comparison; a measured no-flip at roughly 64x the INT-12 budget with uniform-random leaves also closes this KR with the negative result retained.

**Status and active interval:** current · establishment unknown; first exact observation September 24, 2026 → this review.

**Verdict: does not hold.**

**Evidence:** The [INT-15 artifact](../../../../experiments/data/int-15-belief-recommendation-flip-v1.json) records generation at **2026-07-18T21:47:34.838269Z**, 512 traversals and 16 worlds per scenario, paired seeds, and post-hoc selection of `countered-wipe-four-wide-v1`, seed 197. It retains all nine candidate/seed cells and the selected positive result:

- `top_action_changed = 1.0`;
- top actions “Pass priority” and “Cast Pyroclasm”;
- visit margins 66 and 38;
- identical live-capability/Study public bytes.

The positive fixture bytes remain intact, but the fresh source-digest check proves its current registration mismatches. The [current endpoint regression](../../../../tests/etude/test_belief_advisor.py:156) expects `advisor_artifact_mismatch` with no strategy, evidence, or deltas.

The historical flip survives as offline evidence. The current serving requirement fails; a typed-unavailable response is neither a served flip nor the permitted measured negative result.

### 3. One full game through ./scripts/play requests advice mid-game whose belief receipt hashes the live exact-Bayes posterior tracked from the compatible-deal prior — no static authored payload on the path, fail-closed identity discipline preserved, a declared compute class, and the same identity resolving reproducibly through Study.

**Status and active interval:** current · establishment unknown; first exact observation September 24, 2026 → this review.

**Verdict: does not hold.**

**Evidence:** At the reviewed HEAD, [POST `/api/advice`](../../../../etude/server.py:3303) routes versioned requests directly to `request_versioned_fixture_advice`. The [frontend bootstrap](../../../../frontend/src/routes/+page.svelte:44) loads a pinned completed-match demonstration rather than advice for the live posterior.

Canonical `ed2` addresses, retained roots, and a posterior resolver exist. They do not complete this route. The [September 24 carry-forward plan](../../../../docs/plans/live-belief-advice.md:3) explicitly records the missing production connection. A further [current regression](../../../../tests/etude/test_live_advice.py:63) specifies rejection of the retained likelihood checkpoint because it declares the removed `max_conditions` field.

These are concrete contradictions of the configured-path requirement. No full-game result was inferred from the existence of its component infrastructure.

### 4. A committed repro script emits, for seeded games, posterior mass on the actual hidden opponent hand versus the compatible-deal prior across the decision sequence, with the action-likelihood model byte-locked to a real checkpoint and no RulesProviderGap remaining on the exercised path.

**Status and active interval:** current · establishment unknown; first exact observation September 24, 2026 → this review.

**Verdict: unknown.**

**Evidence:** The runner and frozen INT-17 contract exist. The [retained report](../../../../experiments/int-17-belief-calibration.md:1) specifies seed 0, both viewers, 132 Commands, 266 planned measurement points, a real checkpoint hash, and a provider receipt covering 62 commitments.

Its machine-readable failure receipt records an attempt from **July 18, 23:42:57 UTC to July 19, 01:20:53 UTC**, ending after three completed updates. It records zero emitted curves, zero observed provider gaps, and no retained scientific result. This establishes a historical systems failure, not poor belief calibration.

The prepared materialization repair landed September 24 as `e856a90`. The inspected current tree does not contain the prescribed committed v2 calibration contract and completed result. The newer learned-belief demo measures a different population and does not supply this exact-Bayes game-sequence comparison.

Because the failed attempt predates the supported exact-wording interval and the repaired path was not exercised here, it is retained as historical failure evidence rather than treated as proof that the current repaired runner cannot satisfy the claim.

### 5. A committed world-pinned arena run rates the exact-range player against a uniform-determinization control at matched compute alongside the frozen anchors, with ratings, the full payoff matrix, and paired-deal uncertainty — the measured answer to whether beliefs improve play.

**Status and active interval:** current · establishment unknown; first exact observation September 24, 2026 → this review.

**Verdict: unknown.**

**Evidence:** [INT-18](../../../../experiments/int-18-first-world-pinned-arena.md:1) retains a July 18 production result covering 720 games, 15 payoff cells, both seats, 24 paired deal blocks per cell, and 2,000 bootstrap fits.

That cohort contains the five code-only anchors and dPUCT-32. Its separate exact-range receipt explicitly records:

- `play_started: false`;
- `registered_likelihood_artifact_unresolved`;
- unavailable exact-range registration and semantic lifecycle;
- no substituted checkpoint, neutral likelihood, or authored belief.

This result cannot answer the requested comparison. No later qualifying retained run was found. Missing comparative evidence does not establish that beliefs help or hurt play.

### 6. Deferred until the flip, live-path, and calibration KRs hold: a supervised belief head maps viewer history to a normalized BeliefState over the same hypothesis domain, supervised only by access-controlled actual worlds, measured on log loss and calibration against the exact-Bayes baseline, with learned non-uniform beliefs mixed into policy/value training so serving is not out of distribution.

**Status and active interval:** current · establishment unknown; first exact observation September 24, 2026 → this review. The implementation direction has changed, but no exact replacement KR was recovered.

**Verdict: unknown.**

**Evidence:** September 24 HEAD includes a learned world scorer using typed public commitments, normalized whole-world weights, and authority-only supervision. The [learning demo](../../../../manabot/belief/learning_demo.py:265) separates inference inputs from hidden-world labels and reports held-out NLL, Brier, and calibration measurements.

[Wave memory](../../../../wave/intelligence/MEMORY.md) records a 160-episode population, a 128/32 split, one scripted opponent population, and 27 focused tests passing on September 24. That is a retained test claim, not a fresh execution here.

The comparison is against the physical-deal prior `p0`; it does not establish the full requested exact-Bayes baseline comparison and policy/value training-to-serving outcome. Live-path and calibration prerequisites remain unresolved. The stale deferral requires planning reconciliation, but does not itself prove failure of the learned model.

### 7. Background convergence gating no results KR: one running teacher consumes an immutable viewer-rooted PlanningProblem over managym authority with exact forks and semantic Commands, retiring the compatibility adapter.

**Status and active interval:** current · establishment unknown; first exact observation September 24, 2026 → this review.

**Verdict: unknown.**

**Evidence:** Current code contains authoritative world materialization, conditional priors, search branches, and semantic Commands. [CanonicalBeliefWorldSpace](../../../../manabot/sim/conditional_search.py:350) still adapts managym worlds to the existing conditional-PUCT interface; the historical fixture adapter also remains.

`PlanningProblem` appears as an architectural target in documentation. Searches found no qualifying current implementation and retained execution receipt proving the specified teacher consumes that interface and has retired the intended compatibility adapter.

The [results-first roadmap](../../../../docs/plans/results-first-roadmap.md:116) explicitly treats this convergence as background work that gates no result. Absence of a qualifying receipt is insufficient to assign a pass or infer a particular runtime failure.

## Shipped behavior

The inspected history establishes delivery of:

- Canonical exact-range belief infrastructure: `3efd25b`, July 18.
- Versioned advice and the curated flip: `5f80f52` and `23a4c8b`, July 18.
- Calibration failure evidence: `b359ae0`, July 19 UTC.
- The separate frozen-anchor/dPUCT arena result: `a9ffb45`, July 19 UTC.
- Prepared world materialization: `e856a90`, September 24.
- Belief-forming agent and learned-world demonstrations: `7ad9a41`, September 24.

No known chapter-start baseline supports a precise chapter-wide before/after comparison. These deliveries do not imply a live advice outcome or belief-strength result.

## Task and PR receipts

| Task | Observed state and contribution |
|---|---|
| [ETU-21](https://linear.app/loopflow/issue/ETU-21/resolve-live-advice-against-the-tracked-exact-bayes-posterior) | PM lists incomplete; supplied runtime is terminal/abandoned. User-directed abandonment is preserved. The missing worktree is not authorization to resume it. |
| [ETU-18](https://linear.app/loopflow/issue/ETU-18/refresh-the-int-15-runtime-digest-verifier-after-merge) | Completed as superseded without launch; intentionally produced no implementation. |
| [ETU-19](https://linear.app/loopflow/issue/ETU-19/commit-the-first-belief-calibration-curves) | PM completed; delivered retained systems evidence, not calibration curves. |
| [ETU-20](https://linear.app/loopflow/issue/ETU-20/freeze-the-first-belief-conditioned-recommendation-flip) | PM completed; historical curated positive fixture remains preserved. |
| [ETU-22](https://linear.app/loopflow/issue/ETU-22/run-determinized-puct-over-conditional-world-priors) | PM completed; conditional-search implementation and fixture contribution. |
| [ETU-23](https://linear.app/loopflow/issue/ETU-23/expose-a-versioned-belief-conditioned-strategy-advisor) | PM completed; versioned provider contract and historical measurements. |
| [ETU-24](https://linear.app/loopflow/issue/ETU-24/build-the-first-exact-range-belief-aware-player) | PM completed; infrastructure exists, but the requested arena result remains unsupported. |

Several completed Tasks retain `ready` runtime residue and missing historical worktrees. Those are lifecycle discrepancies, not new execution instructions.

[PR #179](https://github.com/loopflowstudio/etude/pull/179), associated with ETU-34 in the other Intelligence Project, supplied relevant learned-belief work now present at HEAD. Its merge establishes delivery, not closure of these KRs.

## Priority and ownership findings

The old portfolio’s primary strength question has no qualifying result. Its only PM-incomplete Task is the deliberately abandoned live-advice attempt.

Learned-belief work proceeded through ETU-34 despite this Project’s surviving blanket deferral. This is a planning mismatch to reconcile, not grounds to discard the implementation.

The new direction prioritizes complete Allies versus Lessons games and trained challengers. Historical work often uses the Interactive mirror or a curated Counterspell position; transfer to the selected human matchup must be demonstrated. Preserve ETU-21’s abandonment, ETU-31’s deferral, and the distinction between ETU-55’s provider repair and scientific calibration closure.

## Surprises

Identity discipline correctly prevents current source from impersonating historical advisors. It also means preserved positive fixtures and green rejection tests can coexist with unavailable product advice.

A substantial arena result excluded the Project’s central comparison. Similarly, a Task named for calibration curves completed with no curves. Both boundaries are explicit in the retained scientific records.

## Open work

No Task is established by the supplied ledger as actively executing this Project’s remaining outcomes. ETU-21 is abandoned despite its PM checkbox. No Task, PR, or runtime state was changed.

## Evidence gaps

- Complete historical KR revisions and chapter dates.
- Successful current-identity advice serving, followed by full-game live-posterior and fresh Study reproduction.
- Completed, preregistered calibration evidence after the provider repair.
- Exact-range versus uniform arena results at matched compute with frozen anchors and paired uncertainty.
- Learned-belief evaluation against the specified exact-Bayes baseline and demonstrated training/serving consistency.
- A running `PlanningProblem` teacher with an explicit adapter-retirement receipt.

Original-ordinal verdict mapping:

```json
{
  "1": "does not hold",
  "2": "does not hold",
  "3": "does not hold",
  "4": "unknown",
  "5": "unknown",
  "6": "unknown",
  "7": "unknown"
}
```

