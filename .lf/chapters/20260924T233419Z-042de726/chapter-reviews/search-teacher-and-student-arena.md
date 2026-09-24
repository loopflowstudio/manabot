# Project chapter report: Search Teacher and Student Arena — interval unknown (baseline)

Run id: `run_713009e9c31a46a1821706e5b5428664`

Reviewed September 24, 2026, at HEAD `7ad9a41f99cf367240ab380b197226a122398820`.

## Project synthesis

**Intended user improvement:** Give agent developers a reproducible search → data → student → arena loop that supports measured training decisions and supplies attributable evidence to Study. The definition emphasizes research machinery; it does not explicitly promise a stronger human opponent.

**Current user experience:** Developers can inspect substantial retained teacher/student experiments and a verified production arena rating. They cannot yet rely on a completed production multi-seed learning iteration or ordinary Study sessions receiving historical policy/search evidence. Current execution compatibility of the older training artifacts remains unverified.

**Chapter conclusion:** The Project delivered useful experimental machinery and bounded results, including negative results that changed the next training decision. Its principal production learning comparison remains unfinished. This is meaningful research progress, but it does not establish repeatable improvement in a currently playable trained challenger.

**Decisive facts:**

- The July 17 INT-4 smoke retained 507 labels, four matched student arms, a 175-decision replay audit, and a Study artifact. On September 24, all 13 retained payload files matched their recorded sizes and hashes.
- INT-18 retained 720 games across every cell of its six-player cohort. September 24 verification authenticated that result, its connectivity, and 2,000 bootstrap replicates without generating or replaying games.
- The August 19 ETU-31 review explicitly recorded no production result or retained verification tree. The User deferred that experiment on September 24.
- At reviewed HEAD, normal Game sessions still install `UnavailableHistoricalStudyEvidenceProvider`. An exported Study artifact therefore does not establish delivery through the ordinary product path.

**Measurement fit:** Identity, replay, matched-data ablations, complete arena matrices, and declared compute are credible evidence for trustworthy experimentation. They do not independently prove stronger play. The retained results demonstrate why that distinction matters: improved value calibration weakened complete players, and learned smoke priors failed their continuation gates.

**Next-chapter implication:** Likely **rewrite**. Preserve the experimental evidence and its integrity requirements within the proposed repeatable training and measured-strength objectives. Keep ETU-31 explicitly deferred until its prerequisites and budget are reconciled. Historical Study evidence should remain supporting work unless it directly serves the accepted trained-challenger outcome. No disposition was applied.

## Objective evidence packet

**Coverage: incomplete.** Every supplied current KR is reviewed. The read-only PM view returned the same seven claims, with no current wording drift. Its snapshot timestamp was September 24, 2026, approximately 21:36 UTC; it is not provider revision history.

**Expected KR rows: 7 · Actual KR rows: 7**

**Verdict totals: 1 holds · 0 does not hold · 6 unknown**

The exact supplied ledger was observed at **2026-09-24T21:59:35.585349Z**. Establishment dates remain unknown.

Repository history establishes a July 16 teacher/distillation predecessor and a July 18 planning reference to the first production-rating KR. Those records do not recover a complete historical Project ledger. A shortened KR quotation in the INT-18 design is insufficient to establish a separate, exact former wording. Missing sources are the chapter-start record and complete Project/KR revisions, including removed claims.

**Fresh bounded checks:**

- INT-18’s documented `--verify-only` command passed: **720 games, 76,991 decisions, 15 cells, 24 deal blocks, one component, 2,000 bootstrap replicates, zero recorded integrity failures**. It reported `no_generation=true` and `no_replay=true`.
- A standard-library byte audit authenticated all **13** files in the retained INT-4 payload. This verifies retention, not model execution or replay.
- The INT-4 verification command stopped at import with `ModuleNotFoundError: No module named 'numpy'` in this checkout’s existing environment. No dependency installation or repair followed. This limits fresh verification; it is not a rules or training failure.

No experiments, broad test sweeps, code edits, archival writes, or planning mutations occurred.

## Definition

> Run a real search teacher, dataset, student, and arena loop on the selected matchup. Improve the working system through measured iterations: establish a runnable baseline, replace flat controls with authoritative visit-based search, train practical students, compare matched compute, and expose honest evidence to Study. As of 2026-07-19 the loop, fail-closed production harness, and world-pinned arena all exist; INT-18 committed the first frozen-cohort production rating, while the exact-range-versus-uniform comparison remains typed evidence_wait and the production visit-teacher iteration remains open. Standing evidence: INT-7 value heads improved calibration while weakening complete players; INT-8 killed one-seed learned priors inside PUCT; learned components re-enter only through arena admission.

**Verdict: unknown.**

**Evidence:** Retained experiments substantiate the baseline, production-rating result, and negative research findings. The definition itself acknowledges the unfinished production iteration. Current ordinary Study delivery and current execution of the historical learning loop remain unproven. The evidence supports substantial parts of the definition, not its complete present-tense outcome.

## KR ledger

### 1. One documented command freezes replayable trajectories from actual managym games, trains matched policy-only and policy/value students, and plays a seat-balanced arena with pinned dataset, engine, model, seed, latency, throughput, and compute identities.

**Status and active interval:** current · exact wording first observed September 24, 2026 → September 24 review; establishment unknown.

**Verdict: unknown.**

**Evidence:** The [INT-4 smoke report](../../../../experiments/int-4-visit-teacher-smoke.md) documents the command and bounded execution. Its retained manifest dates execution to July 17, 18:34:37–18:34:50 UTC. Four games produced 507 labels; four seed-197 chosen/visit × policy-only/policy-value arms trained; the arena completed two-game cells. A separate source-game audit replayed 175 decisions.

The September 24 byte audit confirms retention of the shards, checkpoints, audit, manifest, and Study artifact. However, current executable verification stopped before loading them because NumPy was absent. The runner also explicitly validates runtime fingerprints before execution. Historical success and preserved bytes do not establish that the documented loop runs under current source and model interfaces.

No statistical strength threshold is added to this KR; its unresolved boundary is current reproducible execution.

### 2. An authoritative MCTS or PUCT teacher runs the selected matchup at declared budgets, emits legal root visit distributions and values under the acting viewer boundary, and replays every sampled trajectory exactly.

**Status and active interval:** current · exact wording first observed September 24, 2026 → September 24 review; establishment unknown.

**Verdict: unknown.**

**Evidence:** [Teacher-1](../../../../experiments/w2-234-teacher1.md) introduced adaptive PUCT, root visits and values, declared determinization budgets, and authoritative-root isolation. Its original decision shards were explicitly not replayable semantic trajectories.

The later July 17 INT-4 evidence closes that earlier limitation for its sampled audit: all 175 decisions and the sampled search root reportedly reproduced, including aggregate/per-world visits, values, selected action, and search costs, with zero mismatches. Its report also records a repaired hidden-order determinization defect.

Those are substantive bounded historical results. This review authenticated retained bytes but could not freshly execute teacher/replay verification at the scoped HEAD. The complete current conjunction therefore remains unknown. The earlier repaired defect is not treated as a current failure or assigned to an unestablished KR interval.

### 3. A practical student trains from visit and value targets and then plays the actual matchup; matched-data ablations separate visit-distribution versus chosen-action supervision and policy-only versus policy/value learning.

**Status and active interval:** current · exact wording first observed September 24, 2026 → September 24 review; establishment unknown.

**Verdict: unknown.**

**Evidence:** INT-4 retained all four matched seed-197 arms and executed student gameplay. [INT-7](../../../../experiments/int-7-value-target-comparison.md), committed July 18, extended value-target comparisons to twelve checkpoints across three initialization seeds, with 544 games and 81,449 replayed decisions reported.

INT-7 found policy-only strongest within its diagnostic matrix; better value calibration did not improve complete players. [INT-8](../../../../experiments/int-8-student-signal-guidance.md) separately compared retained chosen-action and visit-policy guidance and rejected both under its bounded continuation rule.

These support historical training and ablation capability. They do not resolve current checkpoint compatibility or current gameplay execution. The September 24 architecture reconciliation explicitly records changed checkpoint interfaces and historical loader guards. No fresh student-loading or gameplay proof was obtained here.

The one-corpus limitation constrains research generalization; it is not itself a failure of this narrower implementation KR.

### 4. The world-pinned arena commits its first production rating run over the frozen anchors and the dPUCT-32 challenger, with ratings, the full payoff matrix, connectivity, and paired-deal uncertainty retained; development pairwise matches remain explicitly non-admission.

**Status and active interval:** current · full wording first observed September 24, 2026 → September 24 review. The core rating requirement is referenced in July 18 planning history; exact full-wording establishment remains unknown.

**Verdict: holds.**

**Evidence:** [INT-18](../../../../experiments/int-18-first-world-pinned-arena.md) records the July 18 result, committed July 19 at `a9ffb45`. The retained production result covers five frozen anchors plus dPUCT-32: **720 games, all 15 cells, 24 paired deals per cell, both seat assignments, one connected component, and 2,000 successful bootstrap fits**.

September 24’s generation-free verifier authenticated the complete retained envelope at manifest identity:

`af0c3f56745ba4f60e5e3f612787b11c65d6b125917ec1e59b1835e113765b2b`

This verdict concerns the explicitly requested first retained result, not ongoing strength or present-runtime regeneration. dPUCT-32 rated 1333, below flat-MC-64’s 1513, and remained `rated_not_promotion_eligible` because no same-compute incumbent existed. Development pairwise results remain non-admission evidence.

### 5. The fail-closed production visit-teacher harness runs one complete production iteration with multi-seed students after the absent frozen Teacher-0 control bytes are recovered or re-frozen under a versioned contract note; the iteration either improves the admitted agent on the declared strength and competency gates or records an honest failure that determines the next build.

**Status and active interval:** current · exact wording first observed September 24, 2026 → September 24 review; establishment unknown.

**Verdict: unknown.**

**Evidence:** The [production report](../../../../experiments/int-4-visit-teacher-production.md) states July 17: “fail-closed harness ready; production not run,” with both required Teacher-0 checkpoint digests unresolved.

The [ETU-31 review](/Users/jack/src/etude.run-the-production-visit-teacher/scratch/review-slice.md), dated August 19, records an additive contract/coordinator but no production result, retained manifest, or verify-only result. Its preflight attempts stopped before Python execution. Those infrastructure failures are not the completed production comparison or declared experimental negative result required by this KR.

September 24 direction explicitly defers ETU-31. No qualifying result was found. With no known deadline or completed in-scope production attempt, missing evidence is **unknown**, not a fabricated experimental failure.

### 6. Across multiple seeds, the search teacher is compared with policy-only, scripted, and frozen opponents on legality, competencies, seat-balanced strength, calibration, p50/p95 decision latency, rollout throughput, and label cost, producing an explicit continue, revise, or kill decision.

**Status and active interval:** current · exact wording first observed September 24, 2026 → September 24 review; establishment unknown.

**Verdict: unknown.**

**Evidence:** INT-4 supplies bounded teacher economics and integrity; INT-7 supplies three initialization seeds over one inherited corpus; INT-8 supplies a one-seed guidance decision; INT-18 supplies a complete production rating for a code-only cohort.

These experiments answer different questions. They cannot be combined into the required teacher comparison across the specified opponents, seeds, and full metric set. The Teacher-1 admission pilot remains documented as preregistered but unrun, and ETU-31 owns the missing production comparison.

The record contains useful continue/kill decisions, but none inspected satisfies this complete conjunction.

### 7. One historical Study position receives viewer-safe DecisionEvidence from the running policy and search systems, with played Command, policy mass, visits, value, sampled-world robustness, uncertainty, and unavailable fields kept distinct.

**Status and active interval:** current · exact wording first observed September 24, 2026 → September 24 review; establishment unknown.

**Verdict: unknown.**

**Evidence:** The retained July 17 Study artifact contains distinct played Command, policy probabilities, visits, search values, sampled-world robustness, and uncertainty for one audit position. Its provenance binds the checkpoint, search budget, replay digest, and producer; September 24’s payload audit authenticated its bytes. The [exporter](../../../../manabot/sim/study_evidence.py:1) projects search evidence into the viewer-safe contract.

The ordinary product path remains materially different. [Normal session construction](../../../../etude/server.py:2413) creates a default session whose historical provider is unavailable; [that provider](../../../../etude/study_runtime.py:44) always raises an unavailable error. The ETU-29 directive also preserves a July 18 checkpoint/root-handoff gap.

The historical export proves an adapter result. It does not establish that the running Study path delivers that evidence now. Conversely, default unavailability alone does not disprove the narrower existential historical-export interpretation. The wording and execution evidence do not support a stronger verdict.

## Shipped behavior

Repository history establishes delivery of the Teacher-0 foundation, authoritative PUCT, visit-supervised training, replay auditing, Study export, arena instrumentation, and retained INT-7/8/18 results. September 24 HEAD also contains ETU-34’s belief-forming agent work.

No chapter-start baseline was recovered, so these deliveries cannot be presented as a precise chapter-over-chapter improvement.

## Task and PR receipts

- [ETU-30](https://linear.app/loopflow/issue/ETU-30/teacher0-data-student-and-vector-stepping-foundation): provider-complete; foundation attributed to PRs #110, #111, #114, and #115.
- [ETU-38](https://linear.app/loopflow/issue/ETU-38/run-the-first-visit-based-teacher-and-student-arena-iteration): provider-complete; INT-4 engineering execution is retained. Completion does not imply production KR closure.
- [ETU-32](https://linear.app/loopflow/issue/ETU-32/commit-the-first-world-pinned-arena-rating-run): provider-complete; independently authenticated INT-18 result supports KR4.
- [ETU-35](https://linear.app/loopflow/issue/ETU-35/measure-which-student-signal-improves-the-next-puct-teacher) and [ETU-36](https://linear.app/loopflow/issue/ETU-36/compare-value-targets-inside-the-visit-trained-player): provider-complete; bounded negative/continuation results retained.
- [ETU-29](https://linear.app/loopflow/issue/ETU-29/materialize-viewer-safe-policy-and-search-evidence-for-study): provider-complete, but its preserved directive and current default provider leave the delivery boundary unresolved.
- [ETU-34](https://linear.app/loopflow/issue/ETU-34/map-search-and-learning-architecture-and-laws-of-physics): PM lists incomplete; runtime says done; reviewed HEAD contains merged PR #179. Reconcile lifecycle and actual scope rather than restart it.

## Priority and ownership findings

ETU-31 directly owns the missing production comparison and remains relevant, but the User’s explicit deferral controls its scheduling.

ETU-34 delivered broader agent and belief-learning behavior than its architecture-mapping title suggests. Its ownership description needs reconciliation; its merge does not complete ETU-31.

The old Project’s Study objective has no demonstrated ordinary delivery path despite completed Task status. Under the accepted future direction, that gap should compete against repeatable training and human challenger evidence rather than automatically carry.

## Surprises

- Better value calibration produced weaker complete players.
- Learned smoke priors added cost without earning continuation.
- A fully retained production rating still could not justify promotion.
- Historical artifacts remain valuable while current model interfaces reject their reuse. Preservation and present compatibility are separate requirements.

## Open work

[ETU-31](https://linear.app/loopflow/issue/ETU-31/run-the-production-visit-teacher-multi-seed-comparison) remains provider-open, runtime-ready, and locally dirty with no published active PR in the supplied ledger. The User deferred execution on September 24. Its worktree was inspected read-only and left untouched.

ETU-34’s provider/runtime discrepancy remains reconciliation work. Completed historical Tasks with ready runtime residue were not treated as authorization to resume them.

## Evidence gaps

- Complete historical KR revisions and chapter dates.
- Current-runtime execution and replay verification for the documented training loop.
- Current compatible checkpoints with explicit identities and exercised gameplay.
- The complete ETU-31 production comparison, or its admissible bounded negative result.
- One available historical Study response through the real configured policy/search handoff.
- Human-match and repeatable improvement evidence for the proposed next chapter; neither is retroactively required by these old claims.

```json
{"1":"unknown","2":"unknown","3":"unknown","4":"holds","5":"unknown","6":"unknown","7":"unknown"}
```

