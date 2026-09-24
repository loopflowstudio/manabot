# Project chapter report: Identity and Event Semantics — interval unknown (baseline)

Run id: `run_d1c60080ebd24e358ac2f499ed4bd935`

Reviewed September 24, 2026, at HEAD `7ad9a41f99cf367240ab380b197226a122398820`.

## Project synthesis

**Intended user improvement:** Give engine consumers reliable object identity and mutation semantics across replay, search, and presentation. The definition identifies these technical consumers; it does not explicitly promise a player-facing improvement.

**Current user experience:** Developers have exact object references, last-known information, a curated replacement/prevention pipeline, and explicit pre-priority stabilization. Focused historical regressions support these capabilities. Current execution was not verified, and some public mutation/event interfaces still expose storage identities without incarnation.

**Chapter conclusion:** The Project delivered substantial rules infrastructure. Its narrow implementation boundaries are documented, but the third KR promises broader public-boundary migration than the implementation supplies. The other two current outcome claims remain unverified in this review.

**Decisive facts:**

- **July 15:** `71b10bf` introduced exact identity and LKI; its retained commit report records 167 passing rules tests.
- **July 16:** `c1b4fad` introduced the proposed-event pipeline and records Rust, Python compatibility, and replacement-trace verification.
- **July 16:** `4f49ad5` introduced stabilization and records nine passing targeted SBA tests.
- **September 24:** Current source still exposes `Game::move_card(CardId, …)` and incarnation-free legacy events. The stabilization contract explicitly preserves these compatibility surfaces.

**Measurement fit:** Leave/reenter traces, replacement ordering, and stabilization checks directly test the intended mechanisms. They are credible safeguards, but neither completed Tasks nor their historical test summaries establish current correctness across every consumer. KR3 also exceeds the documented migration scope.

**Next-chapter implication:** Likely **retire** as an independent Project, preserving its mechanisms and regressions within consolidated Rules ownership. The future direction favors faithful Allies versus Lessons games and demonstrated training blockers; it does not justify a separate, comprehensive identity migration merely because an old KR remains unresolved. No disposition was applied.

## Objective evidence packet

**Coverage: incomplete.** All three supplied current KRs are included. Read-only `lf pm show --wave rules --project identity-and-event-semantics --json` returned identical wording and no drift. That view reports a cached September 24 snapshot, not complete provider revision history.

Repository-history searches and the retained proposed-event design recovered implementation history, but no additional authoritative historical KR wording. Chapter boundaries, establishment dates, removed claims, and revisions remain unavailable.

**Expected KR rows: 3 · Actual KR rows: 3**  
**Verdict totals: 0 holds · 1 does not hold · 2 unknown**

The exact ledger was observed at **2026-09-24T21:59:35.585349Z**. Earlier implementation dates are evidence dates, not inferred promise dates.

This review inspected source, contracts, tests, and dated commit reports. No tests, builds, experiments, writes, or planning mutations were performed. Historical passing-test statements below are attributed reports, not fresh executions.

## Definition

> Make object continuity and mutation exact across zones, replacements, triggers, and state-based actions so curated-card behavior remains correct under replay, search, and presentation.

**Verdict: unknown.**

**Evidence:** Current code implements the named mechanisms, with bounded historical regression evidence. The complete cross-consumer correctness outcome was not exercised here. Remaining compatibility identities limit the claim, but do not alone demonstrate an incorrect curated-card game, replay, or search result.

## KR ledger

### 1

> A leave-and-reenter vertical slice changes object incarnation, rejects stale ObjectRef values, and supplies correct last-known information in positive and negative trace tests.

**Status and active interval:** current · establishment unknown; exact wording first supported by the September 24 ledger → this review.

**Verdict: unknown.**

**Evidence:** The July 15 implementation report records 167 passing rules tests. Current [identity tests](../../../../managym/tests/rules/cr_400_identity.rs:29) explicitly cover:

- Preserved entity identity and advanced incarnation after departure/reentry.
- Rejection of stale, wrong-zone, and missing-entity references.
- Retention of departed controller, definition, and presentation information.
- Same-zone moves remaining identity-preserving no-ops.
- A Dragonfly Swarm death trigger retaining its original source through reentry.

The [lookup implementation](../../../../managym/src/flow/identity.rs:36) checks incarnation before resolving the permanent. Additional Man-o’-War traces cover a target leaving and reentering.

These are strong implementation and historical verification evidence. No current execution of the complete positive/negative trace set was obtained. No contrary current result was found; the unresolved verdict is verification uncertainty, not a demonstrated identity failure.

### 2

> Damage, life change, destruction, counters, and zone movement use a proposed-event to replacement/prevention to commit pipeline for the curated replacement-effects slice.

**Status and active interval:** current · establishment unknown; exact wording first supported by the September 24 ledger → this review.

**Verdict: unknown.**

**Evidence:** The July 16 commit report records full Rust tests, focused linting, 68 Python compatibility tests, and post-rebase replacement traces.

Current [pipeline code](../../../../managym/src/flow/proposed_event.rs:71) validates identities, collects replacements, applies them, and commits the surviving proposal. It declares all five mutation families. Destruction submits its resulting zone move through the same authority; player damage commits derived life loss once, while direct life changes use their own proposal.

[Replacement tests](../../../../managym/tests/rules/cr_614_replacement.rs:94) cover full prevention, observable ordering, repeated seeded traces, nonmatching effects, and entry properties. Separate unit tests reject stale proposals.

The [contract](../../../../docs/rules/proposed-events-v1.md:1) limits replacement definitions to prevention/doubling drivers and entering tapped/with counters. These are scenario-local definitions; this slice added no production cards. Affected-player replacement ordering and other general replacement semantics are expressly excluded. Those exclusions do not themselves falsify this deliberately curated KR.

Current pipeline execution and complete routing coverage were not freshly verified. The source and historical report support the architecture, but do not close the current behavioral conjunction.

### 3

> Trigger collection and state-based actions reach a deterministic fixpoint, and public rules boundaries no longer expose ambiguous bare storage IDs where incarnation matters.

**Status and active interval:** current · establishment unknown; exact wording first supported by the September 24 ledger → this review.

**Verdict: does not hold.**

**Evidence:** The first clause has substantial support. [Stabilization](../../../../managym/src/flow/stabilization.rs:16) repeatedly processes committed events, supported SBAs, and waiting triggers before priority. Current tests cover chained processing, simultaneous losses/departures, stale delayed references, and matching seeded state/event sequences. The July 16 report records nine passing targeted tests.

The second clause has a current source-level counterexample:

- Public [Game::move_card](../../../../managym/src/flow/zones.rs:226) accepts only `CardId` and a destination. It resolves whichever incarnation is current when called; the caller cannot supply an expected incarnation for rejection.
- Public [GameEvent](../../../../managym/src/flow/event.rs:33) variants retain bare card identities for zone movement, damage sources, and triggered abilities.
- The [compatibility contract](../../../../docs/rules/stabilization-v1.md:35) explicitly preserves storage-ID projections at external surfaces while moving touched internal continuations to exact references.

Observed September 24, this contradicts the KR’s unrestricted public-boundary clause. It does **not** establish a wrong-target incident through the revision-bound structured Command path.

## Shipped behavior

Dated repository history establishes delivery of exact-reference/LKI lookup, curated mutation proposals, deterministic replacement drivers, simultaneous SBA batches, and explicit pre-priority stabilization.

No chapter-start baseline was recovered, so these are dated contributions rather than a measured chapter-wide before/after outcome.

## Task and PR receipts

| Task | Contribution and observed state |
|---|---|
| [ETU-66](https://linear.app/loopflow/issue/ETU-66/implement-objectref-incarnation-and-lki-vertical-slice) | Provider-complete; identity/LKI contribution at `71b10bf`, July 15. |
| [ETU-65](https://linear.app/loopflow/issue/ETU-65/route-curated-mutations-through-proposed-events) | Provider-complete; proposed-event contribution at `c1b4fad`, July 16. |
| [ETU-64](https://linear.app/loopflow/issue/ETU-64/make-trigger-and-sba-stabilization-explicit) | Provider-complete; stabilization contribution at `4f49ad5`, July 16. |

The supplied runtime snapshot reports all three as ready/blocked on missing historical worktrees, with merged-PR successor recommendations and no active PR. This lifecycle residue does not establish unfinished implementation or authorize restarting them.

## Priority and ownership findings

The Tasks consistently specify narrow migrations and compatibility preservation. KR3’s broad public-boundary wording is therefore an objective/task scope mismatch.

No provider-open Task or evidenced active implementation belongs to this Project. No misplaced active Task was identified.

## Surprises

Deterministic replacement ordering is explicitly narrower than complete replacement-choice semantics. Likewise, exact internal continuations coexist intentionally with legacy external identities. Both limits are documented and should survive consolidation.

## Open work

No incomplete provider Task appears in the supplied or current Project view. Runtime discrepancies remain untouched.

## Evidence gaps

- Complete historical KR revisions and chapter dates.
- Current debug execution of the identity/LKI and replacement trace sets.
- A complete inventory of the curated mutation routes and their exercised coverage.
- Explicit reconciliation of KR3’s public-boundary promise with retained compatibility APIs.
- Current replay/search/presentation evidence establishing the definition’s complete outcome.

```json
{"1":"unknown","2":"unknown","3":"does not hold"}
```

