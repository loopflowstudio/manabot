# Project chapter report: Search and Study Runtime Prototype — interval unknown (baseline)

Run id: `run_cca1577b9d084e679fb439abf9647d42`  
Reviewed: **2026-09-24**  
Checkout: **`7ad9a41f99cf367240ab380b197226a122398820`**

## Project synthesis

**Intended user improvement**: Give Intelligence developers and Etude Study users dependable, efficient branches of authoritative game state. Search should retain independent alternatives; a player should explore a historical decision and return without changing its source.

**Current user experience**: Both consumers use compact full clone, with production integration and substantive July measurements. Search has measured interactive and saturated workloads; Study has repeated fork/apply/return and retained-sibling evidence. Current performance and the complete multi-seed stress claim remain unverified.

**Chapter conclusion**: The Project advanced beyond representation benchmarks into real consumers and made an evidence-backed decision to retain the simpler representation. Its five checked KRs overstate what the inspected evidence establishes today.

**Decisive facts**:

- July 17 teacher evidence records identical selected/reference outcomes, with selected throughput of **35.76 decisions/s interactive** and **4.70 decisions/s saturated**.
- July 18 Study evidence records **2,000 sequential cycles**, **512 retained branches**, and zero recorded source, sibling, or canonical-return mismatches.
- September 24 source inspection confirms that both consumers still clone the same native `Game` representation.
- The teacher receipt’s recorded source digest differs from today’s recomputed digest. Study’s recorded native binary is unavailable here, and its retained workload covers **seed 7 only**.

**Measurement fit**: These KRs credibly measure runtime integration, exactness, and cost. They do not establish enjoyable Study interaction, stronger bots, or full-game usability. Their broad stress and saturation language also exceeds the narrower Study evidence.

**Next-chapter implication**: Likely **rewrite**—retain the shared representation and exactness safeguards within Rules’ proposed consolidated Project. Prioritize demonstrated play/training bottlenecks. Unknown historical claims should not automatically become new implementation priorities.

## Objective evidence packet

**Coverage: incomplete.** All five supplied current claims are included. Read-only `lf pm show --wave rules --project search-and-study-runtime-prototype --json` returned the same definition and KRs; no current drift was found. This is the available PM snapshot, not recovered provider revision history.

Bounded inspection of repository history and preserved designs found earlier implementation and repair evidence, but no complete historical Project ledger. An older Task references a different “KR2” reproducible-harness clause without preserving its complete wording. Its full claim, interval, and relationship to the current Project remain unresolved; it cannot responsibly be reconstructed as an additional verbatim row.

**Expected KR rows: 5 · Actual KR rows: 5**  
**Verdict totals: 1 holds · 0 does not hold · 4 unknown**

The exact supplied ledger was observed at **2026-09-24T21:59:35.585349Z**. Establishment dates and chapter boundaries are unknown. July delivery dates are evidence dates, not inferred promise dates.

This review inspected source, contracts, raw retained measurements, test definitions, history, and supplied Task conditions. Cheap read-only checks recomputed the teacher source digest and confirmed all five Study latency arrays contain 2,000 samples. No engine tests, builds, experiments, or browser runs were performed. Missing build artifacts were treated as verification limits, not source defects. No code, PM, Task, charter, or archive mutations were made.

## Definition

> Put one exact branching runtime under real Intelligence search and interactive Study exploration. Select the simplest measured BranchDriver, integrate it into both consumers, and improve it from actual rollout, latency, memory, determinism, and viewer-safety observations rather than benchmark-only representation work.

**Verdict: unknown.**

**Evidence**: The architecture and historical consumer measurements strongly support this direction. The [representation decision](../../../../docs/benchmarks/search-branching-decision-v1.md:1) retains compact full clone after alternatives failed their registered adoption thresholds. The teacher and Study integrations remain present.

However, the complete current exactness/performance outcome is not established by the older source-bound measurements. The definition is excluded from KR totals.

## KR ledger

### 1. One selected BranchDriver runs an actual Intelligence teacher workload on the curated matchup with exact baseline-equivalent outcomes, no source or sibling mutation, and recorded whole-search throughput, p50/p95 decision latency, and peak RSS.

**Status and active interval**: Current · establishment unknown; exact claim first supported by the supplied September 24, 2026 ledger → review.

**Verdict: unknown.**

**Evidence**: The [RUL-2 consumer report](../../../../docs/benchmarks/selected-branchdriver-teacher-v1.md:1), delivered in `d7fe8ac` on **2026-07-17**, records:

- A seed-1197 exactness audit covering **421 decision roots**, with matching legal offers, Commands, visit/Q/value outputs, witnesses, viewer projections, event boundaries, RNG continuations, and terminal outcome.
- Zero reported mismatches, source mutations, viewer exposures, or fallbacks.
- Real selected-matchup measurements:

| Selected consumer cell | Decisions/s | p50 decision | p95 decision | Peak RSS |
|---|---:|---:|---:|---:|
| Interactive | 35.76 | 21.88 ms | 65.99 ms | 228.1 MiB |
| Saturated | 4.70 | 607.41 ms | 1,343.29 ms | 951.5 MiB |

The raw selected/reference measurements retain matching outcome hashes and execution counts. The exactness audit used one traversal and did not exercise a retained child edge; [focused tests](../../../../tests/sim/test_selected_branchdriver_teacher.py:54) cover world, child, and leaf execution and a separate differential case.

On **2026-09-24**, recomputing the receipt’s declared source closure produced:

- Recorded: `24d301266db8e48c5577026474e06bd648db7441c64e6e5051e186d1398428ec`
- Current: `3b8826e2c05766694746126b5805457ba765e7affdde79d85f6c633c138464ca`

The historical result therefore cannot certify today’s runtime without revalidation. No current mutation or equivalence failure was demonstrated; source drift alone is not a counterexample.

### 2. From a historical viewer-safe StudyIdentity, the runtime forks authoritative state, executes normal structured Commands, emits canonical events and offers, and returns in one action to an identical source state hash, event cursor, offer, and viewer projection.

**Status and active interval**: Current · establishment unknown; exact claim first supported September 24, 2026 → review.

**Verdict: unknown.**

**Evidence**: Study integration shipped through `1a0676a` and source-return hardening through `72a5757`, both **2026-07-17**.

The current [provider](../../../../etude/study_branch.py:238) resolves an authorized historical address and clones its retained authority root. Commands use native structured execution. [Return](../../../../etude/study_branch.py:173) checks the retained source digest, emits the recorded decision with its source identity, and consumes the branch.

The [July 18 retained measurement](../../../../experiments/rul-6-study-branch-v1.md:1) records:

- 2,000 sequential cycles and 512 retained returns.
- Zero canonical-return, source-digest, sibling-isolation, replay, or trace-event mismatches.
- 2,257 accepted structured Commands and zero Study fallbacks.
- Nine typed failure cases.

The [integration regression](../../../../tests/etude/test_study_branch.py:109) checks frame, offer, Command, presentation cursor, continuation, source digest, sibling preservation, and consuming return.

This is substantial bounded historical evidence. Current execution was not established, the receipt’s native binary is unavailable here, and the complete canonical-event emission clause was not independently demonstrated. These are evidence gaps, not observed failures.

### 3. The same branch representation supports retained search siblings and ephemeral Study branches, or a measured consumer comparison explicitly justifies two implementations and their shared exactness contract.

**Status and active interval**: Current · establishment unknown; exact claim first supported September 24, 2026 → review.

**Verdict: holds.**

**Evidence**: This bounded representation claim is supported by both historical consumer execution and current source tracing.

On **2026-09-24**:

- Search retains child environments in its [PUCT tree](../../../../manabot/sim/mcts.py:328). Its selected native fork uses `FullCloneDriver`.
- [FullCloneDriver](../../../../managym/src/search_state.rs:204) clones native `Game`.
- Study calls `root.clone_env()`, whose binding delegates to [Env::fork](../../../../managym/src/agent/env.rs:656), also cloning native `Game`.

These are distinct adapter entry points over the same representation, not two independently selected state representations.

July teacher measurements exercised retained child branches, while July Study evidence exercised 512 retained branches and repeated consuming returns. The [measured selection decision](../../../../docs/benchmarks/search-branching-decision-v1.md:1) explicitly rejected production adoption of clone-plus-undo and page-COW.

This verdict establishes shared representation support. It does not certify current performance or every isolation invariant.

### 4. Single-worker and saturated consumer measurements establish regression budgets for rollout throughput, interactive fork/apply/return latency, and peak RSS; the budgets run against the production integrations rather than a benchmark-only surrogate.

**Status and active interval**: Current · establishment unknown; exact claim first supported September 24, 2026 → review.

**Verdict: unknown.**

**Evidence**: The teacher’s [registered contract](../../../../experiments/contracts/rul-2-selected-branchdriver-teacher-v1.json:1) defines one-worker and four-worker cells, throughput floors, latency ceilings, and RSS limits. Its July result reports passing gates on the actual teacher integration.

The [Study contract](../../../../docs/benchmarks/study-branch-contract-v1.md:1) measures the real `GameSession → StudyForkProvider → Env` path. It sets:

- Fork/return p95 ≤1 ms; structured apply p95 ≤1.5 ms.
- End-to-end p95 ≤3 ms.
- Sequential throughput ≥500 cycles/s.
- Retained-sibling RSS delta ≤128 MiB.

The July 18 result reports **1.070 ms** end-to-end p95, **943.0 cycles/s**, and **54.0 MiB** retained-sibling RSS growth.

However, 512 retained branches in one process are not a demonstrated saturated concurrent Study workload. No inspected result closes that distinction or revalidates the complete budgets at current HEAD. The historical source and host restrictions are explicit in the contracts.

Missing saturation/current evidence leaves the conjunction unknown; it does not establish a budget violation.

### 5. Search and Study stress runs preserve deterministic replay, object incarnation, nested rollback, viewer-private information, and exact failure reproduction across multiple seeds with zero silent fallbacks to an unmeasured branch path.

**Status and active interval**: Current · establishment unknown; exact claim first supported September 24, 2026 → review.

**Verdict: unknown.**

**Evidence**: There are complementary instruments:

- [Native search-state tests](../../../../managym/tests/search_state_contract.rs:17) cover exact branches, multiple trace seeds, nested rollback, viewer projections, and stale references.
- Teacher measurement cells use seeds **1197, 1419, 1887, and 2197**. Focused selected-runtime tests cover rejected alternate execution paths and exact differential execution.
- July Study evidence records 2,257 viewer-private observation checks, zero opponent-hand exposures, zero object-reference mismatches, nine typed failures, and zero fallback.

But the production Study benchmark and its session fixture are fixed to **seed 7**. Its retained receipt does not establish multi-seed Study stress or nested branch/rollback coverage. The teacher intentionally records zero marks and rollbacks because it retains independent clones; separate native rollback tests cannot silently substitute for the full cross-consumer claim.

No actual privacy, replay, incarnation, or fallback failure was demonstrated in the reviewed scope. The complete stress denominator remains unproven.

## Shipped behavior

Dated history establishes these contributions:

- **July 16:** representation-neutral fork/rollback tests, reproducible benchmark provenance repairs, and a measured decision to retain full clone.
- **July 17:** selected-driver integration in the actual teacher; historical Study forks and source-bound consuming return.
- **July 18:** production Study lifecycle measurements, retained-sibling checks, and typed failure evidence.
- **September 24:** prepared possible-world materialization landed in `e856a90`.

Because the chapter interval is unknown, this is delivery history rather than a precise chapter-start comparison.

## Task and PR receipts

The supplied ledger contains **12 Tasks: 11 completed and one incomplete**.

| Task | Contribution evidence |
|---|---|
| ETU-46 | Receipt-source reproducibility repair, `dc6744d`, July 16. |
| ETU-47 | Study fork and exact source return, PRs #129/#134, July 17. |
| ETU-48 | Selected driver in the real teacher, `d7fe8ac`, July 17. |
| ETU-51–54 | Representation comparison, clone-plus-undo candidate, fork/rollback contracts, and benchmark harness. |
| ETU-49/50 | Semantic coverage and conformance infrastructure; adjacent supporting work. |
| ETU-56 | Viewer-relative possible-world/query provider. |
| ETU-57 | Study measurement and stress evidence, PR #150, July 18. |
| ETU-55 | Prepared provider implementation present in `e856a90`; broader scientific objective remains unsettled. |

These receipts establish contributions, not automatic KR completion.

## Priority and ownership findings

The observed work moved from alternative representations toward real consumer bottlenecks, consistent with the Project definition. No evidence justifies reopening page-COW or undo adoption merely because those implementations exist.

[ETU-55](https://linear.app/loopflow/issue/ETU-55/materialize-possible-world-batches-without-re-enumerating-support) combines a Rules-owned provider repair with a calibration experiment belonging to Intelligence’s scientific evaluation. Those outcomes require separate evidence and disposition. The User authorized provider-only shipping and deferred scientific closure; that distinction remains intact.

Future trained-challenger direction supports retaining runtime safeguards and measuring actual training needs. It is not retroactive evidence for these old claims.

## Surprises

- More sophisticated representations failed their adoption thresholds; full clone remained the measured choice.
- The teacher’s reported speed improvement came from its structured execution path while retaining the same representation.
- Two thousand Study cycles provide strong repetition at one position, but do not provide multi-seed coverage.
- The teacher receipt no longer matches current source; the Study receipt cannot be fully verified with the available native artifacts.

## Open work

**ETU-55** is incomplete in the PM snapshot. Its supplied runtime condition at **2026-09-24T21:59:39Z** is blocked by unsettled commits. The provider implementation is nevertheless present in reviewed HEAD. Scientific calibration completion is not established.

The other eleven Tasks are marked completed but retain missing-worktree/runtime residue. ETU-51 has an abandonment recommendation; ETU-57 retains a successor recommendation associated with an abandoned PR. These are lifecycle discrepancies, not authorization to restart implementation.

Nothing was changed by this review.

## Evidence gaps

- Complete historical KR wording, revisions, establishment dates, and chapter boundaries.
- Current source/binary-bound teacher exactness and performance evidence.
- Current Study execution proving the complete return and canonical-event clauses.
- A declared saturated Study consumer workload with measured budgets.
- Multi-seed production Study stress, including nested behavior and reproducible failures.
- Separate reconciliation of ETU-55’s shipped provider scope and deferred calibration objective.

```json
{"1":"unknown","2":"unknown","3":"holds","4":"unknown","5":"unknown"}
```

