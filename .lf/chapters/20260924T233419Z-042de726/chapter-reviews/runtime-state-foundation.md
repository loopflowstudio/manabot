# Project chapter report: Runtime State Foundation — interval unknown (baseline)

Run id: `run_431b38e07b804a38b323602d7415144d`

Reviewed September 24, 2026, at HEAD `7ad9a41f99cf367240ab380b197226a122398820`.

## Project synthesis

**Intended user improvement:** Not explicit at the player level. The definition names an engine foundation: shared immutable content, stable mutable-state identity, and deterministic witnesses. Its direct beneficiaries are managym maintainers and manabot developers implementing replay and isolated search.

**Current user experience:** Developers have a concrete shared-content implementation, typed definition identities, deterministic hashing, and focused regression tests. A retained allocation experiment demonstrates that enlarging immutable definitions did not increase clone allocations on its measured workload. This review does not establish the complete current claims across every environment and branch.

**Chapter conclusion:** The Project delivered substantive infrastructure and improved the discipline of its evidence. Its historical measurements support bounded accomplishments; they do not certify universal current behavior after subsequent engine changes.

**Decisive facts:**

- **July 15–16:** Commits `61b7e15` and `e26103b` delivered shared definitions and deterministic state hashing.
- **July 16, 01:47:23 UTC:** The retained allocation receipt records an additional **4,364,032 immutable definition bytes** with **zero additional allocations or allocated bytes** across each 1,024-clone workload.
- **July 16:** Non-contract before/after performance claims were removed; the retained report explicitly states that no valid performance comparison exists.
- **September 24:** Current source preserves the sharing and hashing mechanisms, but differs from the allocation receipt’s measured revision. No fresh engine execution was performed.

**Measurement fit:** Allocation ownership and deterministic-trace checks directly measure this Project’s intended mechanisms. They do not establish faster training, stronger agents, or a better human game. Consumer performance is appropriately assigned elsewhere.

**Next-chapter implication:** Likely **retire as an independent Project**, preserving its contracts and evidence as safeguards within the consolidated Rules Project. All seven provider Tasks are complete, consumer performance already has another owner, and the accepted direction prioritizes faithful Allies versus Lessons play and demonstrated training blockers. Retirement would not mean the universal KRs were independently certified.

## Objective evidence packet

**Coverage: incomplete.** Both exact current KRs are included. The read-only `lf pm show --wave rules --project runtime-state-foundation --json` returned identical wording, with no current drift.

A historical scan preserved in `9d45266:scratch/garden-scan.md`, committed **July 16, 2026, 05:52:07 UTC**, reports **three** Runtime State Foundation KRs, with a current-source regression gate still open and W2-207 assigned to it. This establishes missing historical scope. It does not preserve the third KR verbatim, so no reconstructed wording is substituted for that promise.

Missing sources are the chapter-start record and complete provider KR revisions, including the former third KR and any earlier wording of the surviving two.

**Expected current KR rows: 2 · Actual exact KR rows: 2**  
**Historical coverage: at least one additional former KR remains unenumerated.**  
**Verdict totals: 0 holds · 0 does not hold · 2 unknown**

The supplied ledger was observed at **2026-09-24T21:59:35.585349Z**. Establishment dates remain unknown. July implementation dates are delivery evidence, not inferred dates when today’s exact wording became a promise.

Evidence comprises source, test definitions, retained JSON/Markdown measurements, repository history, and supplied Task conditions. No tests, experiments, builds, edits, PM writes, or lifecycle changes occurred.

## Definition

> Provide the compact authoritative state foundation used by every playable world and fork: immutable versioned content is shared, mutable match facts have stable identity, and equivalent seeded executions have deterministic state witnesses. Consumer performance now belongs to the running Search and Study Runtime Prototype.

**Verdict: unknown.**

**Evidence:** Current source and retained measurements substantiate the foundation’s architecture and bounded historical behavior. They do not establish the complete “every playable world and fork” denominator at the reviewed revision. No current isolation or determinism failure was demonstrated. The consumer-performance ownership statement matches the Project’s deliberately narrow evidence boundary.

This verdict is excluded from KR totals.

## KR ledger

### 1. Every environment and branch shares one versioned ContentPack while mutable-state cloning duplicates no immutable card definitions, with independently reset roots and siblings remaining isolated.

**Status and active interval:** Current · establishment unknown; exact wording first supported by the supplied September 24, 2026 ledger → this review.

**Verdict: unknown.**

**Evidence:** The [W2-208 receipt](../../../../experiments/w2-208-content-pack-clone-allocations.md) records a successful measurement on **July 16, 01:47:23 UTC**, at source revision `a53d7cd5927396c597b2e4e45642b8496496ff66`.

Its controlled workload retained 4,096 content-pack references without allocating, then measured 1,024 clones of each mutable representation:

| Boundary | Baseline allocated bytes | Expanded allocated bytes |
|---|---:|---:|
| `GameState` clones | 51,653,632 | 51,653,632 |
| Exact `Game` clones | 51,719,168 | 51,719,168 |

The expanded pack added 256 unused definitions and over 4 MiB of serialized definition data. Allocation counts and bytes had exactly zero expanded-minus-baseline deltas. This directly supports allocation independence for that fixture and revision.

Current [ContentPack tests](../../../../managym/tests/content_pack_tests.rs) check shared pack and definition pointers across independent matches and clones, along with sibling isolation. The [environment contract](../../../../managym/src/agent/env.rs:1145) also checks independently reset environments, root/sibling forks, and retained rollout slots. Current [game branching](../../../../managym/src/flow/game.rs:152) retains the content reference when constructing page-COW branches.

These are strong mechanisms and targeted test definitions. However, the historical measurement predates substantial engine changes, including event storage and additional branch consumers. The report itself requires regeneration when measured source changes. Neither a current allocation result nor complete coverage of today’s branch paths was obtained.

The scenario mutation seam intentionally copies a definition on mutation; that is distinct from duplicating immutable definitions during ordinary cloning and is not treated as a counterexample.

### 2. Stable CardDefId-based definitions and versioned deterministic state witnesses reproduce equivalent seeded traces without allocation-address, Debug-format, or card-name identity at the state boundary.

**Status and active interval:** Current · establishment unknown; exact wording first supported by the supplied September 24, 2026 ledger → this review.

**Verdict: unknown.**

**Evidence:** Commit `e26103b`, dated **July 16, 00:23:22 UTC**, delivered the versioned hashing contract and identity checks.

Current [hash implementation](../../../../managym/src/state/hash.rs) represents physical cards using typed definition IDs, incorporates content identity, and explicitly canonicalizes mutable state. The [contract documentation](../../../../docs/architecture/match-state-hash-v1.md) specifies ordering, excluded allocator/formatting inputs, RNG non-mutation, and versioning requirements.

The [focused tests](../../../../managym/tests/match_state_hash_tests.rs) cover independently constructed seeded traces, distinct allocations, clone equality, mutation sensitivity, content changes, RNG preservation, compatibility-only card renaming, typed stack identity, and a source guard against pointer, name, registry-key, or Debug-format dispatch. Further [search-state tests](../../../../managym/tests/search_state_contract.rs) compare equivalent roots with independently allocated content packs.

These tests closely match the claim, but were not executed here. Source inspection establishes implementation intent and substantial coverage; it does not establish current trace reproduction across the complete claimed boundary. No valid current divergence or forbidden identity dependency was demonstrated.

## Shipped behavior

Dated history establishes delivery of:

- **July 15:** Immutable definitions shared separately from mutable card facts.
- **July 16:** Typed definition identity, deterministic state hashes, and compatibility checks.
- **July 16:** Controlled allocation evidence and independent root/sibling/rollout-slot contracts.
- **July 16:** Withdrawal of unsupported performance comparisons and correction of allocation-receipt provenance.

The chapter start is unknown, so these are delivered contributions rather than a precise chapter-wide before/after comparison.

## Task and PR receipts

All seven supplied provider Tasks are complete:

| Task | Contribution |
|---|---|
| [ETU-73](https://linear.app/loopflow/issue/ETU-73/separate-contentpack-and-measure-clone-costs) | Shared ContentPack/CardDefId foundation; `61b7e15`. |
| [ETU-72](https://linear.app/loopflow/issue/ETU-72/add-deterministic-matchstate-hash-and-identity-boundary-checks) | Hash and identity contract; `e26103b`. |
| [ETU-71](https://linear.app/loopflow/issue/ETU-71/replace-w2-179-local-diagnostics-with-contract-evidence) | Removed invalid performance claims; `b27c250`. |
| [ETU-70](https://linear.app/loopflow/issue/ETU-70/prove-contentpack-sharing-and-clone-allocation-boundary) | Allocation and sharing proof; `8aa743b`. |
| [ETU-69](https://linear.app/loopflow/issue/ETU-69/refresh-w2-208-allocation-receipt-after-landed-rebase) | Receipt refresh following PR #81. |
| [ETU-68](https://linear.app/loopflow/issue/ETU-68/re-run-contentpack-allocation-receipt-from-merged-pr-84-revision) | Further provenance correction following PR #84. |
| [ETU-67](https://linear.app/loopflow/issue/ETU-67/duplicate-of-w2-216-no-work) | Duplicate closed without implementation. |

Task descriptions preserve repeated stale-revision problems after rebases. Those are evidence-provenance failures, not demonstrated sharing or isolation failures.

## Priority and ownership findings

No incomplete provider Task appears in the current Project. Consumer throughput, latency, memory budgets, and branch-representation selection explicitly belong to Search and Study Runtime Prototype.

Historical W2-207 shows that runtime regression gating was once unfinished here. Its exact former KR and subsequent disposition remain missing; its omission from the current ledger is not proof of completion.

## Surprises

Repeated receipt-refresh Tasks were needed after landing/rebasing. Evidence provenance required separate work even when the implementation remained useful.

The original performance narrative was withdrawn rather than converted into an unsupported improvement claim. Shared content is established architecture; its overall speed benefit remains a different question.

## Open work

No current incomplete provider Task was found. Six completed Tasks retain `ready` runtime records and missing historical worktrees in the supplied snapshot; ETU-67 has no runtime record. These are lifecycle discrepancies, not authorization to restart implementation. All states remain untouched.

## Evidence gaps

- Exact chapter dates and full historical KR lineage, especially the former third KR.
- Current source-bound allocation and sharing/isolation results covering the supported environment and branch paths.
- Current deterministic-trace and identity-boundary execution evidence.
- Reconciliation of completed provider Tasks with historical runtime residue.

```json
{"1":"unknown","2":"unknown"}
```

