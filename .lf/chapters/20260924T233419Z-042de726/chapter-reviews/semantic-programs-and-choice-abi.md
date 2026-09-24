# Project chapter report: Semantic Programs and Choice ABI — interval unknown (baseline)

Run id: `run_0b766168ce9e4dd19f1a1f1c28a4bdc3`

Reviewed September 24, 2026, at HEAD `7ad9a41f99cf367240ab380b197226a122398820`.

## Project synthesis

**Intended user improvement:** The definition primarily names infrastructure for rules authors, learning developers, and gameplay consumers: compiled card semantics, complete legal offers, atomic Commands, and viewer-safe learning inputs. It does not explicitly promise better human gameplay or stronger agents.

**Current user experience:** Developers have a checked-in compiler, compiled gameplay integration, structured choices beyond the legacy tensor width, and a semantic learning projection. Eleven focused checks passed against current Python source. The complete native-engine and consumer-path claims were not freshly verified.

**Chapter conclusion:** The Project delivered substantial infrastructure and useful bounded measurements. Its historical completion flags exceed what this review can certify today. No current behavioral counterexample was established; the remaining uncertainty concerns execution, coverage, and provenance.

**Decisive facts:**

- **July 16 UTC:** The retained decoder benchmark reports 35 target candidates, 64 represented attacker declarations, zero illegal decodes, and 6,435/6,435 matching decisions. Only 106 decisions used the supported structured path; 6,329 used explicit legacy fallback.
- **July 16 UTC:** The semantic-projection receipt covers 4,096 observations across 19 games, with zero projection failures or unadmitted visible objects.
- **July 17:** Commit `731ade2` connected the exact authored matchup to compiled semantics. The earlier interpreter report’s “additive only” limitation therefore does not describe current integration.
- **September 24:** Six compiler/admission checks and five projection checks passed against this checkout. Historical records also revealed a former sixth KR whose exact wording was not recovered.

**Measurement fit:** The KRs credibly measure migration and representation mechanisms. They do not establish complete rules fidelity, convenient human controls, or stronger play. In particular, legacy agreement can preserve an existing rules simplification, and a synthetic decoder benchmark is not a trained-agent strength result.

**Next-chapter implication:** Likely **retire as an independent Project**, preserving its infrastructure and safeguards within consolidated Rules work. The accepted trained-challenger direction gives these mechanisms a concrete consumer; it does not justify automatically repeating old certification work or restoring the transferred learning experiment.

## Objective evidence packet

**Coverage: incomplete.** The supplied ledger and fresh read-only `lf pm show --wave rules --project semantic-programs-and-choice-abi --json` contain the same five exact current claims.

Repository history establishes an additional former KR6 concerning a four-arm transfer experiment:

- `1138b8a`, **2026-07-16T01:28:37Z**, records that experiment as an open Project KR.
- `9d45266`, **2026-07-16T05:52:07Z**, reports five of six KRs holding, with the gameplay transfer KR open.
- `17ff49f`, **2026-07-16T07:45:58Z**, records that policy training and transfer became Intelligence work and were no longer a Rules Project KR.

These records preserve its existence and disposition, but not its verbatim claim. Row 6 below explicitly records that unresolved historical obligation rather than inventing its wording. Complete provider revisions, other possible earlier wordings, and chapter boundaries remain unavailable.

**Expected starting-ledger rows: 5 · Actual reported rows: 6**  
Five exact current rows plus one unresolved historical row.  
**Verdict totals: 0 holds · 0 does not hold · 6 unknown**

All current exact claims were first supported here by the supplied ledger observed at **2026-09-24T21:59:35.585349Z**. Earlier implementation and ordinal references do not establish their exact wording dates.

**Fresh verification:** Eleven existing test functions were executed directly through `uv`, using an existing environment, disabling bytecode writes, and verifying imports resolved to this checkout:

- Six compiler checks: deterministic current artifacts, exact product-deck admission, complete definition/token closure, registry bindings, typed instructions, and opcode compatibility.
- Five projection checks: deterministic catalog, structural recombination, required token families, identity ablation under reordered IDs, and viewer-only projection with deterministic batching.

These were bounded Python checks. Native Rust tests, actual-environment projection tests, full games, and benchmarks were not rerun. No files, planning objects, Tasks, or branches were changed.

## Definition

> Compile creator-selected card semantics into reviewed, checked-in typed programs and expose complete legal decisions through a versioned structured-offer and atomic-command ABI shared by rules, learning, and gameplay. Export a stable viewer-safe semantic projection as an Intelligence input, but keep learned representation, policy training, and transfer evidence in the Intelligence wave. Prove the rules migration incrementally with differential legacy evidence and reproducible boundary performance, without expanding into general Magic coverage.

**Verdict: unknown.**

**Evidence:** Current source and dated deliveries substantiate the compiler, typed programs, structured boundary, learning projection, and compiled matchup integration. Fresh checks establish important artifact and projection properties. They do not establish the complete legal-decision and cross-consumer conjunction through current native execution. The definition is excluded from KR totals.

## KR ledger

### 1. Every card in the two-deck acceptance slice is admitted by an offline compiler into checked-in, versioned typed IR; a generated coverage report is complete for that slice, and source checks plus trace tests demonstrate no card-name dispatch in the interpreter.

**Status and active interval:** Current · establishment unknown; exact wording first observed September 24, 2026 → review.

**Verdict: unknown.**

**Evidence:** Six fresh checks passed on September 24. The generated coverage contains **31 definitions: 29 deck definitions plus two referenced token definitions**, matching the actual 41-card UR and 40-card GW manifests. Current compiler output matches checked-in artifacts. See [compiler tests](../../../../tests/semantic/test_compiler.py:64).

The [interpreter tests](../../../../managym/tests/semantic_interpreter_tests.rs:1) enumerate all 37 programs, reject name-bearing instruction fields, and exercise representative branches and interactions. The [W2-223 report](../../../../experiments/w2-223-typed-ir-interpreter.md:1), delivered July 16 in `be2e3c4`, retains a 263-test Rust pass including 14 interpreter tests.

Later compiled gameplay integration is present in [authored-match tests](../../../../managym/tests/authored_match_tests.rs:35), delivered July 17 in `731ade2`. This prevents misclassifying the current interpreter as merely an unused trace instrument.

The admission portion has fresh exhaustive evidence. Current Rust parsing, execution, and trace behavior were not exercised, so the entire conjunction remains unverified. No name-dispatch counterexample was found.

### 2. Priority, Lightning Bolt targeting, and declare attackers emit typed structured offers and accept revision-bound, prompt-bound atomic commands; tests prove the offers legal and complete, including a state with more than 32 legal branches, with no truncation or candidate cap.

**Status and active interval:** Current · establishment unknown; exact wording first observed September 24, 2026 → review.

**Verdict: unknown.**

**Evidence:** Priority and targeting delivery is recorded in `9f6d56c`, July 15 local time; attacker offers followed in `75b8c1c`, July 16 UTC.

Current [priority/target tests](../../../../managym/tests/rules/structured_offers.rs:147) cover atomic Bolt casting, pass equivalence, candidates beyond the legacy tensor width, and rejection of fabricated or stale identifiers without mutation.

The [attacker test](../../../../managym/tests/rules/structured_attacker_offers.rs:232) constructs six eligible creatures for each deck, represents all **64 subsets**, and executes every subset through both atomic and legacy paths. [Offer construction](../../../../managym/src/agent/structured_offer.rs:702) represents combinations through selection rather than truncating an enumerated action list.

These are substantive source and historical execution evidence. The current native tests were not run, and no current execution packet established the full legality/completeness conjunction. No truncation or illegal-offer counterexample was observed.

### 3. On the two-deck acceptance suite, differential tests show that the legacy fixed-action adapter and structured command path produce the same legal outcomes and deterministic seeded traces for every action representable by the legacy ABI.

**Status and active interval:** Current · establishment unknown; exact wording first observed September 24, 2026 → review.

**Verdict: unknown.**

**Evidence:** The current exhaustive attacker test compares all 64 declarations in each of two constructed positions. Separate Bolt and pass tests compare their covered boundaries.

The [seeded acceptance test](../../../../managym/tests/rules/structured_attacker_offers.rs:379) runs four games across both deck assignments, comparing observations, events, action spaces, and winners. It uses distinct atomic/legacy paths for attackers; other decisions execute the same legacy `step` on both sides.

The retained decoder result likewise distinguishes **106 supported structured decisions** from **6,329 fallback decisions**. Its 6,435 matching decisions therefore cannot be treated as 6,435 independent structured-path comparisons.

These checks support a bounded migration claim. They do not establish the complete “every action representable” denominator, and current native execution was not observed. A narrower test denominator is an evidence gap, not itself a demonstrated parity failure.

### 4. A structured policy-decoder prototype completes a fixed seeded evaluation containing states with more than 32 legal choices with zero overflow and zero illegal decodes; a checked-in report compares action agreement on shared states, seat-balanced win rate, p50/p95 decision latency, environment throughput, and peak RSS against the legacy adapter.

**Status and active interval:** Current · establishment unknown; exact wording first observed September 24, 2026 → review.

**Verdict: unknown.**

**Evidence:** The [report](../../../../experiments/structured-policy-decoder.md:1) and [raw result](../../../../experiments/data/structured-policy-v1.json:13156), delivered July 16 in `dd00326`, retain the requested historical measurements:

- Eight games per adapter, with both deck assignments and zero cap hits.
- 35 explicit targets and 64 represented attacker declarations.
- Zero recorded overflows, illegal decodes, or trace mismatches.
- Identical deck win rates: UR 12.5%, GW 87.5%.
- Structured versus legacy focused latency: **116.4/191.8 µs** versus **90.2/206.0 µs**, p50/p95.
- Throughput: **180.107** versus **186.994 games/s**; peak RSS **25.7 MiB** for each.

The report explicitly uses a fixed synthetic scorer and legacy fallback for unsupported full-game decisions. Those boundaries are compatible with a narrow prototype, but exclude policy-strength or complete migration claims.

The raw result includes workload identity and host information but no measurement timestamp, source revision, or native-binary identity. Repository history dates its retention, not its exact execution. This review did not reproduce it against current code; the historical result cannot certify the current conjunction.

### 5. A versioned, viewer-safe semantic learning projection exposes each visible card and ability as typed program tokens covering opcodes, values, selectors, control flow, costs, and choice roles; tests prove stable ContentPack binding, deterministic ragged batching, hidden-information safety, and an ablated path with no card-name or opaque CardDefId feature.

**Status and active interval:** Current · establishment unknown; exact wording first observed September 24, 2026 → review.

**Verdict: unknown.**

**Evidence:** Five focused current checks passed, covering the typed catalog, required token families, recombination, identity ablation, and deterministic projection/batching. See [projection tests](../../../../tests/semantic/test_learning_projection.py:104).

The [W2-215 receipt](../../../../experiments/w2-215-semantic-projection.md:1), measured July 15 local time and delivered July 16 UTC in `f567f60`, records:

- 4,096 observations across 19 games.
- 176,802 admitted visible-object bindings.
- 31 definitions, 37 programs, and 2,088 catalog tokens.
- Zero projection failures, unadmitted visible objects, or valid opaque identity features in `semantic_only` mode.

Current [environment tests](../../../../tests/semantic/test_learning_projection_env.py:33) check the actual compiled ContentPack manifest and invariance under hidden-state determinization. They were inspected but not executed. The fresh tests use synthetic manifests and observation objects, so they do not substitute for that native boundary. The complete environment-binding and privacy conjunction remains unknown.

### 6. Historical KR6 — exact claim unavailable

**Exact claim:** Not recovered. Dated records describe a reproducible four-arm gameplay/held-out transfer experiment; that description is not presented as verbatim KR text.

**Status and active interval:** Removed from this Project/transferred in scope · establishment unknown; existence supported July 16, 2026 at 01:28:37 UTC → no longer a Rules Project KR by 07:45:58 UTC.

**Verdict: unknown.**

**Evidence:** The historical records identified in the coverage section distinguish an open transfer result from completed projection and decoder prerequisites. `17ff49f` explicitly assigns learning and transfer to Intelligence. Removal is an ownership decision, not proof of scientific completion. Without the exact claim and its revision history, its original acceptance conditions cannot be judged.

## Shipped behavior

Dated repository history establishes delivery of:

- Offline typed-IR compilation: `8c0f1f0`, July 15.
- Structured priority/target and attacker boundaries: `9f6d56c`, `75b8c1c`.
- Decoder benchmark, learning projection, and generic interpreter: `dd00326`, `f567f60`, `be2e3c4`, July 16 UTC.
- Actual selected-matchup execution from compiled semantics: `731ade2`, July 17.

These are delivered capabilities. An unknown chapter start prevents a precise chapter-wide before/after comparison.

## Task and PR receipts

| Task | Supplied/current PM state | Contribution |
|---|---|---|
| [ETU-58](https://linear.app/loopflow/issue/ETU-58/complete-two-deck-typed-ir-admission-and-interpreter-proof) | Completed | Interpreter proof; historical memory identifies PR #100 as final W2-223 scheduling provenance, superseding an older PR #96 record. |
| [ETU-59](https://linear.app/loopflow/issue/ETU-59/expose-viewer-safe-semantic-program-tokens-to-learning) | Completed | Learning projection; PR #92, `f567f60`. |
| [ETU-60](https://linear.app/loopflow/issue/ETU-60/prototype-structured-policy-decoder-and-benchmark-legacy-abi) | Completed | Decoder benchmark; PR #88, `dd00326`. |
| [ETU-61](https://linear.app/loopflow/issue/ETU-61/add-structured-attacker-offers-and-acceptance-suite-adapter-oracle) | Completed | Attacker offers and differential oracle; `75b8c1c`. |
| [ETU-62](https://linear.app/loopflow/issue/ETU-62/compile-two-deck-card-semantics-into-checked-in-typed-ir) | Completed | Compiler and admission artifacts; `8c0f1f0`. |
| [ETU-63](https://linear.app/loopflow/issue/ETU-63/prototype-structured-offers-for-priority-and-bolt-targeting) | Completed | Priority and Bolt offers; `9f6d56c`. |

These receipts establish contributions, not complete current KR satisfaction.

## Priority and ownership findings

No incomplete provider Task appears in this Project’s current list. Learning and transfer were explicitly moved to Intelligence in July; they should not return to Rules merely because historical KR6 remains unverified.

The new chapter’s Learn/Lesson concern is compatible with complete typed admission: compiling the implemented mechanic does not prove that mechanic faithfully expresses the intended game. Likewise, uncapped engine choices do not establish that the human can navigate them comfortably.

## Surprises

- The historical portfolio had six KRs; the current ledger has five.
- Most decoder benchmark decisions used explicit legacy fallback despite perfect aggregate agreement.
- The initial interpreter report describes an additive instrument, while later code connects compiled semantics to actual gameplay.
- The exact UR product deck has 41 cards; fresh parity checks preserve that manifest rather than silently normalizing it to 40.

## Open work

All six supplied Tasks are PM-completed. Their runtime snapshots retain `ready`/blocked residue, missing historical worktrees, and successor-PR recommendations. Those inconsistencies do not establish six active implementation efforts or authorize restarting them.

No Task, PR, or lifecycle state was changed.

## Evidence gaps

- Exact historical KR6 wording, its revisions and transfer record; complete earlier KR lineage and chapter dates.
- Current debug Rust execution of interpreter, offer, and differential contracts.
- An explicit denominator supporting KR3’s universal legacy-equivalence claim.
- A decoder result bound to source/native identity, with supported and fallback decisions kept separate.
- Current actual-environment ContentPack and hidden-information projection checks.

Ordinal-to-verdict JSON; ordinal 6 denotes the unresolved former KR6:

```json
{
  "1": "unknown",
  "2": "unknown",
  "3": "unknown",
  "4": "unknown",
  "5": "unknown",
  "6": "unknown"
}
```

