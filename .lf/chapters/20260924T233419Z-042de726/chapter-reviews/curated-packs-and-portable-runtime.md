# Project chapter report: Curated Packs and Portable Runtime — interval unknown (baseline)

Run id: `run_9b42a331a69649e195d15e50bb1e3b5a`

Scope: checkout HEAD `7ad9a41f99cf367240ab380b197226a122398820`, committed 2026-09-24 at 21:55:58 UTC. Review performed read-only; no tests, experiments, repairs, archival writes, or PM mutations.

## Project synthesis

**Intended user improvement**: Players should reliably launch the selected Etude Fantasia matchup, continue playing without public networking after installation, and benefit from portable local authority only where measurements justify it.

**Current user experience**: The repository supplies a versioned local asset pack, deterministic fallback treatments, and a documented one-command launcher with offline-reload verification. Retained September 24 results demonstrate short offline play and recovery on macOS. This review cannot certify the complete offline-match claim or clean-machine performance at the current HEAD. Interchangeable process/native-worker authority and the comparative portability benchmark remain unproven.

**Chapter conclusion**: The bet produced useful packaging and launch infrastructure. Available evidence does not close its full outcome claims. Human reports also show that reaching an actionable board is insufficient to establish an experience worth playing through.

**Decisive facts**:

- On July 15, commit `08ee2bd` introduced the exact asset pack and offline full-match browser test. The current manifest still pins 41 UR cards, 40 GW cards, and 31 distinct card/token identities.
- On September 24, retained clean-machine evidence recorded a **61,590 ms** Ubuntu failure, followed by a **24,080 ms** macOS pass with offline session recovery and zero public requests. The latter is explicitly not Ubuntu certification. See [repair evidence](../website-182-result.md:60).
- At current HEAD, protocol documentation limits certification to the integrated recovery-to-command boundary; it does not establish two interchangeable authority transports. See [protocol boundary](../../../../protocol/README.md:34).
- On September 24, the User reported awkward decisions, missing transitions, and disappointing Learn behavior. These challenge the broader product outcome, although they do not directly falsify asset locality or startup timing.

**Measurement fit**: Pack integrity and launch/reload measurements directly support reliable entry into play. Adapter conformance and portability benchmarks measure enabling infrastructure. None of these KRs establishes that the User can comfortably complete a game or challenge an identified trained checkpoint.

**Next-chapter implication**: Likely **rewrite**—preserve pack integrity and launch/recovery checks as supporting requirements within Game’s consolidated Project. Defer adapter/WASM work unless the trained-challenger experience exposes a concrete need. This is a recommendation, not a disposition applied here.

## Objective evidence packet

**Coverage: incomplete.** All supplied current KRs are enumerated. `lf pm show --wave game --project curated-packs-and-portable-runtime --json` returned the same four claims, with no drift.

Repository history preserves a July 16 assessment reporting this Project as “1/4 KRs hold” (`9d45266:scratch/garden-scan.md`), but that summary does not preserve every verbatim claim or its revisions. Searches of available history and local records recovered no additional attributable, verbatim earlier KR rows. Provider revision history and a chapter-start ledger remain missing.

**Expected KR rows: 4 · Actual KR rows: 4**  
**Verdict totals: 0 holds · 0 does not hold · 4 unknown**

The supplied exact ledger was observed at **2026-09-24T21:59:35.585349Z**. Earlier implementation dates establish delivery history, not when today’s exact KR wording became a promise.

Historical failures are retained below. They do not automatically establish failure of the repaired current HEAD.

## Definition

> Prove that Etude Fantasia can ship UR Lessons versus GW Allies as a self-contained, versioned experience pack and can swap local authority transports behind one protocol, making clean-machine and offline play reliable while requiring measured evidence before any browser or WASM authority investment.

**Verdict**: unknown.

**Evidence**: Packaging and local launch machinery are present, with bounded offline-reload results retained. No inspected evidence demonstrates swapping process and native-worker authority under the required shared conformance suite. The complete definition therefore remains unsupported. This row is excluded from KR totals.

## KR ledger

### 1. A versioned manifest pins the UR Lessons and GW Allies deck definitions, every card art treatment and source or license attribution, content hashes, and a deterministic missing-asset fallback; an automated network-denied match proves play makes no runtime asset request outside the installed pack.

**Status and active interval**: current · exact wording observed September 24, 2026 → current HEAD review; establishment date unknown.

**Verdict**: unknown.

**Evidence**:

- The [manifest](../../../../frontend/src/lib/packs/tla-ur-lessons-vs-gw-allies/v1/manifest.json:1) pins version `1.0.0`, both decks, reachable tokens, authored treatments, provenance, rights records, and the `fallback-v1` algorithm. Rights entries explicitly use `NOASSERTION`; the pack does not claim a redistribution license for third-party art.
- The [loader](../../../../etude/curated_pack.py:246) validates inventory and provenance and derives the manifest SHA-256. [Frontend tests](../../../../frontend/src/lib/curated-pack.test.ts:11) cover all 31 identities, deterministic fallback, and rejection of remote treatment URLs.
- The [offline browser test](../../../../frontend/e2e/offline-pack.spec.ts:58) denies public HTTP/WebSocket traffic, reloads, completes a match, opens replay, and asserts zero public requests.
- These are substantive implementation and test-definition evidence, introduced July 15 and inspected September 24. No retained successful execution of that complete match test at the scoped HEAD was inspected. September 24 clean-machine receipts exercise shorter play/reload sequences, not the full-match denominator.

The checked PM flag is insufficient to close the conjunction. No current asset-locality counterexample was found.

### 2. On a documented clean-machine profile, one command reaches a playable UR Lessons-versus-GW-Allies human-versus-agent match in under 60 seconds; after installation, a network-disabled reload reaches the same playable matchup from local assets.

**Status and active interval**: current · exact wording observed September 24, 2026 → current HEAD review; establishment date unknown.

**Verdict**: unknown.

**Evidence**:

- The [documented profile](../../../../docs/clean-machine-play.md:20) specifies Ubuntu 24.04 x86_64, provisioned tools, absent checkout artifacts, and an external clock including installation and launch.
- The [current browser proof](../../../../frontend/e2e/clean-machine.spec.ts:234) checks the Search 64 matchup, the strict 60,000 ms limit, pack identity, identical session/visible state after reload, another accepted action, and zero public requests.
- The [CI job](../../../../.github/workflows/ci.yml:291) runs this real launcher path without restored installation caches and uploads its receipt.
- September 24 evidence records **61,590 ms** on failed head `2a58adf`, exceeding the budget. A separate retained record, `1b8f73a:scratch/ci-fix.md:6`, reports another earlier failure of **62,784 ms** on `d8a58df`.
- Subsequent repairs are present at current HEAD. Retained macOS results of **24,080 ms** and **24,722 ms** establish bounded launch/reload behavior on their stated hosts, not the declared Ubuntu profile. Historical successful CI is reported, but its complete receipt for the current scoped revision was not inspected.

These failures disprove uninterrupted historical compliance on those earlier heads. They do not prove the repaired HEAD fails; the current conjunction remains unknown.

### 3. Process and native-worker authority adapters pass the same versioned protocol conformance suite for match start, command handling, recovery, and replay fixtures without client-specific game semantics.

**Status and active interval**: current · exact wording observed September 24, 2026 → current HEAD review; establishment date unknown.

**Verdict**: unknown.

**Evidence**: Current [protocol documentation](../../../../protocol/README.md:18) describes Rust/Python/TypeScript validation of a shared fixture and live Python recovery/command representation. It explicitly limits the certification boundary. Cross-language validation does not establish two authority transports passing the requested lifecycle suite.

Searches of runtime, frontend, tests, scripts, and architecture documentation found no qualifying paired-adapter execution report. The [launcher scope](../../../../docs/clean-machine-play.md:128) explicitly excludes authority-adapter work. No direct failing paired-adapter run was observed; missing proof is therefore unknown.

### 4. A reproducible benchmark records cold and warm startup, p50 and p95 command latency, peak memory, shipped bytes, and deployment constraints for process and native-worker authority plus a browser portability probe, ending with an explicit evidence-backed go or no-go for a WASM authority adapter.

**Status and active interval**: current · exact wording observed September 24, 2026 → current HEAD review; establishment date unknown.

**Verdict**: unknown.

**Evidence**: The [Wave charter](../../../../wave/game/README.md:42) calls for measured portability and sequences process/worker benchmarks before WASM. Available launch measurements cover one local FastAPI/Vite path. They do not supply the required two-adapter comparison, warm/cold measurements, latency distributions, memory, shipped bytes, deployment constraints, browser probe, and resulting decision.

No qualifying retained benchmark report was found. Architectural preference and deferred WASM investment are not substitutes for the specified measurement packet.

## Shipped behavior

Repository history establishes delivery of:

- The exact local treatment pack and offline browser scenario: `08ee2bd`, July 15, 2026; public-WebSocket blocking strengthened by `63ed073`.
- The one-command launcher, structured diagnostics, offline recovery proof, and uncached CI job: `d14d801`, July 16.
- Startup/proof scheduling repairs: September 24 changes now included in HEAD.

These are delivered capabilities. No known chapter-start baseline supports a precise before/after user-outcome comparison.

## Task and PR receipts

- [ETU-8 — Freeze one exact matchup asset pack](https://linear.app/loopflow/issue/ETU-8/freeze-one-exact-matchup-asset-pack-with-offline-fallback): supplied PM record says completed; pack implementation is present. Runtime still reports ready/blocked because its historical worktree is missing, with a “start next PR” recommendation.
- [ETU-7 — Prove one-command launch and offline reload](https://linear.app/loopflow/issue/ETU-7/prove-one-command-clean-machine-launch-and-offline-reload): supplied PM record says completed; implementation is present. Runtime reports ready/blocked on a missing historical worktree and retains an open-phase [PR #80](https://github.com/loopflowstudio/manabot/pull/80) record without merge data.

Those lifecycle discrepancies are reconciliation findings, not evidence that the shipped code disappeared or that either Task should resume.

## Priority and ownership findings

Both supplied Tasks serve pack/launch outcomes and explicitly exclude adapter/WASM work. No aligned active Task for KR3 or KR4 appears in this Project’s ledger.

The User’s September 24 direction prioritizes complete games against trained challengers. The [certified minimal launcher profile](../../../../docs/clean-machine-play.md:137) excludes policy-checkpoint opponents, which require the full development installation. That is a concrete gap between the old launch proof and the proposed chapter outcome.

## Surprises

The under-60-second proof regressed despite prior delivery, demonstrating that startup certification requires ongoing evidence.

Offline packaging also coexists with disappointing gameplay. The [Learn implementation comment](../../../../managym/src/cardsets/strixhaven.rs:2) documents the restricted mechanic. Alongside the User’s interaction reports, this limits any inference from “packaged and actionable” to “faithful and comfortable to finish.” It does not itself falsify these narrower packaging KRs.

## Open work

No incomplete provider Task appears in the supplied Project ledger. ETU-7 and ETU-8 retain contradictory runtime residue described above; this review left it untouched.

## Evidence gaps

- **Lineage:** chapter dates and complete historical KR revisions.
- **KR1:** retained successful full-match network-denied execution tied to the scoped revision, alongside manifest/fallback checks.
- **KR2:** current-revision reference-profile receipt and sufficient repeated evidence to support reliability.
- **KR3:** both concrete transports exercised through the same complete lifecycle conformance suite.
- **KR4:** the full comparative benchmark and explicit measured portability decision.

Machine-readable verdicts:

```json
{"1":"unknown","2":"unknown","3":"unknown","4":"unknown"}
```

