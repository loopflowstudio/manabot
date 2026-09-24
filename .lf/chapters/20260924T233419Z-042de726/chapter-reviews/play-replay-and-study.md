# Project chapter report: Play, Replay, and Study — interval unknown (baseline)

Run id: `run_f5436f2188f14aaebf673528c7840dc8`  
Review date: **2026-09-24**  
Scope: checkout HEAD **`7ad9a41f99cf367240ab380b197226a122398820`**

## Project synthesis

**Intended user improvement:** Give a player and permitted watcher one continuous Etude Fantasia experience: play, state beliefs, compare attributable strategy, revisit decisions, try another line, and return safely.

**Current user experience:** The repository supplies substantial replay, contract, branch-isolation, and participant-role machinery. The ordinary application still cannot deliver the complete promised loop: advice uses a pinned completed-match fixture, and normal historical Study has no evidence provider for revealing the played/policy/search comparison. The User also reports missing transitions and difficult action selection.

**Chapter conclusion:** The work produced useful infrastructure and bounded demonstrations, but the continuous player experience remains incomplete. Treating demonstration and substrate completion as completion of the broader outcomes overstated progress.

**Decisive facts:**

- On **2026-09-24**, the ordinary advice endpoint still calls `request_versioned_fixture_advice`; the live page requests the fixture’s address.
- The normal Study configuration selects `UnavailableHistoricalStudyEvidenceProvider`. Successful three-plan comparison tests inject a separate fixture provider.
- Four focused Python contract checks passed against this checkout on **2026-09-24**, including schema agreement and rejection of private-hand data, RNG secrets, and identity drift.
- The User’s **2026-09-24** report of missing transitions contradicts the experienced rendering outcome, despite existing semantic presentation machinery.

**Measurement fit:** The contract KR measures its intended mechanism directly. The other KRs describe integrated player capabilities, but much supporting evidence covers fixtures, backend paths, or selected sequences. That evidence does not establish continuous live/Study integration or enjoyable full-game interaction.

**Next-chapter implication:** Likely **rewrite** around legible, executable, completed Allies versus Lessons games against identified trained opponents. Preserve the useful authority and replay substrate. Advice and richer Study should earn priority against that outcome rather than automatically carry forward.

## Objective evidence packet

**Coverage: incomplete.** All eight supplied current KRs are enumerated. No merged chapter start record or complete historical PM revision ledger was available. Repository history and preserved scratch contain an earlier verbatim observation of KR6, but no additional distinct Project KR wording was identified. Wave-level measures are not silently counted as Project KRs.

**Expected KR rows: 8 · Actual KR rows: 8**  
**Verdict totals: 1 holds · 4 does not hold · 3 unknown**

The supplied [ledger](play-replay-and-study-ledger.json) was observed at **2026-09-24T21:59:35.585349Z**; Task conditions were observed shortly afterward. Its seven checked boxes are status claims, not verdict evidence. No independent PM refresh was performed.

Evidence inspected includes current source, regression definitions, retained branch measurements, and dated Git history. Four existing Python contract test functions were executed directly through `uv`, using the available canonical environment and explicitly verifying that imports resolved to this checkout. All four passed. Rust, TypeScript, browser, and full-game suites were not freshly executed. No rendering environment was available; this checkout also lacked its own `.venv` and `frontend/node_modules`.

No files, PM objects, Tasks, branches, or archives were changed.

## Definition

> Make one selected Etude Fantasia match a continuous AI-assisted testing-house loop: pilot or watch on the authoritative semantic table, state explicit viewer-safe beliefs, inspect attributable conditional strategy, play through ordinary Commands, reopen every historical decision, retry an exact line, compare it, and return without client-side rules, hidden-information leakage, or a second live/Study surface. The remaining open work (2026-07-18) is the live seam: an addressable canonical decision for the in-progress match and a player-authored belief input surface, so the same advice identity resolves mid-game against the tracked posterior instead of a pinned completed-match fixture (the Game half of the results-first roadmap in docs/plans/results-first-roadmap.md).

**Verdict: does not hold.**

**Evidence:** As inspected on **2026-09-24**, [the ordinary endpoint](../../../../etude/server.py:3302) serves fixture advice, while [the live page](../../../../frontend/src/routes/+page.svelte:44) uses fixture metadata rather than the current live decision. [Normal Study configuration](../../../../etude/server.py:937) supplies an unavailable evidence provider. These are direct gaps in the definition’s live posterior and comparison requirements. The definition’s embedded July date does not establish the historical interval of every KR.

## KR ledger

### 1. Casting, targeting, resolving, combat, damage, death, and turn transitions render as ordered semantic beats with skip or fast-forward behavior and no snapshot-diff inference.

**Status and active interval:** Current · establishment unknown; first supported Project-ledger observation **2026-09-24** → current at review.

**Verdict: does not hold.**

**Evidence:** The User reported “lack of transitions” on **2026-09-24**, recorded in [chapter-direction.md](../chapter-direction.md:35). This is counterevidence to the player-visible outcome. Its exact position and application build were not retained, so it does not identify which event family or rendering step failed.

Positive implementation evidence remains: commits `43023ac` and `a6d7479`, both **2026-07-16 UTC**, introduced spell and combat/turn presentation. [Presentation architecture](../../../../docs/architecture/presentation-runtime.md) describes authoritative event ordering; [presentation tests](../../../../frontend/src/lib/presentation.test.ts:132) cover skip, fast-forward, and reduced motion. Those mechanisms do not erase the reported experience failure or establish complete coverage at HEAD.

### 2. A versioned cross-language StudyArtifact and DecisionEvidence contract preserves viewer-safe match, decision, command, model, search-budget, and provenance identities while keeping policy mass, visits, value, robustness, and uncertainty distinct.

**Status and active interval:** Current · establishment unknown; first supported Project-ledger observation **2026-09-24** → current at review.

**Verdict: holds.**

**Evidence:** This is a bounded contract claim. The current [Python contract](../../../../etude/study_protocol.py:292), [Rust contract](../../../../managym/src/study.rs:423), and [TypeScript contract](../../../../frontend/src/lib/study-protocol.ts:150) represent the named metrics separately and carry the identity structure. The contract entered repository history in `6833e32`, **2026-07-16T05:19:27Z**.

On **2026-09-24**, all four checks in [test_study_protocol.py](../../../../tests/etude/test_study_protocol.py:79) passed against current source: fixture round-trip with exact historical bindings, Python agreement with the checked Rust-generated schema, rejection of opponent-private hand identities, and rejection of RNG secrets and binding drift. Rust and TypeScript also contain matching conformance and privacy checks; these were inspected, not freshly run.

This verdict concerns the contract boundary. It does not certify every caller or the complete live experience.

### 3. Every historical player decision in a completed selected match has a stable viewer-safe address that restores the exact canonical frame, offer, played command, and event cursor, with direct play, replay, and Study showing the same facts.

**Status and active interval:** Current · establishment unknown; first supported Project-ledger observation **2026-09-24** → current at review.

**Verdict: unknown.**

**Evidence:** Canonical indexing shipped in `1274363`, **2026-07-17T14:23:15Z**. [The pinned-match regression](../../../../tests/etude/test_replay_index.py:247) generates two matches, compares deterministic output, restores every projected decision for both viewers, and checks continuation completeness. [The GameSession regression](../../../../tests/etude/test_replay_index.py:270) distinguishes deliberate choices from automatic passes.

These are well-targeted tests, but they were not executed in this review. No current complete-match denominator with corresponding direct-play, replay, and Study observations was established. There is no identified exact-restoration counterexample; the universal conjunction therefore remains unknown.

### 4. At one restored decision, a player can Retry before reveal, then compare separately labelled played, policy, and search plans on the table through one bounded canonical continuation.

**Status and active interval:** Current · establishment unknown; first supported Project-ledger observation **2026-09-24** → current at review.

**Verdict: does not hold.**

**Evidence:** The current normal configuration uses [UnavailableHistoricalStudyEvidenceProvider](../../../../etude/server.py:937), whose [implementation](../../../../etude/study_runtime.py:44) raises evidence unavailable. The [normal-runtime regression](../../../../tests/etude/test_study_runtime.py:232) explicitly expects reveal to return HTTP 409. The ordinary [browser regression](../../../../frontend/e2e/study-unavailable.spec.ts:9) expects the unavailable message and no played-plan element.

The successful comparison path instead [injects fixture evidence](../../../../frontend/e2e/study_server.py:125), as does [the positive backend test](../../../../tests/etude/test_study_runtime.py:269). Inspected on **2026-09-24**, this configuration distinction is decisive: the shipped demonstration substrate does not provide the claimed ordinary player capability.

### 5. A player can enter an exact ephemeral branch through ordinary Commands and return in one action to the identical recorded state, event cursor, offer, and viewer projection.

**Status and active interval:** Current · establishment unknown; first supported Project-ledger observation **2026-09-24** → current at review.

**Verdict: unknown.**

**Evidence:** Strong historical backend evidence exists. The retained [Study branch receipt](../../../../experiments/data/rul-6-study-branch-v1.json:20), committed in `058016b` on **2026-07-18T06:30:19Z**, records 2,000 sequential checks and 512 retained-return checks, with zero canonical-return, source-digest, sibling-isolation, or replay-mutation mismatches. Its scope is one declared seed, workload, source closure, and native binary.

Current [branch tests](../../../../tests/etude/test_study_branch.py:109) compare frame, offer, command, event cursor, continuation, source digest, and sibling isolation. The normal [browser return test](../../../../frontend/e2e/study-unavailable.spec.ts:35) checks one-action restoration after Retry.

The historical receipt is substantive, but its source/binary identity was not revalidated against HEAD, and current player-path execution was not observed. No contrary return result was found.

### 6. At the same canonical decision reached live or through Study, the same unified surface accepts at least two explicit viewer-safe belief scenarios, obtains strategy from one pinned advisor identity and compute class, and shows reproducible action-probability, value, or uncertainty deltas while facts, beliefs, and advice remain distinct.

**Status and active interval:** Current · establishment unknown; first supported verbatim observation **2026-07-18T04:24:27Z**, in preserved design included by `3b58863` → current at review.

**Verdict: does not hold.**

**Evidence:** The [preserved design](../../../tmp/scratch-stash/jack-heart-prototype-belief-to-strategy-comparison-20260718T030645Z/prototype-belief-to-strategy-comparison.md:11) quotes this exact KR and explicitly describes a fixture-first prototype.

At HEAD, [live-page advice loading](../../../../frontend/src/routes/+page.svelte:65) still submits `adviceMetaState.address`. [The provider](../../../../etude/advice.py:1635) matches registered fixture requests and returns retained responses. This supports a two-scenario demonstration displayed on the live page, but does not connect the actual live canonical decision to the same Study request.

The dated **2026-09-24** source observation defeats the live-decision conjunct; fixture deltas alone do not satisfy it.

### 7. A pilot and one permitted watcher can inhabit the same canonical decision surface under explicit viewer roles, compare or share viewer-safe belief and strategy artifacts, and explore an isolated line without pausing or mutating the authoritative live match; only the pilot can submit its offered Command, and the same table continues into Study.

**Status and active interval:** Current · establishment unknown; first supported Project-ledger observation **2026-09-24** → current at review.

**Verdict: unknown.**

**Evidence:** Shared-table work entered history in `cebe049`, **2026-07-18T09:42:31Z**. Current tests cover [watcher mutation denial](../../../../tests/etude/test_testing_house.py:139), [private-until-shared beliefs](../../../../tests/etude/test_testing_house.py:228), [isolated watcher exploration while the pilot advances](../../../../tests/etude/test_testing_house.py:268), and [both roles continuing into Study](../../../../tests/etude/test_testing_house.py:449).

However, [belief authorship](../../../../etude/server.py:2761) binds artifacts to fixture metadata, and [browser advice assertions](../../../../frontend/e2e/testing-house.spec.ts:116) check pinned fixture identity. These constrain what the shared experience demonstrates. They do not alone prove that the role/isolation claim fails, but no fresh two-participant execution established the full conjunction at HEAD.

### 8. The in-progress match exposes an addressable canonical current decision (closing the GAM-4 deferral in which the canonical replay address exists only at game close), and a player can author a viewer-safe belief mid-game and receive advice whose identity resolves against the live tracked posterior — the same fail-closed identity discipline as the completed-match path, with the fork/Retry/return UI sequenced behind it.

**Status and active interval:** Current · establishment unknown; first supported Project-ledger observation **2026-09-24** → current at review.

**Verdict: does not hold.**

**Evidence:** Current `ed2` and posterior substrate is real: [test_live_advice.py](../../../../tests/etude/test_live_advice.py:44) covers promotion of the pending address to committed replay identity and unavailable paths. These tests were inspected, not run.

The decisive **2026-09-24** counterexample is [production routing](../../../../etude/server.py:3312): `/api/advice` still selects the fixture adapter. [Game memory](../../../../wave/game/MEMORY.md) explicitly preserves this as unfinished work. Address and tracker availability do not satisfy the player-facing live-posterior conjunction.

## Shipped behavior

Repository history supports these additions, without establishing a chapter-start comparison because the interval is unknown:

- Ordered semantic presentation machinery for spell and combat/turn sequences.
- Versioned, viewer-safe Study evidence contracts and canonical replay addresses.
- Structured branch execution and exact-return machinery.
- Retry-before-reveal controls, with normal evidence unavailability exposed explicitly.
- Pilot/watcher roles, explicit belief sharing, and isolated historical exploration.
- A shared fixture-backed belief comparison surface and live-address/posterior substrate.

## Task and PR receipts

These establish contributions, not completion of the broader outcomes:

| Contribution | Dated repository evidence |
|---|---|
| ETU-10: Study contract | `6833e32`, 2026-07-16 |
| ETU-11/12: presentation sequences | `a6d7479` / `43023ac`, 2026-07-16 |
| ETU-9: canonical replay index | `1274363`, 2026-07-17 |
| ETU-15: shared advice prototype | `3b58863`, 2026-07-18, PR #143 |
| ETU-17: Retry substrate | `9318ef6`, 2026-07-18 |
| Exact-return stress evidence | `058016b`, 2026-07-18, PR #150 |
| ETU-13: pilot/watcher table | `cebe049`, 2026-07-18 |
| ETU-14 predecessor: live-advice substrate | `e61931e`, 2026-07-19; supplied history identifies PR #177 |

## Priority and ownership findings

The **2026-09-24** direction prioritizes full games against trained challengers. Collaborative advice and richer Study no longer have automatic priority merely because their old KRs remain incomplete.

[ETU-74](https://linear.app/loopflow/issue/ETU-74/plan-the-trained-challenger-chapter-for-allies-versus-lessons) is repository-wide chapter planning temporarily housed in this Project; its output is not delivery of the old player-facing definition.

The Learn limitation and awkward-action reports matter to the proposed next chapter. They do not independently falsify the old identity, replay, or isolation claims. The Learn report needs its intended pool/format established; it should not be converted into an unsupported universal rules verdict here.

## Surprises

- A successful Retry comparison browser test uses a different evidence configuration from the normal application.
- A merged “Connect live belief-conditioned advice” contribution coexists with production fixture routing.
- The User reports missing transitions despite completed presentation Tasks and dedicated semantic-event machinery.
- Completed Tasks can retain runtime missing-worktree conditions; those conditions do not make them unfinished product work.

## Open work

- [ETU-14](https://linear.app/loopflow/issue/ETU-14/replace-the-advice-fixture-with-the-live-belief-conditioned-advisor): **incomplete, blocked** in the supplied snapshot. Its successor worktree is missing; serial PR 2 is recorded as working with no publication.
- [ETU-74](https://linear.app/loopflow/issue/ETU-74/plan-the-trained-challenger-chapter-for-allies-versus-lessons): **incomplete, waiting/ready to start** in that snapshot. It owns proposal preparation.
- ETU-9–13 and ETU-15–17 are marked complete. Historical runtime residue was not treated as authorization to reopen them.
- ETU-21’s accepted abandonment remains intact. Nothing in this review resumes that attempt.

All dispositions remain unchanged.

## Evidence gaps

- **Historical coverage:** Earlier Project PM revisions, removed/replaced KRs, and a dated chapter-start ledger.
- **KR1 diagnosis:** Exact position, build, event stream, and reproduced rendering behavior for the reported missing transitions.
- **KR3:** A current complete selected-match decision ledger checked across direct play, replay, and Study.
- **KR5:** Current source/binary-bound return evidence plus ordinary UI execution.
- **KR7:** Current pilot/watcher execution covering the complete role, artifact, isolation, and Study-continuation conjunction.
- **KR4/6/8 closure:** Real configured evidence providers and live-decision integration, followed by demonstrations that exclude fixture substitution.

```json
{"1":"does not hold","2":"holds","3":"unknown","4":"does not hold","5":"unknown","6":"does not hold","7":"unknown","8":"does not hold"}
```

