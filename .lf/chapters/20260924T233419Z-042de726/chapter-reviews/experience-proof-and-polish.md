# Project chapter report: Experience Proof and Polish — interval unknown (baseline)

Run id: `run_6bc3a2a58e7046998938b2c770453196`  
Reviewed: 2026-09-24  
Scope: checkout HEAD `7ad9a41f99cf367240ab380b197226a122398820`

## Project synthesis

**Intended user improvement**: Players should experience a playable, responsive, recoverable, accessible, visually intact Allies versus Lessons game. Maintainers should detect regressions through repeatable release gates.

**Current user experience**: The repository contains substantial release-test infrastructure: two terminal scenarios, nine prompt families, keyboard and reduced-motion checks, and 17 visual references. Those mechanisms do not establish the complete current experience. The User reports missing transitions and difficulty navigating large action spaces; the expected Learn/Lesson interaction is also absent.

**Chapter conclusion**: The Project delivered useful regression machinery, but its full outcome remains unproven. Its performance contract is explicitly narrower than its KR. Historical implementation evidence cannot establish continuous or current satisfaction of the remaining claims.

**Decisive facts**:

- On 2026-07-16, commit `9f46fc9` recorded a passing release matrix: two terminal games covering nine prompt families.
- Current release-test documentation explicitly excludes stale/duplicate/checkpoint recovery and replay-equivalence claims.
- The performance baseline, recorded 2026-07-15 and still present at reviewed HEAD, measures warm development startup and renderer JavaScript heap, excluding required release-startup and broader memory measurements.
- On 2026-09-24, the User reported transition and action-selection friction despite the existing gates.

**Measurement fit**: These KRs provide useful technical safeguards. They do not directly measure whether a person can understand play, express their intended action comfortably, or finish a game willingly. Prompt-family coverage is also weaker than coverage of every reachable interaction.

**Next-chapter implication**: Likely **rewrite** — retain the regression infrastructure within the proposed consolidated Game Project, but connect acceptance to observed full-game usability. Unknown historical KRs should not automatically become new priorities.

## Objective evidence packet

**Coverage: incomplete** — all supplied current KRs are included, but the Project’s historical KR revisions, removed claims, establishment dates, and chapter boundaries could not be reconstructed.

**Expected KR rows: 4 · Actual KR rows: 4**  
**Verdict totals: 0 holds · 1 does not hold · 3 unknown**

The supplied [ledger](experience-proof-and-polish-ledger.json:1), observed at `2026-09-24T21:59:35.585349Z`, and the read-only `lf pm show --wave game --project experience-proof-and-polish --json` returned identical definitions and four KR claims. No current drift rows were found.

Repository-history searches found implementation history beginning in July, but no earlier exact Project KR ledger or merged chapter archive. Implementation dates are therefore evidence dates, not inferred promise dates.

This review inspected source, tests, committed references, retained receipts, and supplied human observations. No tests were executed: this checkout has no prepared Python environment, frontend dependencies, or retained browser results, and the run has no rendering environment. No code, planning objects, Tasks, or archive files were changed.

## Definition

> Prove that the shipped UR Lessons (hero) versus GW Allies (villain) experience remains playable, responsive, recoverable, accessible, and visually intact through repeatable gates against the real release stack, so regressions fail evidence checks instead of becoming subjective cleanup.

**Verdict**: unknown

**Evidence**: Release gates exist, but their documented exclusions and missing current outcome evidence prevent proof of the entire conjunction. The User’s dated reports of missing transitions and difficult action selection are counterevidence to treating technical gate success as sufficient evidence of the intended player experience. They do not, without an exact reproduced position, establish a specific keyboard, pixel-comparison, or recovery failure. See [human observations](../chapter-direction.md:33).

## KR ledger

### 1. A deterministic release-stack end-to-end suite reaches terminal across scenarios that collectively surface every prompt family reachable in UR Lessons versus GW Allies and proves reconnect/resume, stale-command rejection, duplicate-command idempotency, checkpoint recovery, and replay equivalence.

**Status and active interval**: current · established unknown; exact claim first supported by the supplied 2026-09-24 ledger → current PM read on 2026-09-24.

**Verdict**: unknown

**Evidence**:

- Commit `9f46fc9`, dated `2026-07-16T00:42:07Z`, records two terminal release games covering nine families. The current [matrix](../../../../frontend/e2e/release-prompt-matrix.json:230) pins seeds 51 and 62, with 24 and 148 commands respectively.
- The [release browser test](../../../../frontend/e2e/release-prompt-matrix.spec.ts:943) checks reconnect status, restored accessible prompts, terminal outcomes, and aggregate family coverage.
- Separate [server tests](../../../../tests/etude/test_server.py:165) exercise stale rejection and duplicate idempotency using custom Lightning Bolt decks. These are contribution evidence, not the complete selected-matchup release proof.
- A separate receipt introduced by commit `5416dd6` on 2026-07-18 compares 132 commands, 133 state checkpoints, and 798 viewer-projection checks across release WebSocket, headless, and replay surfaces. See [Authored Match Parity](../../../../conformance/authored-match-parity-v1/README.md:3). Its state checkpoints are comparison points; they do not establish restart recovery.
- The release matrix [explicitly excludes](../../../../docs/experience-proof.md:123) replay equivalence and stale/duplicate/checkpoint recovery claims. The presentation ledger is expressly [not a durable cross-process checkpoint](../../../../docs/architecture/presentation-runtime.md:88).

There is substantial partial evidence, but no inspected current result establishes the complete conjunction. No specific current release failure was reproduced.

### 2. A named reference device, OS, browser, and release-build profile has numeric budgets for cold launch-to-playable, input acknowledgement, authority response, animation frame pacing, and peak memory; repeatable measurements fail documented regression thresholds.

**Status and active interval**: current · established unknown; exact claim first supported by the supplied 2026-09-24 ledger → current PM read on 2026-09-24.

**Verdict**: does not hold

**Evidence**: On 2026-09-24, the checked-in [performance contract](../../../../frontend/e2e/experience-proof-baseline.ts:5) still identifies:

- A **development** stack and **warm** navigation measurement.
- Five warm launches and twenty interactions.
- Thresholds of 550 ms warm-launch p95, 100 ms acknowledgement p95, 650 ms authority-response p95, 34 ms frame-delta p95, and 20 MiB renderer heap.
- Explicit exclusions for cold/release startup and backend, browser-process, GPU, and asset memory.

The [measurement implementation](../../../../frontend/e2e/experience-proof.spec.ts:393) matches that narrower contract; it also stubs card art.

A separate [clean-machine contract](../../../../docs/clean-machine-play.md:41) supplies a 60,000 ms cold-playable threshold, but does not supply the missing unified release performance and peak-memory contract. It starts Vite rather than the built frontend preview.

This is a current contract-scope counterexample, not an assertion that measured performance exceeds its existing thresholds.

### 3. The complete selected matchup is operable keyboard-only and under reduced motion, with automated contrast and accessibility checks plus screen-reader assertions for every prompt, status transition, and game result.

**Status and active interval**: current · established unknown; exact claim first supported by the supplied 2026-09-24 ledger → current PM read on 2026-09-24.

**Verdict**: unknown

**Evidence**:

- Commit `06acd17`, dated `2026-07-16T01:13:05Z`, added keyboard activation, accessible instructions, focus management, reduced-motion handling, and accessibility audits.
- Current [prompt assertions](../../../../frontend/e2e/release-prompt-matrix.spec.ts:497) check accessible names/descriptions for every offered action encountered in the two scenarios.
- [Keyboard activation](../../../../frontend/e2e/release-prompt-matrix.spec.ts:744) uses Tab and Enter/Space. Accessibility, contrast, focus-boundary, and reduced-motion audits run on the first occurrence of each family; [terminal assertions](../../../../frontend/e2e/release-prompt-matrix.spec.ts:978) check result announcements and focus.
- These checks establish a substantial test mechanism, but the two scripted trajectories do not enumerate every reachable prompt instance or status transition. No current execution receipt was available here.

The checked PM flag is insufficient proof. Browser accessibility assertions are useful evidence; actual assistive-technology certification is explicitly excluded and is not silently assumed. The User’s large-action-space frustration challenges usability, but does not itself prove keyboard inoperability.

### 4. Versioned visual references cover every prompt family and key board, recovery, and terminal states; a clean release-build playthrough has zero unexpected visual diffs, broken curated assets, runtime console errors, or network-dependent play assets.

**Status and active interval**: current · established unknown; exact claim first supported by the supplied 2026-09-24 ledger → current PM read on 2026-09-24.

**Verdict**: unknown

**Evidence**:

- Commit `9f68843`, dated `2026-07-16T14:03:10Z`, introduced versioned release references. Current [v3 inventory](../../../../frontend/e2e/release-prompt-matrix.json:8) contains nine prompt references, three board references, three connection/recovery-status references, and two terminal references.
- The [release configuration](../../../../frontend/playwright.release.config.ts:19) permits zero differing pixels beyond its configured perceptual threshold and uses built frontend preview plus the actual backend.
- The test checks assets and runtime failures throughout both trajectories and verifies that every named reference was captured. The [CI workflow](../../../../.github/workflows/ci.yml:262) runs normal comparison separately from intentional baseline generation.
- The supplied September review records successful checks for PR #182’s earlier head. That is useful historical evidence, but not an inspected clean-playthrough result for HEAD `7ad9a41`.

The reference inventory is present. The current zero-failure playthrough outcome remains unverified; no broken reference or asset was demonstrated in this review.

## Shipped behavior

July repository history shows a progression from sampled development measurements to deterministic terminal release games, keyboard/accessibility checks, and versioned visual comparison. Maintainers gained reproducible gates for previously unmeasured surfaces.

Because the chapter start is unknown, these are dated shipped contributions, not a claimed start-to-finish chapter delta. Current human completion and satisfaction remain unproven.

## Task and PR receipts

The supplied ledger marks all six Tasks completed:

| Task | Contribution | Evidence or qualification |
|---|---|---|
| [ETU-1](https://linear.app/loopflow/issue/ETU-1/add-versioned-visual-references-to-the-release-prompt-matrix) | Visual references | `9f68843`; runtime still lists historical PR #93 open, conflicting with completed planning status. |
| [ETU-2](https://linear.app/loopflow/issue/ETU-2/prove-release-matrix-accessibility-across-the-selected-matchup) | Accessibility matrix | `06acd17`; implementation inspected. |
| [ETU-3](https://linear.app/loopflow/issue/ETU-3/prove-terminal-release-stack-prompt-family-matrix) | Terminal scenarios | `9f46fc9`; commit retains the two-game, nine-family pass report. |
| [ETU-4](https://linear.app/loopflow/issue/ETU-4/default-manabot-loopflow-tasks-to-codex) | Provider routing | Execution infrastructure; not player-outcome evidence. |
| [ETU-5](https://linear.app/loopflow/issue/ETU-5/establish-the-game-experience-proof-harness-and-budgets) | Initial measurement harness | `21aee29`, 2026-07-15; deliberately narrower than KR2. |
| [ETU-6](https://linear.app/loopflow/issue/ETU-6/route-etude-managed-workers-to-the-healthy-provider) | Provider recovery | Execution infrastructure; not player-outcome evidence. |

These receipts establish contributions, not complete KR satisfaction.

## Priority and ownership findings

Provider-routing Tasks served Loopflow execution reliability rather than this Project’s player-facing outcome. They are completed, so there is no evidenced active misplaced Task to transfer.

The human’s current priority is completing enjoyable games against trained challengers. Existing visual, accessibility, and recovery gates remain useful safeguards; pursuing universal certification as an independent portfolio would need renewed justification.

Recovery evidence crosses into the unavailable historical `experience-contract-and-recovery` Project. Its residual ownership cannot be resolved by treating this Project’s completed Tasks as proof that the recovery promise was fulfilled.

## Surprises

The prompt matrix correctly covers the implementation’s nine families while Learn remains implemented directly as `DiscardThenDraw` at [resolution.rs:381](../../../../managym/src/flow/resolution.rs:381). Thus complete coverage of implemented prompts can coexist with a missing expected mechanic. This does not prove a rules violation in every game without an eligible Lesson pool.

Likewise, accessible labels and stable screenshots can coexist with the User finding choices awkward and game progress hard to follow.

## Open work

No incomplete Task appears in this Project’s supplied or current PM list. Runtime records nevertheless mark all six historical worktrees missing and retain `ready`/blocked residue. ETU-1 also retains an unresolved historical PR record. These discrepancies were left untouched; they are not evidence of six active implementation efforts.

The supplied ledger additionally lists W2-183, W2-194, and W2-195 under the unavailable historical recovery Project. Their disposition remains unresolved and outside this review.

## Evidence gaps

- **Historical coverage:** exact earlier KR revisions, removed/closed claims, establishment dates, and chapter boundaries.
- **KR1:** current release evidence joining terminal coverage, command safety, checkpoint recovery, and replay equivalence.
- **KR2:** the required release profile and complete measurement contract, followed by repeatable readings.
- **KR3:** a declared denominator for reachable interactions/statuses and current keyboard, reduced-motion, and accessibility results against it.
- **KR4:** retained clean comparison and asset/network/error results bound to current HEAD.
- **Player outcome:** exact reproductions and human validation of transition and action-selection friction.

```json
{"1":"unknown","2":"does not hold","3":"unknown","4":"unknown"}
```

