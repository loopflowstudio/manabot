# T4 — triggers, history and stack behavior (Task 488a40)

Brief: the Task directive (75 cards, `docs/rules/tla/allocation.jsonl`,
`planned_pr == "T4"`). Full plan history — slices 1–6, Jack Heart's accepted
design, concept and PR reviews (2026-10-30), the headless demo and gate record —
is at `45b70704a^:scratch/488a40-tla-t4-triggers.md`. The PR-submit scratch clear
removed it, and it is restored here condensed.

## Status (2026-10-30)

- Done: all 75 cards are defined in `tla.rs` with directed tests. Every accepted review
  decision is applied: one `Condition`/`Effect::If`, `Effect::CounterAbility` +
  `AbilityCountered`, and waiting delayed triggers as v5 relation frames kind
  10 with learning role `delayed_trigger` and LLM text. Synced with main's T3
  squash `f122d515b` (`6667baac0`). w17 digest-bound evidence and receipts were
  re-recorded at gate, and learning schema pins 9 → 10.
- Post-gate compress `e83d6c57c`: trigger occurrences are counted once, and the
  exact graveyard-source check is shared (`flow/triggers.rs`, `flow/resolution.rs`).
- Approval (Jack Heart, 2026-10-30): T4 is approved in full; publish and ship without
  another review.

## Remaining

- Republish. PR #407 closed 2026-10-31T01:50Z at the stale head `c57d8bf5b`,
  before the main sync, probably because its T3 base branch went away after T3's squash merge.
  Its red CI (Rust, Python, Semantic Kernel, Academy) ran on that pre-sync head
  and says nothing about HEAD. The branch is pushed at `e83d6c57c` and has no
  open PR. Open a new PR against `main`, let CI run, then ship under the standing
  approval.
- Deferred to CI: the Playwright release e2e and the full matrix.

Check (2026-10-30, HEAD `e83d6c57c`): `cd managym && cargo test --no-fail-fast`
(debug): all targets pass, 0 failed (cards 484).
