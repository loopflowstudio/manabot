# T4 demo — headless, 2026-10-30

Experience reviewed: T4's 75 TLA cards playing through the real native env
(`manabot.managym.Env`, `skip_trivial=True`) at HEAD `8f1b383be`, PR #409
(open against `main`, mergeable; it reported no CI checks when this was
checked). The extension was rebuilt from HEAD first. The installed `.so`
predated the compress commit `e83d6c57c`.

No one was present, so the agent decided in Jack Heart's place. Jack Heart's
standing approval (2026-10-30, plan record at
`f54cda4ea:scratch/488a40-tla-t4-triggers.md`) covers T4 in full.

## Proof (`scratch/t4_demo.py`)

- `unsupported_cards` over all 75 T4 names (from `docs/rules/tla/allocation.jsonl`)
  returns `[]`. World is w17, card digest `89134543…aa76`.
- 40 seeded random-policy games ran, with four decks each holding a quarter of
  the T4 cards plus 8 of each basic. Result: 40 finished, 0 errors, 27.4 turns
  on average.
- Every continuation frame kind 1–10 appeared. That includes 2514 waiting
  delayed-trigger frames (kind 10) and 96 `ChooseTrigger` ordering decisions,
  across 54 distinct stack descriptions.
- An observation that has a waiting delayed trigger encodes through the
  current `ObservationSpace` (v5, which includes the `facts` block).
- Debug `cargo test --no-fail-fast` previously passed every target at
  `e83d6c57c`. Only scratch commits came after it.

## Decision

Advance. The implement pass meant to republish, and that is done: PR #409 is
open against `main`. The demo found no gap. Remaining: let CI run on #409
(the Playwright release e2e and the full matrix are deferred to it), then ship
under the standing approval.

## Gate re-pass, 2026-10-30 (HEAD `8f1b383be`)

Only `e83d6c57c` (a behavior-preserving refactor of `flow/triggers.rs` and `flow/resolution.rs`) changed tracked code since the last gate.
- `cargo fmt --check`: ok. `cargo clippy --all-targets`: no warnings.
- Debug `CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test --test rules_tests --test cards`: 484 + 140 passed, 0 failed (2 ignored).
- Other targets reuse the passing debug `cargo test --no-fail-fast` at `e83d6c57c`. PR #409 CI (full matrix, Playwright) is pending and owns those results.
