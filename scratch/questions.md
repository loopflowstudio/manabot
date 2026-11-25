# MTG-483 questions and assumptions

This file replaces MTG-431's questions, which arrived through the stack sync.

## Assumptions taken during kickoff (2026-11-25, nobody present)

- Stacked this Task on MTG-431 because the pile-split reference exists only
  there. Undo: `lf sync main`.
- Pass 2 waits for MTG-431 to absorb main with MTG-441 (PR #613). MTG-431 and
  MTG-441 conflict in `Board.svelte`, `Lines.svelte` and `Plate.svelte`; that
  resolution is MTG-431's. Nothing was steered on MTG-431.
- No new drafted decision is built in this Task; the audit lands as a doc
  table and the list below.
- Rail test ids become `draft-raw-choices`, `draft-stopped`, `draft-stop` for
  every draft, replacing `per-creature-choices`, `combat-stopped`,
  `combat-stop`, `partition-stop`.
- Old draft storage keys are dropped without migration.

## Open

- Should MTG-431 be asked to sync main as soon as #613 merges, so pass 2 can
  start?
- After a disconnect mid-run the plan stops rather than resuming. Is resuming
  ever wanted?

## Follow-ups for ship-review to file or drop

- Draft Scry, Surveil and library order as one "arrange the looked-at cards"
  panel (class A).
- Draft multi-card picks: SelectCards with max > 1, SearchDistinctNames,
  OpponentSplitsRevealed, multi-card Discard, KeepPermanents `AllBut`
  (class A).
- Draft amounts: `target_shares` and DistributeCounters (class A).
- General read-only native preview (clone, step, fingerprint) for
  AssignCombatDamage, KeepPermanents by power and trigger ordering (class B).
- Offer a structured pile side so `partition.ts` stops matching label text.
- `discard` instruction text says "or decline when offered"; none is offered
  (`managym/src/decision_context.rs:516`).
- ChoosePile opponent view has no UI test; result of the pass 1 reading goes
  here.
