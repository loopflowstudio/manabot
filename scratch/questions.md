# MTG-483 questions and assumptions

This file replaces MTG-431's questions, which arrived through the stack sync.

## Assumptions taken during kickoff & concept review (2026-11-25, nobody present)

- Stacked this Task on MTG-431 because the pile-split reference exists only
  there. Undo: `lf sync main`.
- Pass 2 waits for MTG-431 to absorb main with MTG-441 (PR #613). MTG-431 and
  MTG-441 conflict in `Board.svelte`, `Lines.svelte` and `Plate.svelte`; that
  resolution is MTG-431's. Nothing was steered on MTG-431.
- No new drafted decision and no native/server/protocol change ships in this
  Task; the audit lands as a doc table (A1 / A2 / B / C) and the list below.
- Added `piloting?: boolean` (passed as `liveTable` from `+page.svelte`) and
  `rejected?: boolean` (passed as `gameStore.lastCommandRejection !== null`) to
  `Board.svelte`, plus `boolean | void` return on `onact`, so frame-derived
  drafts survive transient waits identically to combat and rejections are read
  from authority state rather than inferred from revision equality.
- Extracted `PartitionPanel.svelte` out of `Board.svelte` in Pass 2 to establish
  the per-decision panel pattern before Scry/Select/Distribute arrive.
- Rail test ids become `draft-raw-choices`, `draft-stopped`, `draft-stop`, and
  `draft-sending` for every draft, replacing `per-creature-choices`,
  `combat-stopped`, `combat-stop`, `partition-stop`.
- Old draft storage keys (`academy.table.combat-draft`,
  `academy.table.partition-draft`) are dropped without migration; the new single
  key `academy.table.draft` stores `{ draft?, run? }` so marking a run never
  clobbers a saved draft.

## Open

- Should MTG-431 be asked to sync main as soon as #613 merges, so pass 2 can
  start?
- After a disconnect mid-run the plan stops rather than resuming. Is resuming
  ever wanted?

## Follow-ups for ship-review to file or drop

- **Class A1 (ready on current frame):** Draft Scry, Surveil and OrderLibrary
  as one "arrange the looked-at cards" panel.
- **Class A2 (needs small structured fields on `DecisionContextView` /
  `InteractionOffer`, no clone-and-step preview):**
  - Expose structured `min`/`max`/`remaining`/pile-side/share fields so Academy
    never parses Rust `{:?}` text in `decision_context.pending` or `offer.label`
    regexes (`FACE_UP_RE` in `partition.ts`).
  - Draft multi-card picks: SelectCards with max > 1, SearchDistinctNames,
    OpponentSplitsRevealed, multi-card Discard, KeepPermanents `AllBut`.
  - Draft amounts: `target_shares` (`DivideAmongTargets`) and
    `DistributeCounters`.
- **Class B (needs general read-only clone-and-step native preview):**
  AssignCombatDamage, KeepPermanents by total creature power, and trigger
  ordering (`order_triggers`).
- `discard` instruction text says "or decline when offered"; none is offered
  (`managym/src/decision_context.rs:516`).
- `ChoosePile` opponent view has no UI test; result of the pass 1 real-frame
  reading goes here.
