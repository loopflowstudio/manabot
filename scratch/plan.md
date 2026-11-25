# MTG-483 — One chunked-submission runner for drafted Academy decisions, and the audit of what to draft next

This file replaces MTG-431's plan, which arrived through the stack sync (see
Base). `scratch/concept-review.html` and `scratch/tour.yaml` are MTG-431's
notes, not this Task's; they are left untouched so later syncs do not conflict
on them. On a sync conflict in `scratch/plan.md` or `scratch/questions.md`,
this Task's version stays.

## Outcome

A person playing live in Academy drafts a multi-part decision whole and
confirms once. Today two decisions do that (combat declarations, Curator of
Destinies' pile split) through two hand-copied send loops in `Board.svelte`.
After this change there is one runner, one draft store and one stop/recovery
behavior, the pile split behaves exactly like combat when a step is refused or
the connection drops, and `docs/academy.md` says which other native decisions
are worth drafting and what each needs.

No engine, server or protocol change. No new drafted decision ships here.

## Base (choices made for review)

- **Stacked on MTG-431** (`lf sync jack/mtg-431-close-foundations-stack-copies-choices-costs`,
  merge `2647b11c85`). `partition.ts`, the partition loop in `Board.svelte`,
  and `Decision::PartitionPiles` / `ChoosePile` / `KeepPermanents` /
  `OrderEntryChoices` / `target_shares` exist only on that branch (PR #605,
  open). Without it there is one loop and nothing to unify.
  Passed over: basing on main + MTG-441 and porting the pile split later; that
  leaves the duplication the brief names for another Task. Undo with
  `lf sync main`.
- **MTG-441 (PR #613) must be underneath before `Board.svelte` or
  `Rail.svelte` is touched.** It rewrites the combat loop, adds the `pending`
  and `interaction` props, `drag.ts`, `addAttackers`/`setBlock`, the rail's
  collapsed disclosure and `e2e/fixtures/combat.ts`. It conflicts with
  MTG-431 in `Board.svelte`, `Lines.svelte` and `Plate.svelte`
  (`git merge-tree` of this branch with it, 2026-11-25), and that conflict
  belongs to MTG-431's sync with main, not to this Task ("Never `git merge` a
  Task's branch"). Pass 1 touches none of those files. Pass 2 starts with
  `lf sync` and checks that `Board.svelte` declares `interaction:` and
  `pending?:`. If it does not, pass 2 has not got its input: the build stops
  there and says MTG-431 has not yet absorbed main with #613.

Direction received for this plan, verbatim:

> check out MTG-441 (PR #613 ...) before finalizing the plan and building.
> ... Apply this same pattern to standardized chunked drafts rather than
> regressing it. ... the runner must wait until the board is answerable
> (`live && !pending`) instead of stopping after step 1. ... Keep draft state
> visible through pending Command, server work, and priority-stop updates
> (disable edits/Confirm until live). ... On same-revision refusal at step 0
> (`sending.at === 0`), preserve the draft so the user can edit and retry; on
> refusal after committed steps (`sending.at > 0`), clear the draft and
> surface exact partial progress. ... Account for MTG-441 landing on main
> (sync/merge cleanly and preserve all drag + combat draft behavior).

## What was found

Runner, as it stands on this base (`academy/ui/src/lib/board/Board.svelte`
289–313 combat, 367–391 partition):

- The two loops are the same code with different step types and words.
- Neither sees a refusal. `onact` returns nothing; `sendAction` returns
  `false` when a Command is pending, a resume is pending or server work is
  running (`socket.svelte.ts:379`), and a rejection only sets
  `gameStore.lastCommandRejection`. A refused or rejected step leaves
  "Declaring 2 of 5…" on screen until Escape.
- MTG-441 fixes this for combat only: `answerable = live && !pending`, a
  same-revision frame once answerable means the step was refused; at step 0
  the draft is kept, later it is cleared with "Stopped after k of n. The next
  declaration was not accepted." (`origin/jack/mtg-441…:Board.svelte`
  308–338). The partition loop would keep the old behavior after both land.
- Neither loop reacts to a disconnect. After a reconnect the recovered frame
  either has the sent revision (reads as a refusal) or a later one, in which
  case the loop keeps sending without the person having seen the board.
- A reload mid-send drops the run silently: the next frame is a new decision
  with no saved draft and no word about how far the plan got.
- MTG-441's rail folds the raw offers under `<details class="raw"
  data-testid="per-creature-choices">` with the words "Each creature's choice"
  whenever any draft snippet is passed. After both land, the pile split gets
  that combat wording.
- Rail's stop line is `data-testid="combat-stopped"` and the pile split
  already reuses it (`e2e/partition-piles.spec.ts:255`).
- Two storage keys, same shape: `academy.table.combat-draft`,
  `academy.table.partition-draft`, each `{match_id, frame_hash, …}`.

Native audit (`managym/src/flow/decision.rs` = D, `decision_context.rs` = C):

- `skip_trivial` answers any one-action space and loops
  (`flow/tick.rs:110-122`), so the last step of a sequence is often never
  asked: the last card of `OrderLibrary`, the last target's share, the
  `Decline` after every card is picked. A plan lists only the steps that will
  be asked; the runner finishes when its steps run out and never waits for a
  closing frame.
- Scry is not a cursor: every remaining card is offered with both sides each
  step and pick order is the final order (D:2218-2234, 3015-3027).
- `KeepCreatures` does not exist; it is `KeepPermanents{TotalCreaturePower}`.
  Cast-time shares are `PendingChoice::DivideAmongTargets`, and trigger
  ordering is `state.trigger_placement` with its own `ORDER_TRIGGERS` space;
  neither is a `Decision` variant.
- `combat_preview` (`flow/combat_preview.rs:104-199`) clones the game, steps
  the clone and returns `revision`, `fingerprint`, `action_index` per asked
  step. The clone-and-step core is not combat-specific; a general preview
  needs a per-decision plan matcher, a stop rule, and a rule that it stops
  before any step whose next frame depends on a shuffle, draw or opponent
  choice.
- The `discard` instruction says "or decline when offered" (C:516); no
  decline is ever offered (D:2166).

Classification the doc will carry (A = draft from the frame alone, B = needs
a read-only native preview, C = leave sequential):

| Decision (stage) | Class | Why |
|---|---|---|
| Scry (`scry`) | A, N≥2 | all looked cards in `inspected`; nothing commits until the last step |
| Surveil (`surveil`) then `order_library_top` | A, two phases | graveyard picks commit as one batch before ordering |
| OrderLibrary (`order_library_top`/`_bottom`) | A | N−1 asked steps; bottom shows only `cards[next..]` |
| SelectCards (`select_cards`, `private_search`) with max > 1 | A | candidates fixed, nothing moves until finish |
| SearchDistinctNames (`private_search`) | A | draft enforces distinct names; only picked and candidate cards are shown |
| OpponentSplitsRevealed (`split_revealed`) | A | each pick moves at once; all revealed cards visible |
| Discard (`discard`), 2+ cards | A | each discard commits and is public; cleanup discard is asked one card at a time and never states a total |
| DistributeCounters (`distribute_counters`) | A | one counter per step, repeats allowed, each commits |
| DivideAmongTargets (`target_shares`) | A | arithmetic; total = first frame's largest share + targets − 1; mode value is the share, offer index is offset by the minimum |
| KeepPermanents `AllBut` (`keep_permanents`) | A | public board, count limit |
| PartitionPiles (`partition_piles`) | A | shipped |
| AssignCombatDamage (`assign_combat_damage`) | B | asking order and lethal minimums are engine rules (`combat_actions.rs:176-262`); frame shows one pair |
| KeepPermanents by total power | B | layered power sums; may end without a decline |
| Trigger ordering (`order_triggers`) | B | modes and targets are asked between picks; pure only when no trigger has either |
| Retarget, ManaUpfront, ManaSpend, OrderEntryChoices, OrderReplacement | C | later options are legality- or planner-filtered, or answers interleave with other questions; usually one or two steps |
| LookAndSelect, ManifestDread, ChooseToExile, DiscardFromRevealedHand, OrderMeldedCards, ExileLinked, LegendRule, ChoosePile | C | one step (all `LookAndSelect` users have `max_select: 1`) |

Also multi-step and unlisted in the brief: `SearchNamed`, opening-hand
bottoming, `PendingChoice::Pay` (crew, convoke, waterbend), one-per-step
targets. Each commits per step; class C unless a later Task says otherwise.

ChoosePile: one step, so nothing to draft. The companion check is whether the
chooser sees the face-up cards and the face-down count; the prompt line exists
(`prompt.ts:72`) but no UI test or e2e covers the frame. Pass 1 reads a real
`choose_pile` frame (Curator of Destinies through the engine in a Python
session) and the doc records what it shows; a gap is a row in the follow-up
list, not a fix here.

## Design

Source of truth: the authority frame and its offers. A plan is an ordered
list of steps plus a function that finds, in a given frame, the frame's own
offer answering a step, or nothing. A plan comes from the frame (pile split)
or from a native preview (combat); the runner does not care which.

`academy/ui/src/lib/board/chunked.ts`, pure, no Svelte:

```ts
export interface ChunkedPlan<Step> {
  steps: readonly Step[];
  offerFor(frame: ExperienceFrame, step: Step): InteractionOffer | null;
}
/** Step `at` was sent at frame revision `sentAt`. */
export interface Run { total: number; at: number; sentAt: number }
export type Stop =
  | { why: 'refused'; sent: number }      // same revision once answerable
  | { why: 'unpredicted'; sent: number }  // next frame is not the planned step
  | { why: 'disconnected'; sent: number } // connection left 'connected' mid-run
  | { why: 'stopped'; sent: number }      // Escape / Stop sending
  | { why: 'reloaded'; sent: number; total: number };
export type Advance =
  | { kind: 'wait' }
  | { kind: 'send'; run: Run; offer: InteractionOffer }
  | { kind: 'done' }
  | { kind: 'stop'; stop: Stop; keepDraft: boolean };

export function begin<S>(plan: ChunkedPlan<S>, frame: ExperienceFrame): { run: Run; offer: InteractionOffer } | null;
export function advance<S>(run: Run, plan: ChunkedPlan<S>, frame: ExperienceFrame, table: { answerable: boolean; connected: boolean }): Advance;
export function stopWords(stop: Stop, total: number, words: DraftNouns): string;
```

Rules, each a unit test:

- Not answerable: wait. The draft stays drawn and its controls disabled.
- Disconnected while running: stop, `keepDraft: false`, "Stopped after k of
  n: the connection dropped. The board shows where the game stands."
- Same revision, answerable: refused. `at === 0` keeps the draft and says
  nothing beyond the table's own rejection message; `at > 0` clears it and
  reports `sent = at`.
- New revision, steps exhausted: done.
- New revision, `offerFor` finds nothing: unpredicted, `sent = at + 1`.
- New revision, offer found: send it, `at + 1`, `sentAt = revision`.
- Escape or Stop sending: `sent = at + 1`, "the rest was not sent".

`sent` counts steps the game accepted, except Escape and a disconnect, where
the last sent step's outcome is unknown; those two sentences say "sent", not
"accepted".

`academy/ui/src/lib/board/draft-store.ts`: one `sessionStorage` key,
`academy.table.draft`, holding `{ kind, match_id, frame_hash, value }` and,
while a run is sending, `{ kind, match_id, sent, total }`.
`saveDraft(kind, frame, value)`, `loadDraft(kind, frame, parse)` where `parse`
narrows unknown JSON to the consumer's draft type or null, `markRun`,
`takeInterruptedRun(match_id)`. A draft loads only for the same match, frame
hash and kind. A run marker found on mount for the same match becomes the
`reloaded` stop line once, then is removed. Storage failures lose the draft
and nothing else.

`Board.svelte` holds one `run`, one active plan, one effect calling `advance`,
one `stopSending`, one Escape branch, one "Stop sending" button and one
progress line built from the active draft's words ("Declaring 2 of 5…",
"Placing 2 of 5…"). `combat.ts` and `partition.ts` keep their draft types,
words and `stepOffer` / `partitionOffer`, and each exports the
`ChunkedPlan` for its steps.

`Rail.svelte`: the `draft` prop becomes `{ controls: Snippet; raw: { summary:
string; note: string } }`; the disclosure and the stop line take generic test
ids `draft-raw-choices` and `draft-stopped`. Combat passes "Each creature's
choice" / "Confirming the draft sends these one creature at a time."; the pile
split passes "Each card's choice" / "Confirming the split sends these one card
at a time." Replay and Study pass no draft and keep the full list with
recorded probabilities. Choice made for review: one pair of test ids for all
drafts rather than keeping `per-creature-choices` and `combat-stopped`;
MTG-441's and MTG-431's specs are updated in the same pass.

Invariants held (CLAUDE.md, "Academy's shared decision surface"): every step
sent is the current frame's own offer bound to that frame's revision; nothing
is sent before Confirm; raw offers stay reachable in "All legal choices" under
the disclosure; a sent step is never taken back; Escape, Space and the
disclosure work from the keyboard; drafts are offered only when
`interaction === 'play'`.

## Acceptance check at gate

```
cd academy/ui && npm run check && npm test \
  && npx playwright test e2e/chunked-drafts.spec.ts e2e/combat-presentation.spec.ts e2e/partition-piles.spec.ts e2e/drag-play.spec.ts e2e/keyboard-play.spec.ts
```

All pass. `chunked-drafts.spec.ts` drives the pile split (the consumer that
lacks MTG-441's fixes today) through a mocked socket:

1. Confirm a 5-card split; the server answers step 3 with a rejection at the
   same revision. The rail reads "Stopped after 2 of 5…", the panel shows the
   3 cards left, exactly 3 Commands were sent.
2. Confirm; the first Command is rejected. The drafted split is still on
   screen and Confirm works again.
3. Confirm; the socket closes after step 2 and reconnects with a later
   revision. No further Command is sent and the rail names the disconnect.
4. Confirm; reload after step 2. The rail reads "Stopped after 2 of 5: the
   page reloaded…" once.
5. While the split is open, `draft-raw-choices` is collapsed, reads "Each
   card's choice", and opening it lists both per-card offers.

## Absent and error states

- No preview answer (combat): no draft; the board keeps one creature at a
  time, as now.
- Frame is not a recognised draftable decision: no draft, raw offers only.
- Storage unavailable or malformed: empty draft, no run marker, no error.
- Recording, Replay, Study, read-only: no draft, no runner.
- Run stopped for any reason: the board shows the authority's current frame;
  a new draft starts from it.

## Exclusions

- Any new drafted decision (Scry, Surveil, library order, damage assignment,
  shares, …). The doc's table and a follow-up list in `scratch/questions.md`
  are what ship-review turns into Tasks.
- A general native preview; `combat_preview` is unchanged.
- `partition.ts` finding its two offers by label text (`FACE_UP_RE`); a
  structured pile field on the offer is a native change.
- The wrong `discard` instruction text (C:516).
- Drag gestures, the marquee and target lines from MTG-441.

## What goes

- Both send-loop effects, `stopSending` / `stopPartitionSending`, `running` /
  `partitionRunning`, and the two Escape branches in `Board.svelte`.
- `saveDraft` / `loadDraft` and `DRAFT_KEY` in `combat.ts`;
  `savePartitionDraft` / `loadPartitionDraft` and `DRAFT_KEY` in
  `partition.ts`; storage keys `academy.table.combat-draft` and
  `academy.table.partition-draft` (tab-scoped presentation; no migration).
- Test ids `combat-stopped`, `per-creature-choices`, `combat-stop`,
  `partition-stop` (one `draft-stop`).
- The three differently worded stop sentences per consumer.

Survives unchanged: combat and pile-split drafting, their words and panels,
`/combat-preview`, MTG-441's drag and draft behavior, reload-keeps-draft,
`stepOffer` and `partitionOffer` verification.

## Passes

1. **Runner, store and audit, with no shared-file edits.** `chunked.ts`,
   `chunked.test.ts`, `draft-store.ts`, `draft-store.test.ts`; the "Drafted
   decisions" section of `docs/academy.md` (contract, the table above, what a
   class B preview must guarantee, the ChoosePile reading); follow-up list in
   `scratch/questions.md`. This is the pass that can prove the plan wrong: if
   MTG-441's loop cannot be written as `advance` without a combat-only input,
   the interface changes here before anything depends on it. Shows:
   `npm test` green with every rule above as a case, including the MTG-441
   double `command_outcome` (not answerable, then answerable, new revision).
2. **Both consumers on the runner.** Needs MTG-441 underneath (see Base).
   `Board.svelte`, `Rail.svelte`, `combat.ts`, `partition.ts`, their tests and
   specs, `e2e/chunked-drafts.spec.ts`, the combat and Curator paragraphs of
   `docs/academy.md`. Removes everything under What goes. Shows: the
   acceptance command.

## Done when

- A refused, rejected, disconnected, reloaded or Escaped pile split stops with
  the same sentence shape and partial count as combat, and a first-step
  refusal keeps the split.
- `Board.svelte` has one send loop; `grep -c "sentAt" Board.svelte` finds one
  run, not two.
- One `sessionStorage` key holds drafts.
- `docs/academy.md` classifies every multi-step native decision and names
  which need a native preview.
- Combat click and drag drafting, and the pile split, pass their existing
  specs.
