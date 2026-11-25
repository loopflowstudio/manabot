# MTG-483 — One drafted-decision lifecycle, one chunked runner, and the audit of what to draft next

This file replaces MTG-431's plan, which arrived through the stack sync (see
Base). `scratch/concept-review.html` is now MTG-483's concept review as well;
`scratch/tour.yaml` remains MTG-431's and is left untouched. On a sync conflict
in `scratch/plan.md`, `scratch/questions.md`, or `scratch/concept-review.html`,
this Task's version stays.

## Outcome

A person playing live in Academy drafts a multi-part decision whole and
confirms once. Today two decisions do that (combat declarations, Curator of
Destinies' pile split) through two hand-copied send loops, two storage keys,
and inconsistent wait/refusal/keyboard handling in `Board.svelte`. After this
change:

1. **One drafted-decision contract and one chunked runner** govern every
   drafted decision (combat today, pile split today, and future drafts).
   `Board.svelte` owns one runner, one `Stop sending` control, one `Escape` /
   `Space` / `1–9` / auto-tap gate, and delegates panel UI to self-contained
   components (`PartitionPanel.svelte`).
2. **Table role is separated from transient submission readiness** (`piloting`
   vs `interaction`/`live`), so *both* combat and frame-derived drafts (pile
   split and future Class A drafts) stay visible with controls disabled across
   pending Commands, server work, and priority-stop updates, and disappear in
   Study, Replay, Watch, and Interrupted views.
3. **Authority rejections, disconnects, unexpected frames, reloads, and Escape**
   are handled identically and truthfully for every draft, using the authority's
   `CommandRejection` directly rather than guessing from revision equality.
4. **`docs/academy.md`** classifies every multi-step native decision into what
   can be drafted from structured frame data today (A1), what needs small
   structured frame fields instead of Rust `Debug` text scraping (A2), what
   needs a read-only clone-and-step native preview (B), and what stays
   sequential (C), plus the `ChoosePile` opponent-view verification.

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
- **MTG-441 (PR #613) must be underneath before `Board.svelte`, `Rail.svelte`,
  or `+page.svelte` is touched.** It rewrites the combat loop, adds the
  `pending` and `interaction` props, `drag.ts`, `addAttackers`/`setBlock`, the
  rail's collapsed disclosure and `e2e/fixtures/combat.ts`. It conflicts with
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

### Runner & Board lifecycle (current base + MTG-441)

- **Two duplicated loops in `Board.svelte`** (`289–313` combat, `367–391`
  partition) with duplicated stop handlers, Escape branches, Space branches,
  `Stop sending` snippets, and two `sessionStorage` keys
  (`academy.table.combat-draft`, `academy.table.partition-draft`).
- **Live-table role is conflated with transient readiness.** In MTG-441,
  `+page.svelte` collapses `liveTable` and `inputReady` into
  `interaction: 'play' | 'study' | 'read-only'`. When a Command is pending,
  server work runs, or priority stops sync, `interaction` drops to
  `'read-only'`. Combat stayed visible during those waits only because
  `+page.svelte` side-channeled `liveTable` through
  `combatPreview={liveTable ? fetchCombatPreview : undefined}`. Frame-derived
  drafts (partition and every future Class A draft) have no preview fetcher, so
  gating them on `interaction === 'play'` would hide the panel and pop open raw
  choices on every transient wait.
- **Guessing refusals from `revision === sentAt && answerable` fails on real
  server rejections.**
  - `stale_revision` rejections include a `recovery` frame
    (`server.py:1538, 2583`). If that frame has a newer revision, checking only
    `revision === sentAt` mistakes the rejection for an accepted step: on the
    last step it silently reports `done`; earlier it reports `unpredicted` with
    `sent = at + 1` (off by one) or keeps sending.
  - `not_actor` rejections omit `recovery` (`server.py:1550`). If a role change
    made `answerable === false`, checking only `revision === sentAt && answerable`
    leaves the runner stuck in `wait` until Escape.
  - `sendAction` already returns `boolean` (`socket.svelte.ts:364`) and
    `gameStore.lastCommandRejection` already records explicit rejections
    (`game.svelte.ts:147`, cleared on `beginCommand` and `applyFrame`, set after
    `applyRecovery` in `socket.svelte.ts:705-708`), but `Board.svelte` never
    sees either.
- **Disconnect and final-step ordering bugs if checked naively.**
  - Disconnecting sets both `connected = false` and `answerable = false`;
    checking `!answerable` before `!connected` prevents disconnect stop from
    ever firing.
  - When the final step (`at + 1 >= total`) is accepted and advances the
    revision while the opponent starts thinking (`work: 'running'`,
    `answerable = false`), checking `!answerable` before checking completion
    leaves `"Declaring N of N…"` and `"Stop sending"` on screen while the bot
    thinks.
- **Partition lacks keyboard & raw-choice parity.** Pressing `1–9` while the
  partition panel is open fires the raw per-card offer behind the panel's back
  (MTG-441 suppressed `1–9` for combat only). MTG-441's rail folds raw offers
  under `<details data-testid="per-creature-choices">` with hardcoded combat
  copy (`"Each creature's choice"`) and leaves raw offer rows clickable during
  an active run.
- **Reload mid-send is silent.** Neither loop persists an in-flight run marker,
  so reloading mid-sequence loses the count of steps already sent.
- **Board.svelte bloat.** Partition's panel markup, snippet, and CSS (~100
  lines) live inline inside a ~1,900-line `Board.svelte`. Extracting
  `PartitionPanel.svelte` sets the component pattern for every future drafted
  decision.

### Native audit (`managym/src/flow/decision.rs` = D, `decision_context.rs` = C, `agent/effect_execution.rs` = E)

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
- **Structured frame fields vs Rust `Debug` text (`E:267-281, 1528-1766`).**
  `effect_execution.rs` formats decision limits (`min`/`max`, `remaining`,
  `count`, `face-down pile size`) via `format!("{name}: {value:?}")` into
  `decision_context.pending: string[]`. Only `Scry`, `Surveil`, and
  `OrderLibrary` carry all needed state in typed frame fields today (`inspected`
  + `offers`). `SelectCards`, `SearchDistinctNames`, `OpponentSplitsRevealed`,
  `Discard`, `DistributeCounters`, `KeepPermanents AllBut` need small typed
  limit/remaining fields on `DecisionContextView` (not a clone-and-step preview)
  so the UI never parses Rust `{:?}` strings; `PartitionPiles` similarly needs a
  structured pile side on its offers to replace `FACE_UP_RE` regex matching.
- `combat_preview` (`flow/combat_preview.rs:104-199`) clones the game, steps
  the clone and returns `revision`, `fingerprint`, `action_index` per asked
  step. The clone-and-step core is not combat-specific; a general preview
  needs a per-decision plan matcher, a stop rule, and a rule that it stops
  before any step whose next frame depends on a shuffle, draw or opponent
  choice.
- The `discard` instruction says `"or decline when offered"` (C:516); no
  decline is ever offered (D:2166).

Classification the doc will carry:
- **A1** — draftable from structured frame fields today (no native change).
- **A2** — frame-derivable without a clone-and-step preview, once small typed
  count/limit/side fields are exposed on `DecisionContextView` or
  `InteractionOffer` (avoiding Rust `Debug` / label-regex scraping).
- **B** — needs a read-only clone-and-step native preview.
- **C** — leave sequential.

| Decision (stage) | Class | Why |
|---|---|---|
| Scry (`scry`) | A1, N≥2 | all looked cards in `inspected`; nothing commits until the last step |
| Surveil (`surveil`) then `order_library_top` | A1, two phases | graveyard picks commit as one batch before ordering |
| OrderLibrary (`order_library_top`/`_bottom`) | A1 | N−1 asked steps; bottom shows only `cards[next..]` |
| PartitionPiles (`partition_piles`) | A1 (shipped; A2 cleanup) | shipped via `inspected`; still matches `offer.label` with `FACE_UP_RE` until a structured pile side is offered |
| SelectCards (`select_cards`, `private_search`) with max > 1 | A2 | candidates fixed, nothing moves until finish; `min`/`max` only in `pending` Debug text today |
| SearchDistinctNames (`private_search`) | A2 | draft enforces distinct names; `remaining` only in `pending` Debug text today |
| OpponentSplitsRevealed (`split_revealed`) | A2 | each pick moves at once; `remaining` only in `pending` Debug text today |
| Discard (`discard`), 2+ cards | A2 | each discard commits and is public; `remaining` only in `pending` Debug text today (cleanup discard asks one at a time with no total) |
| DistributeCounters (`distribute_counters`) | A2 | one counter per step, repeats allowed, each commits; `remaining` only in `pending` Debug text today |
| DivideAmongTargets (`target_shares`) | A2 | arithmetic (`total = max_share + targets − 1`); relies on offer ordering/labels until share values are structured |
| KeepPermanents `AllBut` (`keep_permanents`) | A2 | public board, count limit; `count` only in `pending` Debug text today |
| AssignCombatDamage (`assign_combat_damage`) | B | asking order and lethal minimums are engine rules (`combat_actions.rs:176-262`); frame shows one pair |
| KeepPermanents by total power (`keep_permanents`) | B | layered power sums; may end without a decline |
| Trigger ordering (`order_triggers`) | B | modes and targets are asked between picks; pure only when no trigger has either |
| Retarget, ManaUpfront, ManaSpend, OrderEntryChoices, OrderReplacement | C | later options are legality- or planner-filtered, or answers interleave with other questions; usually one or two steps |
| LookAndSelect, ManifestDread, ChooseToExile, DiscardFromRevealedHand, OrderMeldedCards, ExileLinked, LegendRule, ChoosePile | C | one step (all `LookAndSelect` users have `max_select: 1`) |

Also multi-step and unlisted in the brief: `SearchNamed`, opening-hand
bottoming, `PendingChoice::Pay` (crew, convoke, waterbend), one-per-step
targets. Each commits per step; class C unless a later Task says otherwise.

ChoosePile: one step, so nothing to draft. The companion check is whether the
chooser sees the face-up cards and the face-down count; the prompt line exists
(`prompt.ts:72`) and `effect_execution.rs:1729-1734` emits `face-up pile` and
`face-down pile size`, but no UI test or e2e covers the frame. Pass 1 reads a
real `choose_pile` frame (Curator of Destinies through the engine in a Python
session) and the doc records what it shows; any gap is a row in the follow-up
list, not a fix here.

## Design

### Product model → code model

At any decision, a live player sees **at most one active Draft** (or none, for
single-step decisions). The lifecycle has four phases owned by one shared
mechanism:

1. **Eligible table (`piloting` vs `live`).** `Board.svelte` separates *table
   role* (`piloting?: boolean`, default `interaction === 'play'`) from *current
   submission readiness* (`interaction: 'play' | 'study' | 'read-only'`,
   `pending?: boolean`). `+page.svelte` passes `piloting={liveTable}`,
   `rejected={gameStore.lastCommandRejection !== null}`, and returns
   `sendAction`'s boolean from `onact`. Replay and Study omit `piloting` (false)
   and never open drafts. When `piloting` is true and `live` drops during a
   pending Command, server work, or stop-bar update, any open draft (combat,
   pile split, or future Class A/B) stays visible with edits and Confirm
   disabled, and raw offers stay folded.
2. **Pure chunked runner (`academy/ui/src/lib/board/chunked.ts`, no Svelte).**
   Source of truth is the authority frame and its offers. A plan is an ordered
   list of steps plus `offerFor(frame, step)` returning the frame's own offer
   or `null`. A plan comes from the frame (pile split) or from a native preview
   (combat); the runner does not care which.

```ts
export interface DraftCopy {
  /** Progress verb, capitalized: "Declaring", "Placing". */
  verb: string;
  /** Singular step noun: "declaration", "card placement". */
  item: string;
  /** Rail disclosure summary & note for raw offers. */
  raw: { summary: string; note: string };
  /** What the stop line says to do next: "Draft again from here.", "Split any remaining cards from here." */
  retryHint: string;
  /** Why an unpredicted frame stopped: "the preview did not predict" | "unexpected". */
  unexpectedPhrase: string;
}

export interface ChunkedPlan<Step> {
  steps: readonly Step[];
  offerFor(frame: ExperienceFrame, step: Step): InteractionOffer | null;
}

/** Step `at` was sent at frame revision `sentAt`. */
export interface Run {
  kind: string;
  total: number;
  at: number;
  sentAt: number;
  copy: DraftCopy;
}

export type Stop =
  | { why: 'refused'; sent: number }      // authority rejected or refused step `sent`
  | { why: 'unpredicted'; sent: number }  // next frame did not match planned step `sent`
  | { why: 'disconnected'; sent: number } // connection left 'connected' after `sent` sent
  | { why: 'stopped'; sent: number }      // Escape / Stop sending after `sent` sent
  | { why: 'reloaded'; sent: number; total: number };

export interface TableGate {
  connected: boolean;
  answerable: boolean; // live && !pending
  rejected: boolean;   // gameStore.lastCommandRejection !== null
}

export type Advance =
  | { kind: 'wait' }
  | { kind: 'send'; run: Run; offer: InteractionOffer }
  | { kind: 'done' }
  | { kind: 'stop'; stop: Stop; keepDraft: boolean };

export function begin<S>(
  kind: string,
  plan: ChunkedPlan<S>,
  frame: ExperienceFrame,
  copy: DraftCopy,
): { run: Run; offer: InteractionOffer } | null;

export function advance<S>(
  run: Run,
  plan: ChunkedPlan<S>,
  frame: ExperienceFrame,
  table: TableGate,
): Advance;

export function progressWords(run: Run): string;
export function stopWords(stop: Stop, total: number, copy?: DraftCopy): string;
```

Exact rule precedence in `advance` (every branch is a unit test):

1. **Disconnected (`!table.connected`):** stop immediately, `keepDraft: false`,
   `{ why: 'disconnected', sent: run.at + 1 }`. Checked first because disconnect
   also makes `answerable` false.
2. **Explicit rejection (`table.rejected`):** stop immediately,
   `{ why: 'refused', sent: run.at }`,
   `keepDraft: run.at === 0 && frame.revision === run.sentAt`. Handles
   same-revision rejections, newer-revision `stale_revision` recovery frames,
   and non-recovering `not_actor` rejections without hanging or miscounting.
3. **Final step accepted (`frame.revision !== run.sentAt && run.at + 1 >= plan.steps.length`):**
   `{ kind: 'done' }` immediately — even if `!table.answerable` because the
   opponent's bot started thinking on the new frame.
4. **Same revision, now answerable (`frame.revision === run.sentAt && table.answerable`):**
   fallback refusal detection when no explicit `CommandRejection` arrived
   (`{ why: 'refused', sent: run.at }`, `keepDraft: run.at === 0`).
5. **Not yet answerable (`!table.answerable`):** `{ kind: 'wait' }`. Keeps the
   draft drawn with edits/Confirm disabled through `commandPending`, server
   work (`command_outcome` with `work: running` before `work: complete`), and
   `stopsPending`.
6. **New revision, answerable, more steps remain (`next = run.at + 1 < plan.steps.length`):**
   - `const offer = plan.offerFor(frame, plan.steps[next])`.
   - If `!offer`: `{ kind: 'stop', stop: { why: 'unpredicted', sent: next }, keepDraft: false }`.
   - If `offer`: `{ kind: 'send', run: { ...run, at: next, sentAt: frame.revision }, offer }`.

Honest count wording in `stopWords`:
- `refused`: at `sent === 0`, empty string (table's own rejection notice already
  explains it, and on same-revision refusal the draft is kept; if `sent === 0`
  with `!keepDraft` because the revision moved on, `"The board moved on before
  the first ${copy.item} was accepted. ${copy.retryHint}"`). At `sent > 0`:
  `"Stopped after ${sent} of ${total}. The next ${copy.item} was not accepted. ${copy.retryHint}"`.
- `unpredicted`: `"Stopped after ${sent} of ${total}: the game asked something ${copy.unexpectedPhrase}. The board shows where it stands; ${copy.retryHint}"`.
- `disconnected`: `"Stopped after ${sent} of ${total} sent: the connection dropped. The board shows where the game stands."` (explicitly says *sent*, because the last sent step's outcome is unknown until recovery).
- `stopped`: `"Stopped after ${sent} of ${total} sent; the rest was not sent."`
- `reloaded`: `"Stopped after ${stop.sent} of ${stop.total} sent: the page reloaded. The board shows where the game stands."` (consumer-agnostic so it works even when the reloaded frame has already moved to a different action space).

3. **Unified tab-scoped persistence (`academy/ui/src/lib/board/draft-store.ts`, no Svelte).**
   One `sessionStorage` key, `academy.table.draft`, storing a single JSON
   envelope `{ draft?: SavedDraft; run?: InterruptedRun }`:
   - `SavedDraft`: `{ kind: string; match_id: string; frame_hash: string; value: unknown }`
   - `InterruptedRun`: `{ kind: string; match_id: string; sent: number; total: number }`
   Keeping `draft` and `run` as sibling fields in the single envelope means
   `markRun` never clobbers the saved draft (preserving reload continuity after
   a step-0 refusal) and `clearRun` never clobbers the draft.
   API: `saveDraft(kind, frame, value)`, `loadDraft(kind, frame, parse)`,
   `clearDraft(kind?)`, `markRun(run, matchId)`, `clearRun()`,
   `takeInterruptedRun(matchId)`.
   - A draft loads only for the same `kind`, `match_id`, and `frame_hash`, and
     only if `parse(value)` returns non-null.
   - `takeInterruptedRun(matchId)` removes any stored `run`, returning it only
     if `run.match_id === matchId` (so switching games discards stale markers).
   - Storage failures lose only the draft/marker and never throw.

4. **One active draft surface in `Board.svelte` + extracted `PartitionPanel.svelte`.**
   - `Board.svelte` owns:
     - One `run` + `activePlan` + `stopNote` + `startRun(kind, plan, copy, onClear)` + `stopRun()`.
     - When `startRun` calls `onact(offer)` and `onact` returns `false`, it does
       not enter `run` (at step 0 it leaves the draft intact; at step > 0 in the
       effect, `onact` returning `false` stops with `refused, sent: run.at`).
     - One `$effect` calling `advance(run, activePlan, frame, { connected, answerable, rejected })`, updating `markRun` on every send and calling `clearRun` on `done`/`stop`.
     - One `onMount` / match-id check calling `takeInterruptedRun(frame.match_id)` to surface `reloaded` once.
     - One `activeDraft` derivation (`combat` | `partition` | `null`) that drives:
       - `line` and `hint`
       - `Space` confirm, `Escape` stop-sending, `1–9` raw-key suppression, and
         `autoTap` suppression for *every* draft uniformly
       - One `sendingControls` snippet (`data-testid="draft-sending"`,
         `data-testid="draft-stop"`) shared by all running drafts
       - `Rail`'s `draft` prop (`{ controls, raw, sending }`)
   - `combat.ts` and `partition.ts` each export their `ChunkedPlan` factory and
     `DraftCopy` constant (`COMBAT_COPY`, `PARTITION_COPY`), plus their pure
     draft helpers and `stepOffer` / `partitionOffer` verification.
   - `PartitionPanel.svelte` holds the foldable two-pile overlay panel and its
     CSS, keeping `Board.svelte` clean and establishing the template for future
     drafted panels (Scry, SelectCards, Distribute, etc.).
   - `Rail.svelte`: `draft?: { controls: Snippet; raw: { summary: string; note: string }; sending?: boolean }`; disclosure uses `data-testid="draft-raw-choices"`, stop line uses `data-testid="draft-stopped"`, and raw choice buttons set `disabled={!live || Boolean(draft?.sending)}` so clicking an open raw list cannot race an in-flight chunked run. Replay and Study pass no draft and keep the full list with recorded probabilities.

### Choices made for review

- **Add `piloting?: boolean` and `rejected?: boolean` to `Board.svelte` (and `boolean | void` return on `onact`)** rather than changing `interaction`'s union or smuggling table role via `combatPreview`. Preserves MTG-441's `interaction`, `pending`, `canDrag`, and `data-live` semantics intact with zero Replay churn.
- **One set of generic rail test ids (`draft-raw-choices`, `draft-stopped`, `draft-stop`, `draft-sending`)** across all drafts, updating MTG-441's and MTG-431's specs in Pass 2 rather than keeping combat-named ids on non-combat drafts.
- **Extract `PartitionPanel.svelte` in Pass 2** rather than leaving partition's panel inline in `Board.svelte`.
- **Split Class A into A1 (structured frame today) and A2 (needs structured frame fields, no clone-and-step preview)** in `docs/academy.md` so future Tasks don't scrape Rust `Debug` strings or overbuild clone-and-step previews.
- **Drop old storage keys (`academy.table.combat-draft`, `academy.table.partition-draft`) without migration** (tab-scoped ephemeral drafts).

Invariants held (`AGENTS.md`, "Academy's shared decision surface"): every step
sent is the current frame's own offer bound to that frame's revision; nothing
is sent before Confirm; raw offers stay reachable in "All legal choices" under
the disclosure; a sent step is never taken back; Escape, Space and the
disclosure work from the keyboard; drafts are offered only when `piloting` is
true (live play, never Study/Replay/Watch/Interrupted).

## Acceptance check at gate

```
cd academy/ui && npm run check && npm test \
  && npx playwright test e2e/chunked-drafts.spec.ts e2e/combat-presentation.spec.ts e2e/partition-piles.spec.ts e2e/drag-play.spec.ts e2e/keyboard-play.spec.ts
```

All pass. `chunked-drafts.spec.ts` drives the pile split (the consumer that
lacks MTG-441's fixes today) through a mocked socket:

1. Confirm a 5-card split; the server answers step 3 with a rejection at the
   same revision. The rail reads `"Stopped after 2 of 5…"`, the panel shows the
   3 cards left, exactly 3 Commands were sent.
2. Confirm; the first Command is rejected at the same revision. The drafted
   split is still on screen and Confirm works again.
3. Confirm; the first Command is rejected with `stale_revision` and a newer
   recovery frame. The stale draft clears and no second Command is sent.
4. Confirm; the accepted frame arrives with `work: running` before
   `work: complete`. The panel stays visible with controls disabled and
   advances once work completes; after the last step advances with
   `work: running`, the run finishes immediately without leaving `Stop sending`
   on screen.
5. Confirm; the socket closes after step 2 and reconnects with a later
   revision. No further Command is sent and the rail names the disconnect
   (`"Stopped after 2 of 5 sent: the connection dropped…"`).
6. Confirm; reload after step 2. The rail reads `"Stopped after 2 of 5 sent:
   the page reloaded…"` once.
7. While the split is open, `draft-raw-choices` is collapsed, reads `"Each
   card's choice"`, pressing `1` sends nothing, and opening the disclosure
   lists both per-card offers.

## Absent and error states

- No preview answer (combat): no draft; the board keeps one creature at a
  time, as now.
- Frame is not a recognised draftable decision: no draft, raw offers only.
- Storage unavailable or malformed: empty draft, no run marker, no error.
- Recording, Replay, Watch, Study, Interrupted, read-only: `piloting` is false,
  no draft, no runner.
- Run stopped for any reason: the board shows the authority's current frame;
  a new draft starts from it (except same-revision step-0 refusal, which keeps
  the draft for editing and retry).

## Exclusions

- Any new drafted decision (Scry, Surveil, library order, damage assignment,
  shares, …). The doc's table and the follow-up list in `scratch/questions.md`
  are what ship-review turns into Tasks.
- Native changes: no general native preview (`combat_preview` is unchanged), no
  A2 structured count/limit fields on `DecisionContextView`, no structured pile
  field on `partition_piles` offers, no fix to the wrong `discard` instruction
  text (C:516).
- Drag gestures, the marquee and target lines from MTG-441.

## What goes

- Both send-loop effects, `stopSending` / `stopPartitionSending`, `running` /
  `partitionRunning`, and the two Escape/Space branches in `Board.svelte`.
- Inline partition panel markup/CSS from `Board.svelte` (moved into
  `PartitionPanel.svelte`).
- `saveDraft` / `loadDraft` and `DRAFT_KEY` in `combat.ts`;
  `savePartitionDraft` / `loadPartitionDraft` and `DRAFT_KEY` in
  `partition.ts`; storage keys `academy.table.combat-draft` and
  `academy.table.partition-draft`.
- Test ids `combat-stopped`, `per-creature-choices`, `combat-stop`,
  `partition-stop` (replaced by `draft-stopped`, `draft-raw-choices`,
  `draft-stop`, `draft-sending`).
- The duplicated per-consumer stop sentences.

Survives unchanged: combat and pile-split drafting, their words and panels,
`/combat-preview`, MTG-441's drag and draft behavior, reload-keeps-draft,
`stepOffer` and `partitionOffer` verification.

## Passes

1. **Runner, store and audit, with no shared-file edits.** `chunked.ts`,
   `chunked.test.ts`, `draft-store.ts`, `draft-store.test.ts`; the `"Drafted
   decisions"` section of `docs/academy.md` (contract, the A1/A2/B/C table,
   what A2 structured fields and a Class B preview must guarantee, the real
   `ChoosePile` frame reading); follow-up list in `scratch/questions.md`.
   This is the pass that proves the state machine before touching shared UI
   files: `npm test` green with every rule in `advance` (disconnect before
   unanswerable, explicit rejection at step 0 same-rev vs newer-rev vs step > 0
   vs unanswerable `not_actor`, final step completing while unanswerable,
   MTG-441 double `command_outcome` wait-then-send, unexpected frame, Escape,
   reload) and every `draft-store` envelope case (run marker does not clobber
   saved draft, match_id scoping, storage error resilience).
2. **Both consumers on the runner + `PartitionPanel.svelte` + `piloting`/`rejected` props.**
   Needs MTG-441 underneath (see Base). `+page.svelte`, `Board.svelte`,
   `PartitionPanel.svelte`, `Rail.svelte`, `combat.ts`, `partition.ts`, their
   tests and specs, `e2e/chunked-drafts.spec.ts`, the combat and Curator
   paragraphs of `docs/academy.md`. Removes everything under What goes. Shows:
   the acceptance command.

## Done when

- A refused (same-rev, newer-rev, or non-recovering), disconnected, reloaded,
  unexpected, or Escaped pile split stops with the same semantics and partial
  count as combat; a same-revision step-0 refusal keeps the split; transient
  waits (pending Command, server work, stop-bar sync) keep both combat and
  pile-split drafts visible and disabled.
- `Board.svelte` has one send loop (`grep -c "sentAt" Board.svelte` is 0 or 1,
  with the state machine in `chunked.ts`) and delegates the partition overlay
  to `PartitionPanel.svelte`.
- One `sessionStorage` key (`academy.table.draft`) holds drafts and in-flight
  run markers without clobbering each other.
- `docs/academy.md` classifies every multi-step native decision into A1 / A2 /
  B / C and records the `ChoosePile` opponent-view verification.
- Combat click and drag drafting, and the pile split, pass all specs.
