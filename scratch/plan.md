# MTG-441 — Drag interactions on the Academy board

## Outcome

In live play a person can move things with the pointer where today they can
only click:

1. **Hand → battlefield (`play`).** Press a playable card in hand, drag it,
   and the card follows the pointer. The lands the automatic payer would tap
   are outlined as they are on hover, and the battlefield shows what a drop
   does ("Cast Lightning Bolt", "Play Island", or "Choose how to play …" when
   the card has several offers). Release over the battlefield and it is played
   or cast; release over the hand, the rail or anywhere else, or press Esc,
   and it goes back with nothing sent.
2. **Source → recipient (`connect`).** Pull a live line from an acting card to
   a legal recipient:
   - During a block draft, drag a declarable blocker onto an attacker it can
     block (`block` line); dropping assigns the block, or moves it when the
     blocker already had one. Click-click still works.
   - During a targeting prompt or a one-at-a-time `DECLARE_BLOCKER` fallback,
     drag from any rendered source of the decision — an announcing or asked
     stack entry in the rail, or an asked permanent on the battlefield — onto
     a legal permanent, player plate or stack entry (`target` line, or `block`
     line for `DECLARE_BLOCKER`).
3. **Marquee → attackers (`marquee`).** During an attack draft, drag a
   rectangle across the battlefield rows; every declarable creature touching
   the box highlights live as you sweep and joins the drafted attack on
   release.

Every existing click, key, accessible name and Command binding stays exactly
as it is. Drag adds no capability: each drop ends in a code path a click
already takes.

## The one design

### A drop is a click somewhere else

The decision surface rule (AGENTS.md, "Academy's shared decision surface")
forbids hiding choices or selecting actions silently. The design meets it by
construction: every drop calls the same action path a click calls.

| Gesture | Drop resolves to | Existing path |
|---|---|---|
| `play` (hand card on battlefield) | `activate(card.name, item.offers, handButton, dropRect)` | `Board.svelte:570` — one offer commits, several open the card menu at `dropRect` while returning focus to `handButton` on Cancel |
| `connect` (blocker on attacker, draft) | `draft = setBlock(draft, blocker, attacker); armed = null` | `combat.ts` — idempotent block assignment (click still takes back via `toggleBlock`) |
| `connect` (source on target / 1-by-1 block) | `activate(title, offersAtTarget, targetElement)` | the click on that permanent, player plate or stack entry |
| `marquee` (attack draft) | `draft = addAttackers(draft, hits)` | `combat.ts` — union of newly caught declarable creatures into `draft.attackers` |

A hand card with two offers (e.g. cast + kicked cast, or cast + cycle) never
picks one: the drop opens the same menu the click opens, anchored at the drop
point. Nothing is sent that `onact` would not have sent for a click, bound to
the same frame revision.

A card in hand starts a `play` drag only when at least one of its offers has
verb `cast` or `play_land` (`types.ts:366`). Discard or "choose a card" offers
on hand cards stay click-only. Tray choices (cards outside the board) stay
click-only.

### Built gestures

- **Hand play.** The revision-bound gesture in `board/drag.ts` carries a play while its Command is pending or its menu is open. An invalid or refused drop settles back to hand.
- **Target and blocker connections.** The same gesture carries the line. A draft recipient changes the block assignment; an offer recipient opens or commits the same offers as its board control. A snapped single offer supplies the payment preview. Standing and pointer lines share endpoint measurement, including cardless stack entries.

The board's `interaction` distinguishes play, Study retries and read-only positions. Only play starts a gesture. During a block drag, the dragged blocker determines the legal highlights and its standing line is hidden. The click-selected blocker survives cancellation. Release hit-tests the final pointer coordinates. A consumed drag click stays suppressed until that click or the next primary press.

### Joint-combat lifecycle and connect affordances (Pass 5)

Post-build review of `HEAD` (`437101a9fe`) found two conceptual mismatches and two minor `play`-drag edge cracks to resolve before gate:

1. **Separate live-play table mode from millisecond `inputReady` waits.**
   - Today `+page.svelte` ties `combatPreview` to `canSubmitLive` (which drops during `commandPending`, `work.running`, and `stopsPending`), `Rail.svelte` folds per-creature offers on `{#if draft && live}`, and `Board.svelte` drops `draftMarks` and restores `asked` while `running`. Confirming a draft therefore flashes the closed "Each creature's choice" list open and pops single-creature `asked` outlines between every step; clicking a stop in the Stop Bar mid-draft tears down the draft and clears `armed`; and a same-revision Command rejection wedges `running` forever because the runner ignores `revision === sending.sentAt` without watching `pending`.
   - **Target model:**
     - `+page.svelte`: pass `combatPreview` whenever the table is an active live-play table (`gameActive && pilotCanAct && !participantStudying && !gameStore.branchProjection`), independent of transient `inputReady` waits (`commandPending`, `work.running`, `stopsPending`). Keep `interaction` (`'play' | 'study' | 'read-only'`) gating actual submissions and pointer drags.
     - `Board.svelte`: `declaring = combatPreview !== undefined && isDeclaration(frame)`. Keep `drafting` and `running` as one steady joint surface (`joint = drafting || running`): suppress single-creature `asked` whenever `joint` is true, keep drafted creatures marked while `running`, disable Confirm/Clear/pile-steppers and disallow draft edits (`onDraft`, `confirmDraft`, `Space`) when `!live`, and let the combat-preview fetch run while `declaring && !running`.
     - `Board.svelte` runner: when `revision === sending.sentAt` and `!pending && live` (a step was rejected at the same revision or failed to send), clear `running`; keep `draft` intact when `sending.at === 0` (nothing committed yet), or clear `draft` with a `stopNote` when `sending.at > 0` (earlier steps already committed).
     - `Rail.svelte`: fold per-creature offers on `{#if draft}` alone (never `draft && live`). Replay and Study never pass `draft`, so they stay unchanged; live joint combat never flashes the raw numbered list during Confirm, `work: running`, or stop updates.
2. **Show exact legal permanent recipients during a `connect` drag.**
   - Today `.tcard.attacking` (`BoardCard.svelte:370`) overwrites `.tcard.legal`'s `box-shadow`, and permanent targets have no outline while dragging a line (only `Plate` and `Rail` draw a `--legal` ring), so QA observed that players had to fish around to discover legal targets on lands and attackers.
   - **Target model:** Derive `connectTargets: ReadonlySet<number>` in `Board.svelte` for the active `connect` drag — the exact `can_block` attacker IDs during a block-draft drag (never other blockers), or permanents in `fieldById` with `offers.length > 0` during targeting / previewless `DECLARE_BLOCKER`. Pass `connectTarget` through `Field.svelte` to `BoardCard.svelte` (`data-connect-target`), styled in `Board.svelte` with a dashed `--accent` outline (`button[data-connect-target]:not([data-drop])`) that upgrades to solid `--accent` on `data-drop`. Also let `.tcard.legal` compose with `.tcard.attacking` instead of being clobbered.
3. **Close two `play`-drag edge cracks.**
   - `promote(gesture, offerId)` in `drag.ts` / `Board.svelte`: promote `holding('menu')` to `'pending'` only when `offerId` belongs to the held card's `offers`; if the player commits a different offer while the menu is open, settle the held card home instead.
   - `overField` in `Board.svelte`: match `!!hit?.closest('.board .half, .board .tray')` so an open `.tray` panel during Priority does not punch a "Release to cancel" hole in the middle of the drawn `.drop-zone`.

### Marquee attacker draft (`marquee`)

- Active only when `drafting && preview.kind === 'attackers'`.
- Starts on a press anywhere inside battlefield `.rows` (both halves' `.rows`,
  empty space or permanent card — never `.plate`, `.hand`, `.rail` or bars).
  `.rows` takes `touch-action: none` while an attack draft is open.
- **Live hit preview (choice made for review):** On every `pointermove`,
  compute the rectangle from `start` to `pointer` and intersect it against all
  declarable creatures' `[data-testid=board-card][data-object-id]` rects.
  Those `hits` receive the `drop` highlight while sweeping so the person sees
  which cards (including individual fanned copies in a pile) are caught before
  releasing.
- On `pointerup`, `draft = addAttackers(draft, hits)` (`combat.ts`) adds all
  caught declarable creatures not already drafted. It only adds; "Clear the
  draft" and clicks remove.

**Choice made for review — the marquee selects only into a draft.** The brief
says "(or selecting lands/cards)". Attackers are the only multi-selection the
shared contract drafts before committing. Tapping a land for a cost and
choosing a card are each one Command, committed when sent, and the next frame
is not predicted by any preview. A marquee over lands would chain unconfirmed,
irreversible commits, which the decision-surface rule excludes ("Cancellation
or undo needs an explicit shared action and cannot erase committed costs").
Noted in `scratch/questions.md`.

### Motion and reduced motion

Following the pointer is direct manipulation, not animation, and stays under
reduced motion. What reduced motion removes: the ghost's return-to-hand
transition (it snaps straight back to `idle`), lift transitions, and `Travel`
after the drop (already gated by `prefersReducedMotion()`). The release proof
config runs every spec with `reducedMotion: 'reduce'`
(`playwright.release.config.ts`); one E2E test runs with `no-preference`.

Visuals use existing `DESIGN.md` tokens only (`design.test.ts` lints all board
styles): the ghost uses `--lift-selected`; snapped drop targets and marquee
hits use an `--accent` outline; the drop zone and marquee use a dashed
`--accent` edge (softening to `--border-strong` when outside the drop zone),
with an opaque `--bg-surface` label pill.

## Acceptance check at gate

From `academy/ui`, with frontend generation finished first:

```bash
npm test && npm run check
ACADEMY_API_PORT=8141 ACADEMY_FRONTEND_PORT=5341 npm run test:e2e -- drag-play.spec.ts combat-presentation.spec.ts keyboard-play.spec.ts mana-payment.spec.ts
npm run test:e2e:release
```

`e2e/drag-play.spec.ts` (new; added to `testMatch` in
`playwright.release.config.ts`) drives real `page.mouse` down/move/up against
the controlled WebSocket authority pattern from `combat-presentation.spec.ts`
and asserts on received Commands:

- **Hand drag (1 offer):** Drag a castable card from hand to the battlefield:
  planned lands are outlined mid-drag with zero Commands, Inspect does not pop
  up mid-drag, and the drop label names the offer; releasing over the
  battlefield sends exactly one Command with that offer id at the frame's
  revision (and holds the ghost until accepted, or returns it cleanly on
  rejection without revision change). Releasing over the hand and pressing Esc
  send none, leave the card in hand, and do not swallow the next real click.
- **Hand drag (2+ offers):** Drop label says "Choose how to play …"; releasing
  over the battlefield opens the card menu at the drop point with zero
  Commands; picking an item commits it, and Cancel returns the ghost and
  restores focus to the hand card.
- **Block draft drag:** Two attackers A and B: drag blocker onto A, then
  reassign by dragging onto B (only one line from that blocker mid-drag; no
  Command until Confirm; Esc mid-drag restores A's line and does not clobber
  click-click `armed`). Confirm sends the previewed steps.
- **Targeting drag:** Build explicit targeting frames (note:
  `protocol/fixtures/bolt-target.json` has no `decision_context` or
  `stack_objects`, so construct frames with `decision_context`!) covering:
  1. Announcing spell in rail's stack → creature permanent, → opponent's plate,
     and → a stack entry (including a non-spell stack ability).
  2. Battlefield permanent source (`stage: 'trigger_targets'`) → creature.
  3. One-at-a-time `DECLARE_BLOCKER` fallback (without combat preview) →
     attacker (`block` line style).
- **Attack draft marquee:** Sweep across two declarable creatures (asserting
  live hit highlight mid-sweep and no marquee starting on `.plate`/`.hand`),
  release drafts both, Confirm sends the previewed steps.
- **Click & keyboard parity + reduced motion:** Short clicks (< 6 px) on every
  drag source still act as clicks; one test runs with `reducedMotion: 'no-preference'`;
  existing specs pass unchanged.

## Affected surfaces

- Live play board (`/`), mouse, pen and touch.
- Recordings and Watch (`live` false) and study positions: no drags start;
  unchanged.
- Keyboard: unchanged. The keys sheet gains one plain row describing drag,
  and the block draft's caption (`Board.svelte:918`) and "No blocks" plan
  (`combat.ts:162`) say "drag a blocker onto an attacker, or click each".
- `academy/ui/DESIGN.md`: the layering sentence.
- Protocol, server, Command shape, storage keys: untouched.

## Docs

Written and committed (`835983d8fa`): `docs/academy.md`, `academy/ui/DESIGN.md`
(layering incl. `--layer-drag`), `managym/academy/README.md`. Update them only
if passes 2–3 change what they describe.

## Absent and error states

- Press on a card that is not a source: an ordinary click.
- Release over something that is not a legal target: nothing sent, ghost
  returns, click-click `armed` preserved on cancel.
- New revision, disconnect or read-out mid-drag: `pressed`/`dragging` ends
  with nothing sent; `holding` clears on the revision transition.
- Combat preview unavailable: no draft, so no blocker-draft drag or marquee;
  the one-at-a-time `DECLARE_BLOCKER` prompt still gets the `connect` (`block`)
  drag.
- Command rejected after a hand drop: `pending` clears with revision
  unchanged, ghost returns to hand, existing error notice explains why.

## Exclusions

- Casting and targeting in one gesture (dropping a hand card on its target):
  two Commands with an unpredicted frame between them.
- Marquee over lands or cards outside an attack draft (see choice above).
- Dragging to take a block back, to reorder the hand, to reorder triggers, or
  from the out-of-board tray.
- Any change to recordings, Watch, the rail's lists or the server.

## What goes

Nothing is removed: no existing path is replaced. The click-click block flow,
`armed`, `toggleBlock`, pile counts and the card menu all stay and are reused.
No earlier drag attempt exists in the tree.

## Passes

1. **Gesture core + hand drag (`play`) — built.** `5cb7f0dc9d`, `979bce2f6d`,
   `9b1e8b4832`. `e2e/drag-play.spec.ts` (in the release config) covers
   commit, cancel over hand, Esc, click parity, multi-offer menu + Cancel
   focus, and refusal. `keyboard-play` and `combat-presentation` pass
   unchanged. `mana-payment.spec.ts` fails its pile-title check
   (`.title:visible` count 0) on the base commit too: pre-existing, unrelated.
2. **Block and target drags — built in `4dcf306764`.** All 12 drag browser cases and 21 focused drag/combat unit cases passed. Svelte reported zero errors or warnings. The drag spec covers assignment, reassignment, Escape restoring the armed blocker, previewless blocking, spell and ability sources, all three recipient types, payment preview, multi-offer choice and Study click parity.
3. **Attack marquee and interaction wording — built in `86c6a5f792`.** Release adds the highlighted attackers. Edge contact counts, including individual fanned copies. A scrolling row clips the sweep to its visible area. The empty block plan, block caption and shortcuts sheet describe dragging.
   The two stack endpoints have distinct lifetimes. Live ability entries can lack a card, and a resolved spell thumbnail can outlive its stack object. The type in `board/recent.ts` documents both.

Pass 3 checks covered 23 drag/combat unit cases and eight focused browser cases, including pointer and keyboard combat, block reassignment and five marquee cases. Svelte reported zero errors or warnings. The 1280×800 screenshots show the selection box and both card outlines in light mode with normal motion and dark mode with reduced motion.

4. **QA in a real game, and the joint-combat steer — built in `437101a9fe`.** Played two live games against the real server (Elemental GR vs Search 16; goblins vs Passive). Every gesture committed exactly one Command, or none on cancel. Two fixes:
   - Real servers answer an accepted step twice (`work` running, then complete). The draft runner stopped on the first, so every multi-creature attack or block declared one creature and printed "Stopped after 1 of N". This happened on the base commit too. The runner now waits until the board is live again. The combat fixture now answers the same way, and the old runner fails four combat tests.
   - Steer (2026-11-24/25, from the Task's comment: "Joint UX only in live play (UI-only, zero engine change)"; "regardless of drag, making the UX for combat only about the select-all-at-once UI rather than mixing individual decisions with click whatever you want"). While a live draft is open, the rail's per-creature offers sit in a closed "Each creature's choice" disclosure without key badges, and number keys do nothing. Replay and Study are unchanged. The attack caption now says "Drag across creatures, or click each". Scripted players in `game-history`, `laptop-comfort` and `experience-proof` open that disclosure through `showEachCreaturesChoice` (`e2e/fixtures/combat.ts`). Release proof: everything passes except `unified-replay`'s verified evaluation import, whose fixture server needs a full manabot environment rather than the play-runtime venv (unchecked locally).
5. **Post-build concept fixes (next pass).** Implement in order so the biggest mismatch is verified first:
   - **Joint-combat lifecycle & rejection recovery** (`+page.svelte`, `Board.svelte`, `Rail.svelte`): keep `combatPreview` steady for any active live-play table across transient `inputReady` waits; fold `Rail` on `{#if draft}` alone; suppress `asked` and keep draft marks steady while `running`; disable draft edits/confirm while `!live`; recover `running` cleanly on same-revision rejection (`!pending && live`). Add E2E proofs in `e2e/combat-presentation.spec.ts` covering: (a) per-creature disclosure stays closed and `asked` stays suppressed during Confirm's `work: running` window, (b) toggling a priority stop mid-draft preserves the draft and armed blocker without flashing raw choices, and (c) rejecting step 0 at the same revision recovers out of `running` with the draft intact.
   - **Exact `connectTargets` affordance** (`Board.svelte`, `Field.svelte`, `BoardCard.svelte`): highlight exact legal permanent recipients with dashed `--accent` (`data-connect-target`) during `connect` drags, upgrading to solid `--accent` on snap (`data-drop`), and keep `.tcard.legal` visible on attacking creatures. Add E2E assertions in `e2e/drag-play.spec.ts` verifying legal recipients (and not ineligible creatures or other blockers) receive `data-connect-target` mid-drag, plus `design.test.ts`.
   - **`play`-drag edge fixes** (`drag.ts`, `drag.test.ts`, `Board.svelte`): `promote(gesture, offerId)` checks offer membership (settling home on a different offer); `overField` includes `.board .tray`.

**Next:** Implement Pass 5, run `npm test && npm run check` and the focused drag/combat E2E specs, then proceed to gate.

## Done when

A person playing at `/` can cast/play from hand, assign and move blocks,
choose targets (permanents, players, stack entries) and draft an attack by
dragging; each gesture previews honestly and cancels cleanly without
disturbing clicks, focus, inspect, or animations; the same game can still be
played start to finish by click alone and by keyboard alone; and the gate
commands pass.
