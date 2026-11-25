# MTG-482 — Academy: swap sides in the middle of a game

Revised after concept review, 2026-11-25 (headless; `scratch/concept-review.html`;
choices below are made for review).

Direction, in the words given:

- "one cool thing we could add also is a "swap sides" button you can do midgame"
- On the bot taking the person's old seat: "we can build a shadow player?"
- On having seen both hands, and the record: "It doesnt matter. i dont care
  about my record or cheating the stats. just however we end it is what counts"

This branch is stacked on MTG-389 (PR #615, open; `lf sync MTG-389`). Play-on,
`LiveAnalysis`, My record and the Settings sheet exist only there. The other
files in `scratch/` came with that branch and are MTG-389's notes, not this
Task's.

## User-visible outcome

During a live Play game the pilot opens Settings → Game and presses **Swap
sides** (under **This game**, separated from **Next game**, where the existing
next-game play/draw button is renamed **Swap play/draw** while keeping its wire
`intent: "swap_sides"`). Almost instantaneously the board flips to the
opponent's seat in the same game: their hand, their deck name under "You", the
same life totals, stack and turn, and the live move list stays intact with a
**Swapped sides** divider. The bot now pilots the seat the person left, and
moves at once if that seat was the one to act. If **Show analysis** was on, it
follows the person to the new seat. Play continues to an ordinary end; the game
appears once in Games (with each player in their final seat and deck), counts
once in My record by how it ends (win if the seat the person finished in won),
and opens in Review showing every decision the person actually made before and
after the swap with a **Swapped sides · Turn T** divider. The control can be
used again to swap back.

## The model (built)

One live game whose seats swap in place: `Trace.seat_swaps` (event indices,
omitted when empty) plus `managym/academy/trace.py` `human_seat_at` resolve the
person's seat for every reader; `GameSession.swap_sides`
(`managym/academy/server.py`) builds the shadow bot for the old seat before
mutating, refuses without touching the game, is idempotent by `request_id`, and
bumps the revision. `AttemptStore.save` updates active `attempt_players` and
`session_games.matchup_json`; `record`/`summaries` are unchanged. Choices made
for review are listed in `scratch/questions.md` (1–9).

## Exclusions

- Changing a seat's native mana modes mid-game (requires native setter +
  `BoundaryIntervention`; separate Task if wanted).
- Swapping in Watch, in a Study branch, or handing a seat to another human.
- Marking or excluding games in which both hands were seen.

## What goes

- `scratch/probe/swap_replay.py` (kickoff probe for the superseded replay
  approach) once pass 1's tests are in place.

## Acceptance check at gate

`uv run python -m pytest tests/academy/test_swap_sides.py -q` passes, covering:
1. Start a game in seat 0 with `mana_payment="auto"` and priority stops against
   a fixture policy opponent, play several decisions (with at least one having
   `analysis_shown=True` if tested), swap sides. Assert:
   - Same `match_id`, `revision` incremented, response frame shows seat 1's
     hand, swapped `deck_names`, same life totals, and `Swapped sides` log beat.
   - `config.to_rust()` unchanged; `trace.seat_swaps == [k]`; pre-swap events
     keep original `actor`, `bot_decision`, `committed_at`, and `analysis_shown`.
   - If the opponent is a `recurrent=True` policy, its session observed every
     step from seat 0 before choosing.
   - Stale command for the old revision is rejected; duplicate `request_id`
     returns the current frame without swapping back.
   - Play to `game_over`: `ReplayedViews` re-executes `complete`,
     `game_numbers` succeeds, `analyse_game` reads all deliberate human
     decisions across both seats, `agent_line` works before and after the swap,
     `AttemptStore.record` counts the game once under the final seat's decks,
     Games lists exactly 1 row, and the recording index exposes `seat_swaps`.
2. Refusal: a policy refused for the other seat leaves the live game untouched
   and still accepting Commands at the same revision.
3. Double swap: swapping and swapping back produces 1 attempt with two entries
   in `seat_swaps`, re-executes `complete`, and counts under seat 0.
4. Play-on + swap: a game started via `played_from` that then swaps sides
   stays excluded from `AttemptStore.record`.
5. Protocol conformance: `tests/academy/test_shared_table_protocol.py` passes.

In the browser (Playwright): Play → Settings → Game → **Swap sides** → hand
flips to the other seat's, deck under "You" updates, opponent label still names
the bot, live move list keeps earlier moves with a "Swapped sides" note, and
the next Command is accepted; plus existing `play-session.spec.ts` updated for
**Swap play/draw**.

## Docs written first

Shipping docs updated for this direction:
- `docs/academy.md` (`### Swapping sides mid-game`, Settings → Game **This game** / **Next game** split, **Swap play/draw** rename, Previewing payments, Perspective, Analysis, Analysis while you play, The game in numbers)
- `managym/academy/README.md` (`### Swapping sides mid-game` and `analysis.py` summary)
- `protocol/README.md` (`swap_sides` under Shared-table control v1)

## Passes

1. Done (4a7adb5b29) — server swap, record and readers; `tests/academy/test_swap_sides.py` 6 passed.
2. Done (c97527f9c3) — protocol, Settings This game / Next game, Swap play/draw rename, Review dividers, estimate breaks, `e2e/swap-sides.spec.ts` green.

Remaining: none in implementation. Gate runs full CI (Python academy suite,
protocol conformance, UI unit + Playwright) — deferred to gate.

## Done when

A person mid-game presses **Swap sides** and immediately keeps playing the same
game from the other seat against the same bot; the finished game appears once
in Games, counts in My record by its result, and opens in Review with every
decision they made and each swap marked.
