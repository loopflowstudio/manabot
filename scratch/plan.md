# MTG-482 — Academy: swap sides in the middle of a game

Revised after post-build concept review, 2026-11-25 (headless;
`scratch/concept-review.html`; choices below are made for review).

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
`intent: "swap_sides"`). As soon as the replacement bot policy is prepared, the
Settings sheet closes and the board flips to the opponent's seat in the same
game: their hand, their deck name under "You", the same life totals, stack and
turn, and the live move list stays intact with a **Swapped sides** divider. If
the bot is now to act in the seat the person left, the flipped board shows the
bot thinking/searching while it takes its turn (just like after any human
move). If **Show analysis** was on, it follows the person to the new seat
without flickering off. Play continues to an ordinary end; the game appears once
in Games (with each player in their final seat and deck), counts once in My
record by how it ends (win if the seat the person finished in won), and opens
in Review showing every decision the person actually made before and after the
swap with a **Swapped sides · Turn T** divider — with the bottom/top plates,
top bar, rail heading, Perspective, and `This game` numbers all naming who
actually piloted each seat at the shown position. The control can be used again
to swap back.

## The model

One live game whose seats swap in place:

- **Permanent per native seat (`0 | 1`)**: decklist, `GameConfig.to_rust()`,
  native mana modes, and full-game `GameNumbers.seats[seat]`.
- **Dynamic per position (`event_index`)**: who pilots each native seat (`You`
  vs the bot), resolved everywhere from `Trace.seat_swaps` (event indices,
  omitted when empty) + `human_seat_at(trace, event_index)` (Python) /
  `humanSeatAt(startingHeroSeat, seatSwaps, event)` (TypeScript).
- **Live commit boundary**: `GameSession.swap_sides` prepares and admits the
  shadow bot before mutating, refuses without touching the game, is idempotent
  by `request_id`, bumps the revision, persists active `attempt_players` and
  `session_games.matchup_json`, immediately publishes the flipped board via
  `committed(...)` (preserving live-analysis policy state if restarting), and
  then calls `_advance()` so the person sees their new hand while the bot
  plays their old seat.
- **Authorized Play decisions**: across `Recording`, `_live_decision_summaries`,
  `GET /api/traces/{id}/decisions`, and `GET /api/traces/{id}/decisions/{addr}`,
  a Play game's authorized decisions are its `DecisionSource.CLIENT` decisions
  across whichever seats the person piloted — never `DecisionSource.POLICY`
  rows.

Choices made for review are listed in `scratch/questions.md`.

## Exclusions

- Changing a seat's native mana modes mid-game (requires native setter +
  `BoundaryIntervention`; separate Task if wanted).
- Reusing a warm stateless `PolicyRunner` across seats without calling
  `Policy.load(seat=...)` (separate optimization Task if wanted).
- Swapping in Watch, in a Study branch, or handing a seat to another human.
- Marking or excluding games in which both hands were seen.

## Passes

1. Done (`4a7adb5b29`) — server swap, record and readers; `tests/academy/test_swap_sides.py` 6 passed.
2. Done (`c97527f9c3`, `381937fb24`, `61f8621a45`) — protocol, Settings This game / Next game, Swap play/draw rename, Review dividers, estimate breaks, QA fixes (`liveAnalysis` reset on seat change, narrow rail move name, Play again matchup decks), `e2e/swap-sides.spec.ts` green.
3. **Next (from post-build concept review `scratch/concept-review.html`):** fix the three conceptual mismatches left in `HEAD`, plus closed-decklist consistency and leftover probe cleanup:
   - **3a. Review seat-vs-pilot separation (`academy/ui/src/routes/replay/+page.svelte`, `academy/ui/src/lib/board/replay.ts`, `academy/ui/src/lib/numbers.ts`):**
     - Resolve who piloted `seat` at the shown position via `currentHero = humanSeatAt(startingHeroSeat, seatSwaps, current?.event ?? null)` (for non-Watch Play games; Watch keeps fixed seat identities).
     - Use position-aware pilot naming for `<Board you={...} opponent={...}>` (plates + top bar) and `positionHeading(...)` (rail heading), while keeping `deckOf(seat)` permanent per native seat.
     - Fix `identity(seat)` to attach `index.opponent` producer/provenance to the `kind === 'bot'` player (not hardcoded `seat === 1`, which attached bot provenance to `You` whenever the bot finished in seat 0).
     - When `seatSwaps.length > 0`:
       - Label fixed Perspective options by native seat and deck (e.g. `Seat 1 at the bottom · UR Lessons`), not solely by final pilot.
       - In `This game`, show starting vs final order when they differ (e.g. `Started on the play · finished on the draw`), label the seat-total table columns by native seat/deck (noting who finished in each) rather than plain `You` / `<Bot>`, and omit the single-line `estimateArc` (`start · peak · end`) so complementary pre- and post-swap scores are never combined into one fake arc.
       - Update `docs/academy.md` if `This game` wording changes.
     - Add unit/component tests covering Review at a pre-swap position of a once-swapped game (plates, heading, provenance, Perspective, `estimateArc` omitted).
   - **3b. Authorized Play decisions on `GET /api/traces/{trace_id}/decisions` (`managym/academy/server.py`, `managym/academy/replay_index.py`):**
     - Stop calling single-viewer `project_replay(replay, final_seat)` for swapped Play traces in `get_trace_decisions` (`server.py:5253`), which drops pre-swap human `CLIENT` decisions and leaks pre-swap bot `POLICY` decisions whose addresses then 404 on `/decisions/{address}`.
     - For Play traces (`not watch_mode`), return every `DecisionSource.CLIENT` decision across seats with its `DecisionAddressV2` address and zero `DecisionSource.POLICY` decisions (keep `project_replay(replay, viewer)` single-viewer for Watch/per-seat callers and preserve exact payload shape for unswapped Play traces).
     - Align closed-decklist redaction in `prepare_trace_payload` (`managym/academy/trace.py:476`) with `get_recording_index` (`managym/academy/server.py:4993`) so both redact the current bot seat consistently when `not reveal_hidden`.
     - Add tests in `tests/academy/test_swap_sides.py`: `GET /api/traces/{id}/decisions` on a once-swapped game returns pre-swap Seat 0 and post-swap Seat 1 `CLIENT` decisions, returns zero `POLICY` decisions, and every returned `address` restores 200 on `GET /api/traces/{id}/decisions/{address}`; plus a closed-decklists swapped test checking consistency between `/api/traces/{id}` and `/recording`.
   - **3c. Immediate board flip before `_advance()` (`managym/academy/server.py`, `tests/academy/test_swap_sides.py`):**
     - Thread `_dispatch_table_request`'s existing `committed` callback into `GameSession.swap_sides(...)` (matching `hero_command`).
     - Right after `_commit_seat_flip` + `_persist_attempt` and before `self._advance()`: bump `record.table_revision` (or emit via `committed` with the updated viewer), send the immediate flipped frame (`reason="explicit_resync"`) preserving `"analysis": {"policy": live_analysis_policy}` when live analysis is restarting so the client's `"Analysis on"` chip never flickers, and set `self._last_presentation_cursor = self.presentation.next_seq` before `_advance()`.
     - Then run `self._advance()`, start `LiveAnalysis` for the new seat if it was on and the game is not over, and return the settled message.
     - Add a WebSocket test verifying that during `swap_sides` the pilot (and any watcher) receives the flipped board (`rules_viewer == 1`, `"Swapped sides"` log entry) BEFORE the bot finishes choosing, and then receives the settled frame after the bot moves.
   - **3d. Remove superseded probe:** delete `scratch/probe/` (`swap_replay.py`) if still present.

## Acceptance check at gate

- `uv run python -m pytest tests/academy/test_swap_sides.py tests/academy/test_shared_table_protocol.py tests/academy/test_replay_index.py tests/academy/test_study_runtime.py -q` passes, covering all prior swap-sides assertions plus:
  1. `GET /api/traces/{id}/decisions` on a once-swapped Play game includes both pre-swap Seat 0 and post-swap Seat 1 `CLIENT` decisions, excludes all `POLICY` decisions, and every returned address restores 200.
  2. WebSocket `swap_sides` emits the committed flipped board (`rules_viewer` flipped, `"Swapped sides"` log) before a slow bot finishes `_advance()`, without flickering live analysis off.
  3. Closed-decklists swap redaction is consistent between `/api/traces/{id}` and `/recording`.
- Frontend unit tests (`npm --prefix academy/ui test`) and typecheck (`npm --prefix academy/ui run check`) pass, including new tests for Review pre-swap pilot attribution (`pilotAt` / plates / heading / provenance / Perspective / `estimateArc`).
- Playwright `academy/ui/e2e/swap-sides.spec.ts` passes (and verify Review at Turn 1 of a once-swapped game shows `You` on the pre-swap bottom plate and `Your choice` as the heading).

## Done when

A person mid-game presses **Swap sides**, immediately sees the flipped seat and
hand while the bot plays the seat they left, and keeps playing the same game to
an ordinary end; the finished game appears once in Games, counts in My record
by its result, exposes only their own `CLIENT` decisions through every API, and
opens in Review with truthful pilot names, decks, dividers, and analysis at
every position before and after each swap.
