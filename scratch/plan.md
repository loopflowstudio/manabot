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

## The model: one live game whose seats swap in place

A game has two fixed native seats (0 and 1) and one immutable starting setup
(`GameConfig`, `config.to_rust()`). Swapping sides mid-game changes only **who
pilots which seat** from that step onward — never the native game, its seed,
its match id, or its history.

Why this (and not the kickoff plan's second replayed game):
- `managym.Env(decision_memory=True)` already runs by native seat and captures
  decision memory for **both** seats 0 and 1.
- `CanonicalReplayV1` already records `viewer: 0 | 1` and
  `source: "client" | "policy"` on every decision, plus both seats'
  presentation tracks from step 0.
- `AttemptStore.record`, `summaries`, `session_results`, `session_totals`, and
  `GameNumbers` already key off `attempt_players.seat` and native seat 0/1.
- Swapping in place is instant (~5–50 ms for all real policies, instead of
  2–5 s replaying 100+ frames), creates no hidden stopped half-game, preserves
  the live move list, Study roots, real `bot_decision` outputs, `committed_at`
  timestamps and `analysis_shown` marks, handles multiple swaps cleanly, needs
  no opponent-seat mana fields on `GameConfig`, and keeps `played_from` pure
  (so a game started via **Play on from here** that also swaps sides stays out
  of My record).

### One record field and one pure helper

`Trace` gains one optional field:

- `seat_swaps: list[int] = field(default_factory=list)` — the `len(trace.events)`
  indices at which the pilot swapped seats, omitted from serialized JSON when
  empty (so unswapped traces are byte-identical to today).

One pure function in `managym/academy/trace.py` owns seat resolution for every
reader:

```python
def human_seat_at(trace: Mapping[str, Any], event_index: int | None = None) -> int:
    """Native seat the human piloted at `event_index` (or at game end when None)."""
    seat = int(trace["config"].get("human_seat", 0))
    swaps = trace.get("seat_swaps") or ()
    flips = len(swaps) if event_index is None else bisect_right(swaps, event_index)
    return seat ^ (flips & 1)
```

At every step `i`, `event["actor"] == "hero"` means the human acted (in seat
`human_seat_at(trace, i)`) and `event["actor"] == "villain"` means the bot
acted (in seat `1 - human_seat_at(trace, i)`). `academy_actions(obs, seat)` is
called with `human_seat_at(trace, i)`, so every stored action description
re-executes and verifies identically across any number of swaps.

### The shadow player (loaded at swap time)

Choice made for review: build the bot for the person's old seat at the swap,
not at the start of every game.

- All real policies (`BaselineSession`, `AnchorSession`, `Search`, `Random`,
  `FlatSearchPolicy`, `Passive`) are stateless across `observe` and read
  decision memory from `env.capture_decision(seat, 0)`, which the live `Env`
  already captures for both seats. Loading the policy with
  `build_villain_policy(config, seat=old_human_seat)` and calling
  `observe_policies((new_policy,), self.env, self.obs)` gives it the exact
  state a bot sitting in that seat from turn 1 would have — in milliseconds.
- If `isinstance(new_policy, PolicyRunner) and new_policy.base_descriptor.recurrent`
  (test fixtures, or any future recurrent policy): before mutating the live
  session, warm `new_policy` up on a lightweight `managym.replay.walk` of the
  game so far (`observe_policies` at reset and after each step `0..k-1`, plus
  discarded `choose_bot` at `old_human_seat`'s decisions if needed by the
  session contract). No `ExperienceFrame`, no Pydantic, no `clone_env`.

### Exact swap transition (`GameSession.swap_sides`)

All fallible work runs **before** mutating the live session (disposable
`new_policy` guarded by `ExitStack`):

1. **Idempotency & precondition checks** (under the table lock, behind
   `TableCapability.CONFIGURE_MATCH`):
   - Duplicate `request_id` matching the recorded `(match_id, expected_revision)`
     returns `self.current_message()` without swapping again; reusing the same
     `request_id` with different bindings raises `ValueError`.
   - Refuse (leaving the live game untouched) if: no active game, `watch_mode`,
     `obs.game_over`, `mirror_retry is not None`, active Study branch on the
     table, `match_id != self.match_id`, or `expected_revision != self.revision`.
2. **Prepare replacement policy** for `new_bot_seat = self.human_seat`:
   - `build_villain_policy(self.trace.config, seat=new_bot_seat)` (admits
     against `config.to_rust()` and loads). Any failure closes the disposable
     policy and raises `swap_unavailable: <reason>`, leaving the live game
     untouched and accepting Commands at the same revision.
   - If recurrent, warm up via `walk` of `record_of` up to `len(trace.events)`.
     Otherwise `observe_policies((new_policy,), self.env, self.obs)`.
3. **Commit the swap in place**:
   - Remember live analysis policy/publisher if `self._live_analysis is not None`,
     then stop `_live_analysis` and close the old `self.villain_policy`.
   - Install `self.villain_policy = new_policy`.
   - Flip `self.human_seat = 1 - self.human_seat`.
   - Flip `self.deck_names = {"hero": old["villain"], "villain": old["hero"]}`.
   - Append `len(self.trace.events)` to `self.trace.seat_swaps`.
   - Update `self.record_players` so the human holds `self.human_seat` (with
     that seat's deck) and the bot holds `1 - self.human_seat` (with that
     seat's deck).
   - Clear `self._f6_turn = None`, `self._step_boards = []`,
     `self.published_prompt = None`, `self._pending_decision = None`.
   - Refresh `self.presentation_events` from
     `self.canonical_presentation[self.human_seat][-MAX_PRESENTATION_EVENTS:]`
     and `self._last_presentation_cursor = self.presentation.next_seq`.
   - Bump `self._advance_protocol_revision()` so any in-flight Command for the
     old seat is rejected as stale.
   - Record receipt `self._swap_receipts[request_id] = (match_id, expected_revision)`.
   - Append a live log beat `{"actor": "system", "turn": current_turn, "text": "Swapped sides"}`.
   - Persist (`self._persist_attempt(config_changed=True)`), which updates the
     active attempt's `trace_json`, `attempt_players`, and
     `session_games.matchup_json` atomically.
   - Run `self._advance()` (so if the person's old seat was to act, the bot
     immediately moves; if that bot decision times out, `mirror_retry` behaves
     normally on the swapped game).
   - If live analysis was on and the game is not over, restart `LiveAnalysis`
     for the new `self.human_seat` on its background thread (never blocking the
     swap).
   - Return `self._wire_message(reason="explicit_resync")` (broadcast to table
     participants so watchers and `table_snapshot.access.identity.rules_viewer`
     stay in sync).

### What follows across layers

- **Engine replay (`managym/academy/recording.py` `record_of`)**:
  Fix the pre-existing omission where `record_of` called
  `PlayerSetup.retain(player, config.decklists)` without forwarding
  `mana_floats` and `mana_payment` from `config.to_rust()`. With `GameConfig`
  staying the immutable starting setup, `record_of(attempt)` reproduces every
  game — including games with `mana_payment="auto"` or `mana_floats!="off"` —
  with no new fields on `GameConfig`.
- **Native mana modes**: stay with their native seats as set up at game start
  (changing a seat's native mana mode mid-game needs a native setter +
  `BoundaryIntervention`, excluded as its own engine Task if wanted). Browser
  Auto-tap stays with the person.
- **Persistence (`managym/academy/attempts.py`)**:
  In `AttemptStore.save`, when updating an existing `status='active'` attempt
  (only when the `UPDATE ... WHERE status='active'` affected 1 row), replace
  `attempt_players` with the current `players` when they differ, and update
  `session_games.matchup_json` if the attempt belongs to a session. No schema
  migration, and **zero changes** to `record`, `summaries`, `session_results`,
  or `session_totals`: one game = one row, counted once under the final seat
  and decks, and a `played_from` game that also swapped stays excluded from My
  record.
- **Session setup (`play_session.py`)**:
  `play_sessions.draft_json` is untouched by a mid-game swap. Game number stays
  unchanged (`Game 1` stays `Game 1`). Next-game **Rematch** and **Swap
  play/draw** keep their exact meaning.
- **Recording & Review (`managym/academy/recording.py`, `trace.py`, `server.py`,
  `academy/ui/src/lib/board/replay.ts`, `routes/replay/+page.svelte`)**:
  - `ReplayedViews` resolves each step's seat via `human_seat_at(trace, index)`
    and final board via `human_seat_at(trace, None)`.
  - `Recording` includes every `ReplayDecision` where
    `row.source == DecisionSource.CLIENT` (all of the person's decisions across
    both seats), reads `self.tracks[viewer]` and `person_board` using
    `human_seat_at`, and exposes `seat_swaps` (list of `{event, turn}`) plus
    `starting_human_seat` on `/api/traces/{id}/recording`.
  - `prepare_trace_payload` normalizes each event `i` to
    `human_seat_at(prepared, i)` and `final_observation` to
    `human_seat_at(prepared, None)`.
  - Study restoration (`get_trace_decision`, `_live_decision_summaries`,
    `restore_decision`) authorizes client-sourced decisions (`source == CLIENT`)
    of the participant's game in whichever seat they were made.
  - Play on from Review (`_play_recorded_prefix`): when playing on from an
    event after a swap in a swapped game, applies the source's `seat_swaps`
    before `upto` so the person lands in `human_seat_at(source, upto)` with the
    bot in the other seat.
  - Review UI (`replay.ts`, `replay/+page.svelte`): `replayLog` and
    `stripMarks` resolve the human seat per event via `seat_swaps`, insert a
    **Swapped sides · Turn T** system divider at each swap, show a caption in
    notices ("Swapped sides at turn T"), and default `perspective` to `chooser`
    when `seat_swaps` is non-empty so the bottom seat follows the person across
    the swap (while fixed Seat 1 / Seat 2 perspectives remain selectable).
- **Analysis & Numbers (`managym/academy/analysis.py`, `numbers.py`,
  `server.py`)**:
  - `recorded_steps` and `game_numbers` pass `human_seat_at(trace, index)` to
    `step_matches`.
  - `analysed_events(trace)` selects all deliberate human decisions
    (`event["actor"] == "hero" and not event.get("auto")`), across both seats
    if swapped (choice made for review; passed over: analysing only the final
    seat after the swap).
  - `analyse_game` loads one runner per seat piloted (still 1 for unswapped
    games, at most 2 when swapped) and reads each decision from
    `human_seat_at(trace, index)`.
  - `see_line` (`agent_line`) loads the analysing policy for
    `seat = human_seat_at(trace, request.event)` and the opponent for
    `1 - seat`, so **See its line** works at every decision before or after a
    swap.
  - `LiveAnalysis` walks `recorded_steps` with `human_seat_at` and reads the
    current seat's pending decisions.
- **Wire protocol (`shared_table_protocol.py`, `protocol/`, `socket.svelte.ts`)**:
  - Add `SwapSidesMessage` (`type: "swap_sides"`, `grant_revision`,
    `request_id`, `match_id`, `expected_revision`) behind `CONFIGURE_MATCH`,
    and update `protocol/testing-house-v1.schema.json`,
    `protocol/fixtures/testing-house-control-v1.json`, and
    `academy/ui/src/lib/shared-table-protocol.ts` in lockstep.
  - On the client, `sendSwapSides()` binds `match_id`, `expected_revision`,
    `grant_revision`, and a fresh `request_id`; `applyRecovery` on the same
    `match_id` with higher `revision` already updates the board, deck names,
    stops, and presentation cursor without wiping `actionLog`.
- **Settings → Game & Game Over (`+page.svelte`, `PlaySessionStrip.svelte`)**:
  - Group Settings → Game into **This game** (**Swap sides** button with short
    caption, disabled when not pilot / game over / pending / Study /
    `mirror_retry`, shows pending state and inline refusal error) and **Next
    game** (`PlaySessionStrip` + **New game**).
  - Rename the next-game button in `PlaySessionStrip.svelte` and Game Over
    from **Swap sides** to **Swap play/draw** (keeping wire
    `intent: "swap_sides"`), updating `PlaySessionStrip.svelte.test.ts`,
    `academy/ui/e2e/play-session.spec.ts`, and `docs/academy.md`.

## Source of truth

- Live: `GameSession.human_seat`, `GameSession.trace.seat_swaps`,
  `GameSession.trace.config` (immutable starting setup), and `GameSession.env`.
- Persisted: `attempts.trace_json` (`config` + `seat_swaps` + `events`),
  `attempt_players` (each player's current/final seat and deck).

## Affected surfaces

- `managym/academy/trace.py`: `Trace.seat_swaps`, `human_seat_at`,
  `prepare_trace_payload`.
- `managym/academy/attempts.py`: `AttemptStore.save` updating active
  `attempt_players` and `session_games.matchup_json`.
- `managym/academy/recording.py`: `record_of` mana-mode fix; `ReplayedViews`
  and `Recording` using `human_seat_at` and `DecisionSource.CLIENT`.
- `managym/academy/analysis.py`, `numbers.py`: `human_seat_at` in
  `recorded_steps`, `game_numbers`, `analysed_events`, `analyse_game`,
  `agent_line`, `LiveAnalysis`.
- `managym/academy/server.py`: `GameSession.swap_sides`, dispatch,
  `_participant_from_resume`, `get_recording_index`, `get_trace_decision`,
  `start_analysis`, `see_line`, `_play_recorded_prefix`.
- `managym/academy/shared_table_protocol.py`,
  `protocol/testing-house-v1.schema.json`,
  `protocol/fixtures/testing-house-control-v1.json`.
- `academy/ui`: `shared-table-protocol.ts`, `socket.svelte.ts`, `recording.ts`,
  `board/replay.ts`, `routes/+page.svelte`, `components/PlaySessionStrip.svelte`,
  `routes/replay/+page.svelte`, plus `PlaySessionStrip.svelte.test.ts` and
  `e2e/play-session.spec.ts`.
- `docs/academy.md`, `managym/academy/README.md`.

Must keep working unchanged: play-on from Review and its refusals, rematch and
next-game play/draw swap, reconnect to pending decision, stale-Command
rejection, watcher tables, traces written before this change, `testing-house-v1`
conformance tests.

## Operational boundary

- Non-recurrent policy swap (all real checkpoints, Search, Random, Passive)
  takes under 100 ms (no engine replay). Recurrent test-policy warm-up walks
  native `Env` without building frames or Pydantic models.
- One swap at a time under the table lock; duplicate `request_id` is
  idempotent; stale `expected_revision` is rejected.

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

### Product question exposed while writing docs

- **Estimate discontinuity across a mid-game swap (`Decisions` strip, `keyMoments`, `estimateArc`)**:
  Post-game **Analysis** and **Show analysis** plot the policy's 0–100 estimate
  from whichever seat the human piloted at each decision. Across a swap the
  estimate flips sides (e.g. 28 → 72 even though the board did not change).
  Should `keyMoments` skip "estimate moved by your next decision" across a seat
  swap (so the swap itself is never flagged as a huge swing) and should the
  **Decisions** strip break the line / mark the swap boundary, or draw one
  continuous line? (Suggested default for Pass 2 if unanswered: skip
  cross-swap pairs in `keyMoments` and break the estimate polyline at each swap
  so the chart stays honest.)

## Passes

1. **Done** — **In-place server swap, record & readers**: implemented
   `Trace.seat_swaps` + `human_seat_at`, `record_of` mana-mode forwarding,
   active `attempt_players` + `session_games.matchup_json` update in
   `AttemptStore.save`, `GameSession.swap_sides` (pre-mutation policy load &
   admission refusal, idempotent `request_id`, recurrent `RecordWalker` warm-up
   using `steps = iter(walker)` before applying each action, `LiveAnalysis`
   handoff), `ReplayedViews` / `Recording` / `analysis.py` / `numbers.py` /
   `_play_recorded_prefix`, deleted `scratch/probe/swap_replay.py`, and
   verified with `tests/academy/test_swap_sides.py` (6 passed).
2. **Done** — **Protocol, Settings UI, Review & Playwright**: added
   `SwapSidesMessage` to `shared_table_protocol.py`, regenerated
   `protocol/testing-house-v1.schema.json`, updated
   `protocol/fixtures/testing-house-control-v1.json` (also fixing pre-existing
   round-trip drift on `resume`/`rematch` so Python & TS conformance tests both
   pass), added `sendSwapSides`, split Settings → Game into **This game**
   (**Swap sides**) and **Next game** (**Swap play/draw**), updated Review's
   divider / perspective default / `keyMoments` cross-swap skip / `Strip`
   polyline breaks, updated `PlaySessionStrip.svelte.test.ts` and
   `e2e/play-session.spec.ts`, and added `e2e/swap-sides.spec.ts` (green in
   Playwright).

Remaining: gate / full CI validation.

## Done when

A person mid-game presses **Swap sides** and immediately keeps playing the same
game from the other seat against the same bot; the finished game appears once
in Games, counts in My record by its result, and opens in Review with every
decision they made and each swap marked.
