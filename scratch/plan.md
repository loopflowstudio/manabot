# MTG-482 — Academy: swap sides in the middle of a game

Kickoff, 2026-11-25 (headless; choices below are made for review).

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

During a Play game the pilot opens Settings → Game and presses **Swap sides
now**. After a short "Swapping sides…" the board shows the same position from
the opponent's seat: their hand, their deck name under "You", the same life
totals, stack and turn. The bot now plays the seat the person left, and moves
at once if that seat was the one to act. Play continues to an ordinary end, and
the game counts in My record by how it ends: a win if the seat the person
finished in won. The control can be used again to swap back.

## The design: a swap starts a new recorded game replayed from the live one

The swapped game is a new recorded game. It has the live game's seed and native
setup, the person in the other seat, and every move made so far replayed
through the Command path with the roles exchanged. The live game ends with
`end_reason: "swapped_sides"`. This is play-on (`played_from`,
`managym/academy/server.py` `_played_from_source`, `_play_recorded_prefix`)
with its three refusals lifted for this one caller: the source is the session's
own live game, the stopping point is "all of it", and the person changes seat.

Why this and not one game with a seat change written into it: a game record
has one `config.human_seat`, and every reader pairs the `hero`/`villain` role
of an event with it — `acting_seat`, `step_matches`, `ReplayedViews` and
`Recording` rows in `recording.py`, `analysed_events` in `analysis.py`,
`normalize_observation_to_hero` in `trace.py`, the `a.winner=h.seat` record
queries in `attempts.py`, the per-viewer presentation track, decision restore
and Study roots in `server.py`. A seat that changes mid-record makes each of
those time-dependent, and `hero_deck` would stop meaning "your deck". A new
game keeps the swapped game an ordinary game for all of them. Passed over for
that reason; its advantages were an instant swap and no stopped source game.

Probe (`scratch/probe/swap_replay.py`, run against the MTG-389 checkout's built
engine, same commit): a live game's moves replay into a second session with the
seats exchanged, the person lands in the other seat, the game plays to
`game_over` and re-executes `complete` under `ReplayedViews`. Three setups
passed (every window surfaced; priority stops; second seat with closed
decklists). What it also showed, each now part of the design:

1. **Action descriptions name players by role.** `academy_actions` writes
   "Hero"/"Villain" into descriptions, so the recorded description of a move
   does not match after the exchange (the first run diverged at step 10). The
   replay compares the acting native seat and the action index, and reads the
   description under the source game's roles.
2. **The record ends on the bot's turn.** After the last recorded move the seat
   to act is the person's old one. The replay stops consulting the record
   there and the bot chooses (the unmodified hook raised "no longer plays as
   recorded" one step past the end).
3. **Native mana modes belong to seats.** `GameConfig.to_rust` gives the
   human's seat `mana_payment`/`mana_floats` and the bot's seat
   `"manual"`/`"off"`. Exchanging the roles moved auto payment to the other
   seat and the replay diverged at step 20. The swapped game keeps each native
   seat's modes, so the config gains the opponent seat's two modes (default
   manual/off, omitted from records when default). Consequence: after a swap
   the person plays the seat as it was set up (manual payment, no floating
   mana, unless they swap back), and the bot plays the old seat with the
   person's modes. The browser's Auto-tap is client-side and stays with the
   person. Games set up with the defaults see no difference.
4. **Cost is about 56 ms per move** (52 moves in 2.9 s, 31 in 1.5 s, 43 in
   1.2 s). The profile puts roughly half in reply frames nobody reads
   (`_accepted_outcome` → `_experience_frame`) and per-step board projections.

### The shadow player

The bot that takes the person's old seat is loaded for that seat and shown the
whole game from it during the replay: `observe` at every step, and for a
policy that keeps history in its session (`PolicyDescriptor.recurrent`) a
discarded `choose` at each of that seat's decisions, as `_play_recorded_prefix`
and `LiveAnalysis` already do. It has exactly the history a policy sitting in
that seat from the first move would have.

Choice made for review: the shadow is built at the swap, not kept running in
every game. Finding behind it: the registered checkpoints keep no history in
the session — `BaselineSession.observe` and `AnchorSession.observe` are `pass`
(`manabot/sim/policy_provider.py`), and decision memory is the engine's
per-seat capture (`env.capture_decision(seat, 0)`), which the replayed game
holds for both seats. Only test fixtures set `recurrent=True`. Passed over: a
second policy session loaded in the person's seat at the start of every game.
It would make the swap instant only together with the one-record design above,
and it charges a load and a policy pass per decision to every game, most of
which never swap.

### What can refuse

- A policy that cannot be admitted or loaded for the other seat: the load
  happens before the live game is closed (as `new_game` does today), so the
  swap answers `swap_unavailable: <the policy's reason>` and the game goes on
  untouched. The manabot provider admits by the pair of decks, not the seat
  (`policy_provider.py` `admit`), so trained opponents normally can take it.
- Game over, a bot decision awaiting retry (`mirror_retry`), a Study branch
  open, a watched game, or a request bound to an older match or revision:
  refused with the game unchanged.
- A replay that stops matching (`swap_diverged`). Same process, build, seed
  and native setup, so this is an engine determinism bug, not an expected
  state; it is reported and both games stay stored.

### The record

- The swapped game stores `played_from: {trace_id, event, swapped: true}`,
  `event` being the number of moves replayed. Its players are recorded in the
  seats they finish in: the human in the new seat with that seat's deck, the
  bot in the old one.
- My record (`AttemptStore.record`) counts a swapped game by its result, in
  the row of the opponent and decks the person ended with. It still leaves out
  a game played on from a finished one. No "both hands seen" mark is recorded.
- The source game stays stored, `stopped`, `swapped_sides`. Games and the
  session's tallies leave it out (choice made for review; passed over: showing
  it as one more stopped game, as a game left for a new one shows today — each
  swap would add a stopped half-game beside the one that counts).
- Review of a swapped game shows "Swapped sides at turn T" where a played-on
  game shows where it was played on from, and a divider at that move. Analysis
  of "your decisions" starts at the swap: the moves before it in that seat
  were the bot's (`analysed_events`).
- The session's next-game setup is unchanged by a mid-game swap; the existing
  next-game **Swap sides** keeps its meaning.

## Source of truth

The live game's parsed `GameConfig` and its trace events. The swapped config is
derived from that `GameConfig` directly (hero/villain decks, names, sideboards,
import sources and pool keys exchanged; `human_seat` flipped; the two seats'
mana modes exchanged), never re-parsed from a setup draft: a draft re-resolves
pool decks and imports. `config.to_rust()` of the swapped config equals the
live game's native setup; a test asserts it.

## Affected surfaces

- `managym/academy/trace.py`: `GameConfig` opponent-seat mana modes;
  `PlayedFrom.swapped`; the config exchange.
- `managym/academy/server.py`: `GameSession.swap_sides`, the generalized prefix
  replay, live stops and auto-pass carried into the swapped game, dispatch.
- `managym/academy/shared_table_protocol.py`: a `swap_sides_now` request bound
  to match id and revision with a request id, behind `configure_match`.
- `managym/academy/attempts.py`: `record`, `summaries`, session results/totals.
- `managym/academy/analysis.py`, `recording.py`, the recording index endpoint.
- `academy/ui`: the Settings sheet's Game group (`+page.svelte`
  `gameSettings`), the game store, Review's header (`routes/replay`), Games.
- Academy docs where play-on and the Settings sheet are described.

Must keep working unchanged: play-on from Review and its refusals, rematch and
next-game Swap sides, reconnect to the pending decision (the swapped game is a
new match, delivered as a new game is), stale-Command rejection, watcher
tables, traces written before this change.

## Operational boundary

- Swapping a 100-move game takes under 3 s on the development laptop (today's
  path would take about 5.6 s). The replay skips building reply frames and
  defers persistence to one save, as play-on's prefix defers persistence now.
- One swap at a time per table, under the table lock; a repeated request with
  the same id returns the swapped game instead of swapping again.

## Exclusions

- Changing a seat's native mana modes mid-game (so auto payment follows the
  person): a native change that must be in the replay record.
- Swapping in Watch, in a Study branch, or handing a seat to another person.
- Marking or excluding games in which both hands were seen.
- Speeding up the Command path beyond what the replay skips.

## What goes

- `scratch/probe/swap_replay.py` once pass 1's tests cover it.
- Nothing shipped is removed. Play-on's three refusals stay for play-on.

## Acceptance check at gate

`uv run python -m pytest tests/academy/test_swap_sides.py -q` passes, and its
central scenario is: start a game in seat 0 with native auto payment and
priority stops against a fixture policy opponent, play twenty decisions, swap.
Then: the response is a new match whose frame shows the other seat's hand and
the same life totals; `config.to_rust()` matches the source's; the first N
events have the source's native seats and action indices; the bot's session
observed every step from the old seat; the game plays to `game_over`;
`ReplayedViews` re-executes it `complete`; `AttemptStore.record` counts it once
under the final seat's decks; Games lists it and not the source; the recording
index reports `played_from.swapped`. Beside it: a policy refused for the other
seat leaves the live game accepting Commands; a swap back returns the person
to the first seat.

In the browser (one Playwright spec): Play → Settings → Game → Swap sides now →
the hand changes to the other seat's, the opponent label still names the bot,
and the next Command is accepted.

## Passes

1. **Server swap and record** — the pass that could prove the plan wrong. It
   shows the real replay under exchanged roles holding native modes by seat, a
   policy opponent taking the other seat with its history, refusal before
   anything closes, the 3 s bound, and the record, Games and analysis rules.
   Ends with the acceptance test file green.
2. **Wire, Settings control, Review** — the request and its dispatch, the
   button with its pending and refused states, the Review marker, the browser
   spec, docs.

## Done when

A person mid-game presses Swap sides now and keeps playing from the other seat
against the same bot; the finished game appears once in Games, counts in My
record by its result, and opens in Review with the swap marked.
