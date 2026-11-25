# MTG-389 — Study mode

Direction (Jack Heart, 2026-11-18): "Study mode: Game history, policy
suggestion overlays, win % chart, [chess.com](<http://chess.com>) inspirations,
game stats, replay viewer, "open hands" mode".

Earlier words this builds on (MTG-384 record,
`../manabot.mtg-384-redesign-academy-from/scratch/plan.md`):

- "The game log / history is probably importnat too. I'd also be interested in
  like an overlay for what the policy would do in your seat"
- "Chess.com is defintiely another \"role model\" to learn from."
- Of the Study sketch (eval bar, timeline, move list marked against the
  policy): "i think this is super cool but probably needs to be expandable
  and/or some of it is saved for post-game only"
- "Ours should be elegant. A study tool. Somewhat academic. Precise, not casual."

On game records (2026-11-16, the logging redesign): game records hold inputs
and committed actions and Academy re-executes them; "agree with your
proposal".

The brief ends: "Start with a design pass and stop for direction before
building." Kickoff and the concept review both ran headless, so nobody
answered. The design below is approved by nobody; each choice a person may
reverse is marked **(for review)** and listed in `scratch/questions.md`. The
review page is `scratch/concept-review.html`.

## The idea in one paragraph

A played game can be played again. It already stores its seed, both setups
and the action taken at every step, which is everything the engine needs to
reproduce it. So Study stores almost nothing new: it **reads** a finished
game by re-executing it. Your own view at every step, both hands once the
game is over, the name of every move, the game's numbers, and what a policy
would have done in your seat are all readings of the same game. The policy's
reading is an **Analysis**: asked for after the game, as on chess.com, from
any registered policy, as many times as you like, and saved with the game.
During play the same reading can be shown at the current decision if you
turn it on, and a decision made with it visible says so for ever. That
flag and each step's commit time are the only new facts a game records.

## Why not record the advisor while you play

The kickoff plan loaded a second policy in your seat during every game and
wrote its evidence, your view and the spectator view into each step of the
trace, sealed until the game ended. The concept review replaced it, for
these reasons:

- **You want it after, but had to choose it before.** The advisor was a
  Setup field. Forget it, or play the out-of-the-box Search 64 opponent, and
  that game could never be analysed. A better checkpoint next week could not
  re-read last week's games. One failed pass left a permanent "not retained".
- **It stored what the engine can reproduce.** `docs/engine-python.md` says a
  record "stores no boards. Anything that shows or counts a game re-executes
  it with managym". Academy's played games were the one exception, and the
  plan added two more boards per step to a row that is already 8 MB at 150
  events and rewritten on every save.
- **It needed sealing rules for things that need not exist yet.** Hidden
  views and advisor evidence sat in the live trace and had to be stripped at
  every reader. Computed after the game, there is nothing to leak during it.
- **It put a forward pass on the path of every decision** (the 100 ms p95
  bound) for evidence most games would never show.
- The four things the kickoff plan left out for want of a record (card
  statistics, another line from a finished game, a live evidence provider
  for Study's Reveal, analysis of a watched seat) are all readings of a
  re-executed game. They stay out of scope here, but stop being separate
  architecture.

## What exists (verified 2026-11-18, base `abecb6c56c`)

| Piece | State | Source |
|---|---|---|
| A played game's inputs | Stored today: `GameConfig.seed` (always set, `server.py:1039`), both decks, sideboards, `human_seat`, `decklists`, and `TraceEvent.action` for every step, automation included. `GameConfig.to_rust()` already "rebuild[s] the recorded root". The row's `world_json` holds the content-pack manifest and rules binary hash. | `trace.py:83-105,157-179`, `server.py:1738,2449` |
| One way to change the game | Play mutates its `Env` only through `step_semantic_command`; every step asserts the Command's offer is `offers[action_index]` of the native frame, so the stored index names the step. | `server.py:1675-1690` |
| The engine's record | `managym.recording.GameRecord` (`GameInputs`: seed, players, actions, digests, annotations, end) and `managym.replay.RecordWalker`, which re-executes, refuses another world or card set, and stops at the first divergence. Statistics are `Reducer`s over a walk (`Behaviour`, `CardPerformance`). | `recording.py:121-268`, `replay.py:81-250`, `statistics.py:98-382` |
| Academy re-executes records already | `import_record` walks a `GameRecord` through `GameSession`'s Command authority and saves a Watch recording. A played game never becomes a record and is never re-executed. | `watch.py:488-600` |
| Games library | `/games` lists every recorded game with filters and paging; rows show date, both seats, decks, winner, step count. No turns, duration, play/draw or stats columns. Win/draw/loss only inside one play session. | `routes/games/+page.svelte:236-275`, `attempts.py:484-528,629-798` |
| Review board | One `Board` for Play, Review, Watch and Study; Transport, Strip and Rail already draw recorded probabilities on cards and rows, and a value line per chooser. | `board/Rail.svelte:191-269`, `board/Strip.svelte`, `board/replay.ts:115,253` |
| Policy evidence | `BotDecision` (probabilities, value, value semantic) is recorded only where a bot chose. A human's choice has none, and Play recordings hide the opponent's. | `server.py:1864`, `recording.py:194-197`, `trace.py:384` |
| A policy in your seat | `PolicyRegistry.load(key, setup, seat=…, seed=…)` returns a fresh `PolicyRunner` per seat; without search it reads only `self.view(obs)` plus `env.capture_decision(seat, 0)` and returns a `BotDecision`. | `managym/policy.py:286-323,379-423` |
| Value from the policy | manabot's `BaselineSession.evaluate` returns probabilities and a `ValueEstimate(value, seat, semantic)`; semantic is `signed_outcome` ([-1, 1]), `win_logit`, or undeclared. Nothing converts a value to a win %. Scripted players return neither. | `manabot/sim/policy_provider.py:179-255` |
| Review loses your hand | A consequence board reads the next event's `observation`, the **actor's** view; after your move, if the opponent acts next, your side is their redacted view of you. | `recording.py:341-347` |
| Opponent moves unnamed | A Play recording's move list collapses them to "Opponent decision ×N". | `trace.py:377-383` |
| Both hands | `analysis_observation` is captured only in Watch mode. | `server.py:1582-1588` |
| `/api/advice` | Fixture-only belief/search contract; no UI calls it. | `advice.py:1011-1101` |
| Study branching | Needs the live session's in-memory cloned roots (15-minute TTL). Reveal's evidence provider is "unavailable" outside the e2e fixture. | `server.py:1644-1667,4055-4066`, `study_runtime.py:45-53` |

Not verified: that a real played game re-executes to the same boards. This
checkout has no built extension, so nothing was run. Slice 1 exists to
answer it.

## The design

### 1. A played game is a record

`record_of(attempt) -> GameRecord` builds the engine's own record from what
the row already stores: seed, the two `PlayerSetup`s in native seat order,
`actions = [event.action for event in events]`, `skip_trivial=True`,
identities from `world_json`, and the end (winner, or interrupted with its
reason). It lives in `managym/academy/recording.py`. No new stored field and
no change to how a game is saved, so it works for games played before this
change.

`RecordWalker` re-executes it. A played game has no state digests, so the
reading checks itself against what the row does hold: at each step the
acting seat's re-executed board must equal the stored `observation`, the
action's description must equal `action_description`, and the result must
equal the stored winner. The first difference stops the reading there, as an
imported record stops today.

What a reading is not: a replacement for the stored track. The boards the
person saw stay in the row and Review still opens from them. A game from
another world, or one that diverges, opens exactly as it does today, with
one line saying which build recorded it and that the derived parts
(your-view boards after an opponent's step, both hands, new analyses) need
that build. Saved analyses and summaries are rows of their own and still
show.

### 2. Readings

One function walks the record and hands each step to whoever asked. Three
things ask.

- **Views.** `env.observation_for_player(seat)` at every step. Review's
  consequence boards read the person's own view, so your hand stays whoever
  acts next. Opponent moves are named from the person's view of the step
  (invisible steps fold into the move where they become visible, as MTG-384
  round 7 decided). After authoritative completion, **Show both hands** in
  the Perspective control reads both seats' views, as it does for a Watch,
  and the opponent's recorded probabilities show at its choices.
  Views are derived when a recording is opened and cached in memory with the
  open `Recording`; nothing is written.
- **Analysis.** See section 3.
- **Numbers.** `managym.statistics` reducers over the same walk: the
  existing `Behaviour` (lands played, attacks, each with its opportunities)
  and one new reducer beside it for turns, spells cast, damage dealt and
  taken and cards drawn per seat. They are engine reducers, usable by any
  managym loop, not Academy code. The result is saved once, at completion,
  as the game's `numbers_json`, so `/games` lists never re-execute.

Access keeps today's rules. A reading from the person's seat shows nothing
they did not see. Both seats' views need what `reveal_hidden` needs now: an
authoritatively complete game.

### 3. Analysis: a policy reads your decisions

An **Analysis** is (game, seat, policy) → for each non-forced decision of
that seat, the policy's probability for every legal offer and its value.
It is made by walking the record with a `PolicyRunner` for that seat: at
each of the seat's decisions the runner is called on that seat's
observation and the walk's `Env` for `capture_decision`, its `BotDecision`
is kept, its sampled action is discarded, and the walk applies the action
that was actually played. A decision-memory policy therefore sees the same
history a bot in that seat would have.

- **When.** Opening Review on a finished game starts an Analysis by the
  default policy if none is saved; Review is usable at once and marks fill
  in as the walk advances ("Analysing 34 of 81"). **Analyse with…** in
  Review's Menu names any other registered policy. Several analyses of one
  game can exist; a selector switches which one the marks and line show.
- **Default policy (for review).** The opponent's, when the opponent is a
  registered policy that returns probabilities. Against Random, Passive,
  Heuristic or Search there is no default: Review says "No analysis yet"
  with **Analyse with…** beside it, and with no registered policy installed
  (a managym-only install) it says so in words.
- **Stored** in a new `analyses` table in `play.sqlite`: attempt id, seat,
  the policy's full descriptor (ID, runner, artifacts), status
  (`running`/`complete`/`stopped at N: reason`), one row of evidence per
  decision index, created time. It is never part of `trace_json`, any frame
  or any trace export, so a game's frames and stored row are untouched by
  analysis.
- **No search.** `search_decision` plays out from the walk's `Env`, which
  holds the opponent's real hand; an analysis by search would advise from
  hidden cards. Refused with that reason until belief-conditioned
  determinization exists (the seam `advice.py:546` names).
- **Failure and recovery.** A runner error stops the analysis at that
  decision with its reason; decisions before it keep their evidence and
  **Analyse again** restarts it. A server restart leaves a `running` row
  with no process: it reads as stopped and can be run again. A policy that
  is no longer registered cannot make new analyses; saved ones still show,
  under the descriptor that made them.
- **Who may ask (built narrower, for review).** The participant, of their
  own game once it is no longer being played (`analysis_game_active`; live
  is slice 6). Readers of a shared game see saved analyses but cannot start
  one, and only the person's seat is analysed (questions.md, slice 2 choices).
- **Cost.** One forward pass per decision, off every game's frame path, on
  one thread per analysis (`server.py` `start_analysis`). One analysis runs
  per game at a time.
  Slice 2 measures a whole analysis on the fixture policy and on whatever
  checkpoint is registered; nothing here bounds play.

Types: the per-decision evidence is `managym.recording.DecisionAnnotation`
(probabilities, value), the type records already use for "what a policy
recorded about a decision", plus the value's semantic. `BotDecision` is not
reused: it names an action that was chosen, and an analysis chooses nothing.

Passed over: making `/api/advice` live. Its contract is belief scenarios and
search advice bound to checked fixtures. It stays as it is.

### 4. Review: the game, read against the policy

Review of a played game (the result card's **Review this game**, or Games),
on the existing board, read-only:

- **Move list** (rail). Every move is named. Each of your decisions carries
  one mark in Work Sans beside the Spectral move text:
  - ✓ "Policy's first choice" when you played the argmax;
  - the policy's mass on your move otherwise ("8%"), and the row's detail
    names its first choice ("Policy preferred Cast Lightning Strike, 62%");
  - nothing where the analysis has no evidence.
  No "brilliant" or "blunder" names **(for review)**: a policy's probability
  is its preference, not an evaluation of each alternative. The one derived
  ranking is the summary's "largest disagreements", the five decisions where
  the policy's first choice had the most mass over yours; each jumps to its
  position.
- **Your recorded choice** shows each offer's probability on its card and
  row, yours marked, under the heading "Analysis by <policy> · <artifact>",
  never "Recorded choices".
- **Value chart** (Strip). One line from your seat at each of your decisions.
  `signed_outcome` maps (v + 1) / 2 and is labelled **Expected score**.
  `win_logit` maps through the logistic and is labelled **Win chance**.
  An undeclared semantic draws no line and no option estimates (built;
  for review); probabilities still show.
  The percentage axis is fixed 0 to
  100 with a 50 rule; the caption is "<policy>'s estimate from your seat · N
  of M decisions". Turns stay ruled; click jumps. The opponent's own
  recorded value line appears only under **Show both hands**, flipped to
  your seat, in the tertiary ink.
- **Summary** (the result card and Review's Menu): turns, play or draw,
  duration, lands played, spells cast, attacks, damage dealt and taken,
  cards drawn; and from the selected analysis: decisions, agreement (your
  move was the first choice, k of n), and decisions made with analysis
  shown.

### 5. Games: a history you can read

`/games` keeps its filters and addresses. Rows for your games gain turns,
duration, play/draw (from `numbers_json`) and agreement (from the game's
default analysis, blank when none). Above the list, for **My games**, a
record table: wins, losses and draws over completed games by opponent
identity and by matchup, each with the win rate and its 95% Wilson interval,
across sessions. Counts of recorded games, never a rating. Games with any
decision made while analysis was shown are counted in their own column.

### 6. Analysis while you play (for review; built last)

The brief leaves open "whether allowed live, only post-game, or recorded
when used". Chosen: allowed, off by default, recorded when used.

- Off at the start of every game. No advising policy is loaded and nothing
  about one is in the server's session, so there is nothing to seal.
- **Show analysis** in the rail turns it on for this game. If the opponent
  cannot analyse the game, offer the eligible policies from **Analyse with…**.
  If none are available, link to policy setup. The server
  starts this game's Analysis early: it walks the game so far in the
  person's seat (the same walk as section 3, which also gives a
  decision-memory policy its history), then reads each new decision of
  theirs as it arrives. Offers carry the policy's probability on card and
  row under "Analysis by <policy>"; nothing is selected, raised, focused or
  reordered, and no key commits "the policy's move". The value chart's live
  points appear in the folded Strip.
- A decision committed while analysis was visible is recorded
  `analysis_shown` on its `TraceEvent`. Beside `committed_at` it is the
  only new stored field in a game: a fact about what happened that no
  re-execution can recover. Review marks the move "analysis shown", the summary counts
  them, and Games shows it. Turning it off stops the marking from the next
  decision; it cannot unmark.
- A slow or failed pass never holds the game: the decision's frame goes
  first, the probabilities follow in their own message, and a decision
  committed before they arrived is not marked.
- A Study branch keeps its own rule: evidence stays sealed before Reveal
  (`study.spec.ts` `FORBIDDEN_BEFORE_REVEAL`).

Passed over: post-game only (loses the tutor that "an overlay for what the
policy would do in your seat" describes); always-on (the result would stop
being yours); a separate "practice" game kind (a second kind of game to
explain where one flag per decision says the same thing more exactly).

## Source of truth and consumers

| Thing | Where it lives | Kind |
|---|---|---|
| What happened | The attempt row's `trace_json`, as today | recorded |
| `committed_at` | `TraceEvent`, optional UTC ISO 8601, absent on older rows | recorded (built, slice 1) |
| `analysis_shown` | `TraceEvent`, optional, absent = false | recorded (new, slice 6) |
| Any seat's view at any step, named opponent moves | Re-executed on open | derived, not stored |
| `numbers_json` | New nullable column on `attempts`, written at completion | derived, saved |
| Analyses | New `analyses` table | derived, saved |

`RECORD_VERSION` stays 2. Old rows have no `numbers_json`; it is computed on
first open where the game re-executes, otherwise the columns are blank.

Consumers that change: `Recording` (boards and narration from views;
`value` at the person's choices from the selected analysis); the
`/api/traces` summaries and a record aggregate for My games; new
`/api/traces/{id}/analyses` (list, start, read); the WebSocket's
analysis message when shown live; `recording.ts`, `board/replay.ts`
(`offerEvidence`, `stripMarks`), Rail, Strip, the result card, `/games`;
`managym/statistics.py` (one reducer). Consumers that must keep working
unchanged: saving a game and every frame of a game (nothing in this plan
touches them until slice 4, and then only when analysis is shown); Watch
and imported recordings; Study branches and their sealed evidence;
`/api/advice` and its fixtures; the shared-table protocol; `GameConfig` and
setup.

Academy imports no manabot or Torch: an analysing policy arrives through
`managym.bots` like any opponent.

## Absent and error states

| State | Shown |
|---|---|
| No registered policy that gives probabilities | "No analysis: no registered policy can read this game" and how to install one |
| Opponent was Random, Search, Passive or Heuristic | "No analysis yet" with **Analyse with…** |
| Analysis running | Marks and line fill in; "Analysing N of M" |
| Analysis stopped at a decision | Evidence up to it; the reason; **Analyse again** |
| Game recorded by another world or card set | Today's Review from the stored track; one line naming the build; saved analyses shown; **Analyse with…** disabled with that reason |
| Re-execution diverges at step N | Derived views up to N, the stored track after; the line says where it stopped |
| Game stopped or interrupted | Review to the last position; the participant may analyse the prefix; no both hands; no win or loss invented |
| Value semantic undeclared | No line and no estimates; probabilities only |

## Operational boundary

- Opening a recording re-executes it once. Bound: a 300-event game's views
  add under 500 ms to the first open on this laptop. Slice 1 measures it on
  ordinary recorded games; over the bound, views are built on first use of
  a position, not on open.
- Analysis runs in the existing worker pool, one per game at a time, and
  stops when the server stops. No load is induced to measure it.
- The row rewritten on every save does not grow. The append-only events
  table stays its own work.

## Exclusions

- **Open hands during play.** Face-up hands in a live game change what a
  seat may know and what the opponent's policy observes; that belongs to the
  native disclosure authority (`GameSetup`, beside `decklists`). Both hands
  arrive after the game.
- **Another line from a finished game**, and a live evidence provider for
  Study's Reveal. Both are now a walk to decision N (a root, or a reading)
  and each still gets its own Task.
- **Card-level statistics** (`CardPerformance` over played games) and
  analysing a seat of a watched or imported game: readings this design makes
  possible and this Task does not build.
- **Dropping the stored boards** from a played game's row. The record makes
  it possible; it needs an answer for games from older worlds first.
- An analysis by search, move grades, ratings, mobile layout, the outer
  chrome (MTG-388).
- No change to frozen worlds, checkpoints, fixtures or receipts.

## What goes

- The Play recording's "Opponent decision" labels and "×N" collapse, where
  the game re-executes.
- `Recording._raw_board`'s read of the next actor's observation as the
  person's board, where the game re-executes.
- The Strip's self-choosing y scale (`Strip.svelte:28-34`), kept only for
  undeclared semantics.
- From the kickoff plan, never built: the `advisor` config field and
  Setup's advisor picker, `advisor_decision`, `person_observation`, the
  Play-mode `analysis_observation`, their sealing in `prepare_trace_payload`,
  the 1.3× row bound and the 100 ms per-decision bound.

Must survive: every existing recording opens as it does today; Command
binding and stale rejection; replay and fork identity; the bytes of every
frame of a game in which analysis was never shown, and its saved row apart
from each event's `committed_at`.

## Shipping docs

- `docs/academy.md` — Review, policy analysis, both hands and Games.
- `managym/academy/README.md` — verified replay, access and analysis lifecycle.
- `docs/engine-python.md` — replay reducer example and count definitions.

Docs review question: “When a policy supplies a signed outcome, what should
the chart call the estimate given that draws count for half a win?” The draft
uses **Expected score** for `signed_outcome` and **Win chance** for
`win_logit`. Keep the same fixed 0–100 scale and coverage caption.

Docs review question: “Which policy should Show analysis use when the live
opponent cannot analyse the game?” The draft uses the same eligible policy
choices as **Analyse with…** and points to policy setup if none are available.

## Direction (2026-11-24)

Jack Heart, on `scratch/explore.html` question 1: "1 is the right general
thread." Then, after the chess.com side-by-side (`scratch/chesscom.html`):
"the models are currently worse than me. So i think the approach of showing
how moves are rated visually and the data viz is helpful, but te frame should
not be that if i disagree its a mistake. its just taht contextualizing what the
agent things against what i actually did helps me understand what the agent is
doing". Answers to that page's questions: "1a yes / 2 yes / 3 sure / 4 off by
default but available for sure".

What this changes in the design above (it overrides sections 3, 4 and 6
where they differ):

- **Review reads the agent, not the person.** No grades, accuracy score,
  rating, "loss" numbers or red/amber marks. A decision's mark is neutral:
  "same" when the person chose the agent's first choice, "differs" otherwise,
  plus a two-dot track on one 0–100 scale (the agent's estimate after the
  person's choice, and after its own). Chart dots mark where they differed,
  in the neutral token. "Largest disagreements" become key moments: where
  the two differed most and where the estimate moved most.
- **Each option is rated.** At every analysed decision, every legal option
  carries the agent's probability and its estimate after that option
  (expected score from the person's seat), with "you" and "its choice"
  labelled. The estimate costs one value read per legal option: apply the
  option to a copy of the walk's `Env` and read the value from the person's
  view. Unverified: whether a manabot policy gives a value where the person is
  not the chooser; if not, read at the person's next decision after one
  sampled reply, and say so in the caption.
- **Summary**: decisions with a choice, "chose as the agent would k of n",
  the agent's estimate at start, peak and end. No mean loss.
- **See its line** (in this Task): from any analysed decision, a Study
  branch with the agent's choice played; the person plays on or lets the
  agent keep choosing in their seat. The played game is unchanged. A
  cross-game list of differences is a later Task.
- **A time on every step**: `TraceEvent` gains an optional commit time
  (absent on older rows). Nothing shows it yet; it is stored so time per
  decision can exist later.
- **Analysis during play**: available, off by default at the start of every
  game, each decision made while shown recorded `analysis_shown` (section 6
  stands).
- **MTG-423 is absorbed** (Task definition edited 2026-11-24): in Watch,
  **Show both hands** must show the opponent's revealed cards readably and
  by keyboard in the top seat (today it shows only "Hand 7"), and a
  recorded position that branches by determinization (decision 38 of the
  69-decision Elemental snapshot replay from manabot PR #577) gets a
  neutral boundary on the timeline. Keep acting-view privacy, history,
  chooser identity and timeline navigation; seeds and private intervention
  data stay out of browser responses. Verify on an ordinary private
  recording and a branched replay.

## Status (2026-11-24, after slice 1)

Slice 1 is built. The design's foundation holds: every kind of played game
tried re-executes to the stored boards and result (Random, Passive, Search 4,
second seat with closed decklists, priority-stop automation, three
UR Lessons v GW Allies games of 185–300 events). Docs are trimmed to what is
built; the 2026-11-18 text for later slices is in commit `76f7f06d54`
(`git show 76f7f06d54 -- docs/`), to bring back slice by slice.

What slice 1 taught:

- **Cost.** One verified step costs two board projections
  (`decision_surface_view` ≈ 3 ms each, mostly Pydantic validate/dump and
  `definition_names`). A whole 300-event game re-executes in ~1.0 s (2.8 s
  before reusing the acting board), over the 500 ms bound, so views are built
  lazily as the plan allowed: `ReplayedViews.board(i)` walks only as far as a
  read asks. Opening costs nothing; stepping costs one step. Slice 2's
  analysis walk will pay the same per step, plus a forward pass.
- **Narration needs no replay.** The stored acting boards, redacted to the
  person, already name the opponent's moves (MTG-384 round 7). Narration and
  history labels stay on the stored track; only `read`'s board uses replayed
  views. So listing positions never walks.
- **No concession exists** in Play; an abandoned game (`new_game`) is the
  interrupted case and is tested. An imported deck was not tried
  (`hero_source`); it goes through the same `GameConfig.to_rust`.
- **Show both hands** now appears at any position of a finished played game,
  but the top seat still draws the opponent's hand as "Hand N": the MTG-423
  rendering fix (slice 5) is what makes it readable, for Watch and Play alike.
- The opponent's choice positions (with its recorded probabilities) under
  Show both hands were in the WIP patch and were dropped: positions are
  listed per perspective, and adding hidden ones needs its own design.
- `/verify` was not run before committing: in this repository it is the
  post-merge Loopflow check that finishes the Task.

## Status (2026-11-24, after slice 2)

Slice 2 is built (`285724184c`, `2189ffc059`): `managym/academy/analysis.py`
(`analyse_game`, `offer_estimates`, `analysis_choices`, `run_analysis`), the
`analyses` and `analysed_decisions` tables in `AttemptStore`,
`GET/POST /api/traces/{id}/analyses`, and Review's rated options,
same/differs marks with the two-dot track, and the fixed 0–100 strip.
Tests: `tests/academy/test_played_analysis.py`, `replay.test.ts`.

What slice 2 taught:

- **The value question (§Direction) is settled.** A manabot value belongs to
  the chooser: `evaluate_observation` reads the acting viewer's observation
  and its legal actions. So "the estimate after an option" is read at the next
  decision on a clone: the person's own value when they choose next, else the
  opponent's value negated (`1 - score`, exact for both semantics), in a world
  `determinize`d from the person's perspective. No reply is sampled. With
  closed decklists nothing can be re-dealt, so those estimates are absent
  (tested). The session's history-free `evaluate` (the Search contract) is
  required for estimates; a policy without it, or recurrent, gets
  probabilities only.
- **Cost.** With the W/D/L test checkpoint (`write_search_checkpoint`, 8 wide)
  a whole 88-event Lessons v Allies game (46 decisions, 135 options) analyses
  in 3.0 s on the M4 Max: 65 ms per decision including a value read per
  option. A current-world checkpoint is not on this machine; cost grows with
  model width and options per decision.
- **The merge removed `world_version`** from the content-pack manifest, which
  had broken slice 1's `record_of` on this branch (every played game diverged).
  Fixed: the card digest is the identity; `recorded_build` names the build.
- **Untrained policies give ~50/50**, so the browser proof shows a flat line;
  it proves wiring, not readability of real estimates.
- Browser proof (throwaway server, built client): Review auto-started the
  opponent's analysis, 36 move marks, 46 strip points, "Analysis by …" with
  per-option probability and estimate. Shots in `scratch/shots/review-analysis-*`.

Not built from this slice's plan text (recorded in questions.md): analysis of
the opponent's seat; the raw-scale line for an undeclared semantic (no line is
drawn); a reader starting an analysis of a shared game (participant only);
the summary's agreement count (slice 3); key moments navigation.

Remaining from slice 2 (found on realign, 2026-11-24):

- **(fixed, c56051de0c)** Every poll re-executed the game. The eligible
  policies, default, `unavailable` and analysed events moved to
  `GET /api/traces/{id}/analyses/policies`, read once when Review opens;
  `GET /analyses` now returns only `{analyses}` from the store (404 for an
  unknown game). Test: `test_following_an_analysis_does_not_re_execute_the_game`.
- `_ANALYSING` is a mutable module global in `server.py`; it guards "one per
  game" only within one process. Acceptable beside the server's other
  process state; fold into the store (a `running` row owned by this process)
  if it is touched again.

## Status (2026-11-24, after slice 5)

Built in dabcf4429f:

- The top seat's **Hand** in a plate opens its visible cards (Show both
  hands, or a card revealed to the viewer) in the zone dialog, like the
  graveyard, by pointer or keyboard; a partial view reads "2 of 7".
- `RecordedDecision.interventions` carries the record's interventions at
  each boundary. `import_record` stores the branching kinds (determinize,
  shuffle_libraries, reseed; `trace.BRANCHING_INTERVENTIONS`) as
  `TraceEvent.branched` on the decision they precede, never seeds or
  viewers; position metadata exposes `branched`; `replayLog` puts a neutral
  `system` note "Branch · libraries reshuffled" before that decision, which
  jumps to it (`entryPosition`).
- Verified on a constructed branched replay, not the 69-decision Elemental
  snapshot (not found on this machine): `test_watch_publishes_the_determinized_offers`
  (determinize at the last decision only) and the configured-launch e2e
  snapshot spec (shuffle at the first decision, hand dialog by keyboard).
  The ordinary private recording path: `tests/academy/test_played_replay.py`
  passes unchanged.

Checked on realign (2026-11-24): `test_played_analysis.py`,
`test_played_replay.py`, `test_game_snapshot.py` 54 passed; `replay.test.ts`
18 passed. The configured-launch e2e spec was not run here (gate).

Next: slice 3 (numbers): a `managym/statistics.py` reducer over the walk
(turns, spells cast, damage dealt/taken, cards drawn per seat), nullable
`attempts.summary_json` written at completion (computed on first read for old
rows), the summary on the result card and Review's Menu including "chose as
the agent would k of n" from the selected analysis, Games' columns and the My
games record table with Wilson intervals (§5). Then slice 4 (See its line),
then slice 6.

## Status (2026-11-24, after slice 3)

Built in fde4bb32f5, a2226d0a13, 2c74f2ca2d:

- **Engine.** The exposure capture counts damage dealt to players
  (`player_damage_taken`, `player_damage_dealt`; `flow/damage.rs` →
  `Game::player_damage_capture`). `managym.statistics.Totals` reduces turns,
  spells cast (cards put on the stack), cards drawn and damage per seat.
  Test: `tests/env/test_record_reducers.py::test_totals_…` (live equals replayed;
  damage taken equals life lost in Bolt v Bolt).
- **Academy.** `managym/academy/numbers.py` (`game_numbers`, `GameNumbers`,
  `wilson_interval`); `attempts.numbers_json`, saved at completion and on first
  `GET /api/traces/{id}/numbers` for older games; list rows carry `numbers`;
  `GET /api/traces/record` (completed games by opponent and both decks).
  The step check shared with analysis moved to `recording.step_matches`.
  Named "numbers" because the client's `GameSummary` already names a list row.
- **Client.** `$lib/numbers.ts`; the result card's line comes from the server
  (the move-list tally in `result.ts` is gone); Review's Menu **This game**
  (counts table, play/draw, duration, "chose as <policy> would k of n", the
  estimate at start/peak/end); Games rows show turns and duration, Player 1 is
  labelled on the play, **My record** above My games.
- Cost: one walk with card capture, ~5–20 ms for 45–75 event Bolt games.
- Browser proof (static build, throwaway server, Reader fixture policy, three
  games): `scratch/shots/review-numbers-{light,dark}.png`,
  `games-record-{light,dark}.png`.

Not built (questions.md): Games' agreement column; the record split by
opponent alone and by matchup alone (one table grouped by both).

Remaining outside the numbered slices (found on realign, 2026-11-24): key
moments are not built: the plan's "largest disagreements" (where the person
and the agent differed most, and where the estimate moved most) with **Next
key moment** in Review, which the demo uses. Build it with slice 4, since both
start from an analysed decision. The Games "analysis shown" column belongs to
slice 6.

Next: slice 4 (See its line) with key moments, then slice 6.

## Status (2026-11-24, after slice 4)

Built in 9308d9f73a (docs in the commit after it):

- `analysis.agent_line` and `POST /api/traces/{id}/analyses/{analysis}/line`:
  re-execute to an analysed decision, play the policy's first choice on a
  copy, continue (policy in the seat, `build_villain_policy` in the other) to
  the end of the next turn; each step carries the seat's own board. Nothing
  saved. Test: `test_see_its_line_plays_the_policys_first_choice_on_from_a_decision`.
- Review: **See its line** at an analysed decision draws the line on the same
  board (move list, Previous/Next, arrow keys, Escape/Return to the game);
  **Next key moment** and a "Key moment · …" line (`keyMoments`,
  `analysis.test.ts`).
- Cost: a line on a 90-event game is 30–100 ms with the fixture policy.
- Browser proof (static build, throwaway server, Lessons v Allies vs Heuristic,
  Reader analysis): `scratch/shots/review-key-moment-{light,dark}.png`,
  `review-line-{light,dark}.png`.

What it taught: the existing Study branch cannot be reused for finished games
(its roots live in the live session for 15 minutes), so the line is a
read-only play-out.

Remaining from slice 4 (realign, 2026-11-24): **playing on from the line.**
The approved chess.com table (`scratch/chesscom.html`, answered "1a yes / 2
yes / 3 sure") brings "Analysis board: Study's branch, opened from Review at
any decision", and the direction says "the person plays on or lets the agent
keep choosing in their seat". Not built: a playable branch rooted in a
re-executed position (a `StudyForkProvider` over `agent_line`'s walk, or a
Play game whose prefix is the record's; questions.md, slice 4 choices).
Acceptance keeps it; build it after slice 6.

Next: slice 6 (analysis while you play), then playing on from the line.

## Status (2026-11-24, after slice 6)

Built in this pass (three commits after 320c57885d):

- **Server.** `analysis.LiveAnalysis`: a worker thread with its own walk.
  It loads a fresh runner, catches up over the game so far
  (`seat_decisions`, now shared with `analyse_game`, which also uses
  `read_decision`), then applies each committed step and reads each decision
  `_publish_current_prompt` publishes to the person. The shared-table request
  `show_analysis {shown, policy}` (pilot; schema and conformance fixture
  regenerated) answers `analysis_state`. Readings go to the pilot as their own
  `analysis` message through `asyncio.run_coroutine_threadsafe`, never waiting.
  A send that completes marks the event shown. `TraceEvent.analysis_shown` is
  written only when true. Frames carry `analysis: {policy}` while it is on.
  `GameNumbers.analysis_shown`, the position index's `analysis_shown`, and
  `/api/traces/record` `with_analysis` read it.
- **Client.** The rail's **Show analysis** switch, with a policy menu when
  several can read the game. Live evidence goes through `analysedEvidence`
  with no chosen offer, and only for the frame revision it was read at.
  Review marks moves "analysis shown" (`analysisShownMoves`), Review's Menu
  counts them, and Games rows and My record (**With analysis**) show them.
- Tests: `test_shown_analysis_reads_each_decision_live_and_marks_it` (a live
  reading equals the post-game analysis at the same events; only decisions
  read while shown are marked; turning it off stops marking),
  `test_a_game_played_without_analysis_stores_no_mark`,
  `test_showing_analysis_needs_a_policy_that_can_read_the_game`;
  `analysis.test.ts` (live state) and `replay.test.ts` (shown moves).
- Browser proof: a static build on a throwaway server
  (`scratch/probe/live_server.py`) with the Reader fixture policy as
  opponent. Two choices were made with analysis off, three with it on, then it
  was turned off and the game played out. Review showed 3 "analysis shown"
  moves, the Menu said "Chosen with analysis shown: 3 decisions", the Games
  row said "analysis shown at 3 decisions", and My record had 1 game with
  analysis. Screenshots: `scratch/shots/{play,review,games}-analysis-shown-{light,dark}.png`.

What it taught:

- **The first reading took 6–11 s** on this machine with 23 discovered
  checkpoints. Admitting every registered policy (`analysing_policies`) took
  2.5–2.9 s and runs under the game lock. It is now cached once per match.
  The rest is runner load and the catch-up walk, which run off the lock.
  Review's `/analyses/policies` pays the same admission on every open.
- The pre-existing `test_control_fixture_round_trips_through_checked_schema_and_python`
  fails on the base commit too: `resume` and `rematch` dump default `None`
  fields that the fixture omits. Not caused by this slice.

Not built from §6 (questions.md, slice 6 choices): the folded Strip's live
value points; the e2e acceptance spec `study-review.spec.ts`, including its
second test (Show analysis mid-game → "analysis shown" in Review and Games).
That spec still does not exist.

Next: playing on from the line (slice 4's remainder), then the
`study-review.spec.ts` acceptance spec.

## Slices

1. **(built)** **A played game plays again.** `record_of`, the walk's self-check
   against the stored track, and views: Review keeps your hand, names every
   move, and **Show both hands** works on a completed played game. This is
   the slice that can prove the design wrong. Its test re-executes finished
   games of each kind (against Search 16, against the fixture policy, with
   an imported deck, one with priority-stop automation, one conceded) and
   requires every step's board and the result to agree. If a kind of game
   cannot be reproduced, find out why before anything else: a cause in the
   engine or in `GameConfig.to_rust` is a bug to fix; if a played game truly
   cannot be re-executed, stop and say so, because sections 2, 3 and 6 all
   rest on it. The fallback is the kickoff plan's: record, live and sealed
   until completion, the person's view and the spectator view on every
   step and a second runner's `BotDecision` at the person's decisions
   (the fields listed under "What goes").
   Also stores a commit time on every new `TraceEvent`.
   It shows: play a game against Search 16, open Review, step past your
   first move while the opponent acts, your hand is there; the move list
   reads "Search 16 cast …"; **Show both hands** turns the other hand up.
2. **(built)** **Analysis.** The `analyses` table and endpoints, the walk with a runner
   in the person's seat, the default on opening Review, **Analyse with…**;
   Review's rated options (probability and estimate per option), neutral same/differs marks with the two-dot track, and the value chart. Proof uses the
   policy-runtime fixture server (`tests/fixtures/policy_runtime_server`),
   since no current-world checkpoint exists on this machine; the time for
   one whole analysis is recorded.
3. **(built)** **The game in numbers.** The reducer, `numbers_json`, the summary on the
   result card and Review's Menu, Games' columns and the My games record
   table with intervals.
4. **(built, read-only)** **See its line.** From an analysed decision, the
   agent's choice played and its line to the end of the next turn, plus key
   moments. Remaining: play on yourself from it (after slice 6).
5. **(built)** **Watch: revealed hands and branch boundaries** (MTG-423). Independent of
   slices 1–4; may go first if slice 1 stalls.
6. **(built)** **Analysis while you play.** The toggle, the catch-up walk, the live
   message, `analysis_shown` and its marks. Remaining: live points on the
   folded Strip.

Each slice updates `docs/academy.md` and `managym/academy/README.md` where
it changes what they describe (slice 3 also `docs/engine-python.md` for the
reducer), and removes what it replaced.

## Acceptance check at gate

Headless, on isolated ports, built client, origin `automated_validation`:

```
ACADEMY_POLICY_PORT=8389 npm --prefix academy/ui run test:e2e:policy -- study-review.spec.ts
```

(`playwright.policy.config.ts` serves the static build from the policy
fixture server and selects specs by `testMatch`; the new spec joins that
list.) One scenario against the policy fixture server: start a game against
the fixture policy; play to the end, asserting no frame or message carries
analysis; open **Review this game**; assert (a) the position after your
first move shows your hand, (b) the opponent's moves are named, (c) once
"Analysing" finishes, your choices show probabilities under "Analysis by …"
with your move marked, (d) the Strip's line is on a 0–100 axis with one
point per analysed decision, (e) **Show both hands** reveals the opponent's
hand, (f) the summary's "chose as the agent would" equals the count of "same" marks, and no mark or caption grades the person; then open
`/games` and assert the row's turns and the record table's 1 game. A second
test turns **Show analysis** on mid-game and asserts the committed decision
reads "analysis shown" in Review and in Games.

Beside it: `uv run --extra dev pytest tests/academy/<the changed files' tests> -q`
(slice 1's re-execution test over each kind of game; a saved row of a game without shown analysis byte-equal to base apart
from `committed_at`; an old-world row reads as today;
a both-hands read of an unfinished game refused; an analysis by search
refused), `uv run --extra dev pytest tests/<the reducer's test> -q`,
`npm --prefix academy/ui test`, `npm --prefix academy/ui run check`, and
`npm --prefix academy/ui run test:e2e:release` for the surfaces that must
not move.

## Done when

- Review of any played game that this build can re-execute never loses your
  hand, names the opponent's moves, and shows both hands once the game is
  over; games played before this change included.
- Any finished game can be analysed by any registered policy, after the
  fact and more than once; Review shows what that policy would have done at
  each of your decisions, how its estimate of your position moved, and
  where you and it disagreed most.
- Games answers "how am I doing against this policy with this deck" across
  sessions, with honest intervals.
- Analysis while playing is off until asked for, never selects, and every
  decision made with it visible says so in Review, in the summary and in
  Games.
- A game in which analysis was never shown produces the same frames and the
  same saved row as before this change (apart from each step's
  `committed_at`), and every older recording opens as it did.

## The demo

`./scripts/play`, with a registered current-world policy as opponent; play a
game; press **Review this game**. The board opens at turn 1; "Analysing"
counts up and the value chart draws under the board; the move list has "same" or "differs" with its two-dot track beside each of your moves; Next key moment jumps to the decision, where the policy's distribution sits on
your cards; **Show both hands** turns the opponent's hand face up. Then open
a game played last week against Search 64 and press **Analyse with…**.
Screenshots of each, light and dark at 1280×800, go to `scratch/shots/`.
