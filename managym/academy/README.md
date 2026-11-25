# academy

The Academy experience server: authoritative play, presentation, and
study. The Svelte client in [academy/ui/](../../academy/ui/) renders what this
package decides; the client never invents rules meaning.

## Authority model

One versioned experience protocol drives direct play, replay, and decision
inspection as projections of the same authority. The server owns legality,
hidden information, and state; the client receives `ExperienceFrame`s and
`PresentationEvent`s and returns `Command`s bound to the exact frame that
offered them. Recovery envelopes make sessions reloadable — including fully
offline. Schemas and three-language conformance fixtures live in
[protocol/](../../protocol/README.md).

Before choosing an action, read the scheduled turns and extra combats beside
the turn heading. These are already committed events in execution order;
later effects can change them. Turns and combats are separate lists. A current
or pending restricted combat states which creatures may attack; ordinary attack
offers remain authoritative. After reconnecting, the schedule comes from the
recovered game state. Older recordings without schedule data say “Schedule not
recorded,” rather than claiming nothing is scheduled.

When you control another player, the board stays in your perspective. The
decision context names who chooses for whom and whose resources apply. **Search
cards** introduces the native search instructions without assuming whose zones
are searched. Private inspection labels retain each card's owner and zone;
inspection does not make an opponent's cards yours. Reconnect resumes the
pending choice with the same legal offers and permitted information.

## Modules

- **`server.py`**: FastAPI WebSocket server (`/ws/play`) — sessions,
  authoritative game loop, viewer-safe observation serialization
- **`presentation.py`**: semantic presentation events (combat, turn
  transitions, spell sequences, land plays, discards and reveals) derived from engine
  events
- **`curated_pack.py`**: the frozen curated matchup asset pack
- **`study_protocol.py`**: viewer-safe study artifacts and decision evidence
- **`study_branch.py`**: source-bound historical forks, native structured
  execution, typed failure, and consuming exact-return receipts
- **`trace.py` / `attempts.py`**: viewer-safe trace projection and SQLite human
  attempts/feedback (`ACADEMY_TRACES_DIR/play.sqlite`, default `academy/traces/`).
  New records have one SQLite authority; legacy JSON remains readable.
  `attempt_players` identifies both seats with stable public IDs, types, recorded
  names, decks and bot versions. Game resume credentials and browser player credentials
  remain private authorization values. Shared history applies permissions in SQL
  before filtering/pagination; only completed shared replays are readable by
  nonparticipants. `ACADEMY_PLAY_RECORD_ORIGIN` distinguishes declared automated
  validation from human play; absent provenance is unknown.
  A live game is saved when its match, revision or event count changes, not on
  every projection, and each event, decision and presentation event is
  encoded to JSON once (`append_json.py`); the stored bytes are what
  `json.dumps` of the whole record gives.
- **`recording.py`**: positions over retained recordings; `record_of(attempt)`
  builds the engine's `GameRecord` from a played game's stored seed, setups
  and actions, so a played game is re-executed like any other record.
- **`analysis.py`**: a policy's reading of one seat of a played game
  (`analyse_game`), saved per decision in the `analyses` and
  `analysed_decisions` tables beside the game, never in its trace; and the
  same reading during play (`LiveAnalysis`).
- **`numbers.py`**: a played game's numbers (`game_numbers`), reduced with
  `managym.statistics` over one re-execution and saved in
  `attempts.numbers_json`; `wilson_interval` for the My games record.
- **`card_art.py`**: Scryfall art and credits, fetched on first view into a
  local cache and served from `/api/card-art/`
- **`villain.py`**: opponent policies for the hero to face
- **`opponent.py`**: the server-scoped registered policy catalog
  (`GET /api/opponents`), chosen by exact Policy ID and verified before play

Academy never imports manabot or Torch, eagerly or lazily. Direct Python factories
and installed `managym.bots` providers supply playing policies, including manabot's
trained, greedy and heuristic players. Optional historical replay/advice uses the
separate `managym.academy.bots` group; when several such extensions are installed,
the first by entry-point name supplies those legacy capabilities. Built-in play
and finished replay work without a provider. Study validates the existing
`ed2` address against the pre-command frame, and validates the played Command
against its landmark separately. Retained `erd1` addresses still bind the
recorded offer and Command.

## Run it

To use Academy, install it and run one command; no checkout is needed:

```bash
uv venv --python 3.13
uv pip install --python .venv/bin/python --only-binary :all: 'managym[academy]'
uv run --no-project managym academy
```

[Running Academy](../../docs/academy.md#running-academy) is the guide: options,
the URL, stopping, playing, replays, trained bots and troubleshooting.
`managym academy play` and `managym academy replay` open a chosen game or a
selected replay directly, and `managym.academy.launch` does the same from
Python ([Launch from Python](../../docs/academy.md#launch-from-python)). The
commands below are for working on Academy from a checkout.

```bash
./scripts/play          # certified one-command path from fresh checkout
./scripts/play --demo learn  # checked prefix into Learn, then ordinary play
./scripts/play --demo precision  # Treasure-funded Deadly Precision
./scripts/play --demo precision-mana  # the four-mana additional-cost alternative
./scripts/verify-clean-machine   # the clean-machine proof
```

Card art needs no command. The server resolves a card name to its Scryfall id
through the content pool, fetches the image and its credit on first view into
`ACADEMY_CARD_ART_DIR` (default `~/.cache/managym/card-art`) and serves it
same-origin under `/api/card-art/`; the browser never contacts Scryfall. It
fetches only names a viewer's frame or deck list already holds and never
warms the cache from a decklist, so a closed list causes no request. Missing
art keeps the procedural card treatment, `ACADEMY_CARD_ART=off` disables
fetching, and artist credits are in the footer. See
[card art](../../docs/academy.md#card-art).

The Learn demo opens `/?demo=learn`: click a Lesson in the tray to take it, an
outlined hand card to discard it and draw, or **Decline Learn**. **Start game** repeats
the same prefix. The demo fixes UR Lessons in seat 0 against Random GW Allies;
its [evidence and remaining integration work](https://github.com/loopflowstudio/manabot/blob/cac6fa27d/docs/rules/learn-lesson.md)
are separate from complete-game replay certification.

Live play renders the board (`academy/ui/src/lib/board/`): one window with the
board, the hand and a rail holding the stack, the decision and every legal
offer by number. `boardView` places each offer exactly once in that list and
at least once on the card or player it acts on, as a prompt button (pass, pay,
declines and combat answers without another creature) or in a tray for subjects
the board does not draw. A card with one offer commits it on click; several
open at the card. Number keys, Space (primary) and F6 act from the keyboard. Rows size
their cards with `fitRow`, piling identical permanents, keeping a tapped card's
sideways width and wrapping before scrolling. See [playing on the board](../../docs/academy.md#playing-on-the-board).
Priority stops are set on the bottom step bar (off, every time or once)
and in the top bar's Menu; see
[priority stops](../../docs/academy.md#choosing-priority-stops) for defaults,
stack responses and F6. Recorded policy/search evidence is attached to the
decision it describes; the live board has no example Advice panel. Replay,
Watch and Study keep the earlier lane board and choice panel until they move to
the board. Public mana pools, spending restrictions and chooser-only inspected
cards come from native observations. Historical frames without those fields say
details were not recorded.

Academy uses native trivial-decision skipping, like every game runner: the
engine resolves sole legal actions for either seat, so no forced prompt reaches
a person or a policy. Disable **Use priority stops** to surface every priority
window that offers a real choice. Priority stop/F6 automation retains its
Commands and is marked **auto** in the move list; traces from before native
skipping keep their recorded forced **Auto** steps. Training defaults,
world identities, tensor inputs and frozen evidence are unchanged. Native
`decision_context_json` summarizes already-observed choices and the current
continuation; `decision_offer_subjects_json` binds the exact rule entity,
incarnation and board display ID. These read-only projections introduce no legal
actions or editable payment transaction. Other viewers receive only public
ownership/kind and committed facts. Retained frames omit these optional fields.

For Precision, click **Start game**, inspect **Deadly Precision**, then explicitly
choose its cast offer and an Inspector target. In the Treasure demo, reserve that
Inspector as sacrifice, activate Treasure, explicitly choose **Add {B}**, then
finish payment. The context distinguishes the unpaid reservation from the already
sacrificed Treasure. Reload before/after activation or color to recover the actual
pending decision; **Back** cannot refund costs. The mana demo instead offers the
four-mana additional cost while preserving the Inspector until resolution. Both
prefixes use normal Commands from seeded custom stress decks, without scenario
injection. Attempts are marked `automated_validation`, including any later human
continuation, so the authored prefix cannot be counted as human play. They are
runtime fixtures, not curated-deck admission or human usability
evidence. Start game repeats the selected prefix.

Replay labels recorded actions and automated passes separately. Live, Watch,
Replay and Study use the same decision renderer. Study executes the exact semantic
offer on a retained isolated root, then **Return to recorded decision** restores
its original address/frame/Command; **Return to live** is a separate destination.
Expired roots remain unavailable rather than being reconstructed from the screen.

After resolving a spell, open **History and settings** to read its recorded
consequences by turn. Earlier/later pages and **Latest** let you read older
events without being pulled away by incoming or recovered rows. Gaps appear
inline and turn context stays unknown until the next recorded turn change. Narration can be paused,
accelerated or skipped while the pending choice remains usable. Reloaded active
games label unavailable earlier history; older recordings without event data say
**Consequences not recorded**. See
[reading consequences and controlling narration](../../docs/academy.md#reading-consequences-and-controlling-narration)
for controls, exact-subject inspection and Replay/Study behavior.

Long action lists have an optional **Find an action** filter. See the
[retained choice positions and local walkthrough](../../docs/academy.md#choice-navigation)
for target, discard, Learn, combat and optional-cost coverage.

Development server without the launcher:

```bash
uv run uvicorn managym.academy.server:app --port 8000
```

The [local challenger workflow](../../docs/academy.md#playing-a-trained-challenger)
covers the built same-origin server, feedback, and read-only SQL.

Play and both Watch seats use the same registered policy picker and
`managym.policy` registry/session lifecycle
([provider and Python contract](../../docs/academy.md#play-and-watch-a-trained-checkpoint)).
Direct factories need no manifest. Optional `PolicySpec` values name installed
providers in `managym.bots`; multiple providers coexist. `serve(policies=...,
sources=...)` scopes sources to the server without rewriting environment variables.
Providers also discover their ambient configuration once per server start; explicit
sources add to that discovery.
The CLI accepts repeatable `--policy <spec.json>` and uses each file stem as an alias.

manabot owns checkpoint/run/Tournament discovery, bound-setup admission and
training metadata; Academy never interprets its run directories. Its Qwen
provider loads local fine-tuned safetensors with the tokenizer and advertises no
search support. Action-only policies retain no fabricated probabilities or value.
One-use decision bindings and complete legal-set validation precede native bound
Commands. The descriptor pins artifact, runner, config and provider identity in
existing recordings; old checkpoint records and finished replays need no loader.
The browser renders provider-authored groups and detail lines and precise unavailable reasons;
missing training counts remain "not recorded". Optional advice/import capabilities
are separate from policy loading.

An optional `SearchConfig` per bot composes an admitted policy with the
framework-free `managym.search` owner in Play, Watch and launch APIs
([Search](../../docs/academy.md#search)). Its initial budget is one
world at depth one, with a 30-second ceiling; explicit larger budgets are retained.
Native physical-deal sampling and bound Commands remain authoritative; the two-KL numerical update
lives beside it in `managym.search`. Full legal probabilities, compatible
signed values and isolated current-view forks are mandatory. Deadline failure
retains the live decision for an explicit, match/revision-bound pilot retry.
Budget checks bracket synchronous provider calls; an in-flight call returns
before its overdue result is rejected, and no overdue action is accepted.
The searched decision, a `BotDecision` subclass in `managym.search`, carries a
`SearchSummary` with prior probabilities, budget, timing, hexadecimal seed
and receipt digest; model outputs stay out of an opponent's live/viewer export.
Completed Play games expose retained actor-view evidence through Replay's
**Inspect bot search** link without loading a provider.

Replacement admission must keep the old GameSession authority intact until the
new environment resets successfully. Invalid native deck setup can raise
PyO3's `PanicException` (a direct `BaseException`, with no importable Python
class); contain that failure only at disposable-environment construction, while
letting process interrupts escape. Do not broaden gameplay error suppression.

Focused runtime checks: `uv run --extra dev pytest tests/academy/test_policy_runtime.py tests/academy/test_checkpoint_catalog.py tests/academy/test_watch.py -q`.
For shared picker or runtime changes, `npm --prefix academy/ui run test:e2e:policy`
builds the static SPA and checks Play retry/reload, both Watch seats, explicit
runner reselection and replay in both themes. Installed-wheel CI also checks the
provider-free picker; keep its expectations aligned with the shared menu.
The end-to-end browser
proofs (launch, offline reload, accessibility, release prompts) are under
[academy/ui/e2e/](../../academy/ui/e2e/) and documented in
[docs/development.md#clean-machine-play-proof](../../docs/development.md#clean-machine-play-proof) and
[experience proof gate](../../docs/academy.md#experience-proof-gate).

## Watch and recorded bot decisions

Open **Watch** (`/replay?watch=1`) to select both seats, decks and a deal seed;
`managym academy watch --player X --player Y` opens it with the seats selected
(`&seat1=…&seat2=…`) and leaves Start to the person. Trained checkpoints, in
one seat or both, and the random, greedy, heuristic and search anchors use
GameSession's normal Command authority. Live production is separate from
recorded playback. The shared replay viewer walks choices, recorded
consequences and result/latest-state positions for
human games, Watch and imported games. At a live Watch's latest position,
Continue requests one decision and Play can continue production. Browsing
earlier positions reads saved data without rerunning a policy. Completed games
open from Games in the same viewer. See the
[replay guide](../../docs/academy.md#inspect-a-replay) for navigation, speed,
perspective, reload and recovery. Watch credentials stay in the tab's
`sessionStorage`; saved position URLs carry no credentials.

Opening card details or reading earlier history pauses replay. Close the details
when ready, then press **Play** to resume. Returning to **Latest** does not start
playback. A live decision already being produced may finish saving while you read;
your selected board stays put. A pending Search timeout is
retained across reload and shown with its message. At the live edge, **Retry
search** explicitly retries that decision; browsing saved positions never retries
it. The exact selected position remains selected after reload.

`watch.py` admits controllers, managym game records (`import_record`, re-executed
through `managym.replay.walk` with no extension) and retained Command traces
through **Open a game file**; choose a game when a file contains several.
Import verifies the original receipt, Commands and state digests without invoking a player.
The imported Academy replay has its own identity; the source digest is retained
without rewriting the arena evidence. A game's stated source kind is kept
beside it, and Replay is headed **Training game** or **Evaluation game**
accordingly; a training game is one exported by
`manabot.sim.recording`
([guide](../../docs/academy.md#watch-a-game-from-your-training-loop)), in the
same Command format and through the same verification. A game that states its
world is refused before replay when that world is not the running one. Missing historical policy outputs are
marked unavailable. Local-update receipts expose only policy/search vectors
and an acting-view belief summary, never private sampled hands or branch audits.

`TraceEvent.bot_decision` records the actual policy/value/search outputs at
choice time. Traces from the removed compound-decision policies may retain
conditional root-sequence factors and values; these remain labeled separately
from native microchoice marginals. Anchors without critics have no value estimate. Watch's acting-view
inspection endpoint never returns the full mixed canonical replay. The separate
private analysis observation supplies **Show both hands**, available explicitly
for authorized bot spectators, including during active watches. Every read,
including completed inspection, follows the recording's participant/shared
access rules; finishing does not make a private recording public. Active Watch
inspection also requires its watch credential. Choices default to the acting
bot's view, independently of the stable seat-0 board orientation. Consequence
and result boards use the authorized recorded seat-0 view where actor-private
post-action snapshots are absent, with the perspective explicitly labeled.
Unavailable views stay unavailable. Ordinary trace reads always strip the
analysis snapshot, and bot inputs, decision evidence and acting-view history
retain their acting-view scope.

**Try another line** uses retained Study roots from either acting seat, executes
one exact semantic Command on an isolated branch, and returns to the recorded
decision. It does not advance the watched match. Roots and active controllers
have the ordinary game-session lifetime; saved completed inspection remains
available after the session expires, while a fresh Study fork requires retained
roots. Existing structured Study receipts and frozen `erd1` evidence keep their
original command bindings.

## Reading a played game

A played game retains the boards, seed, both setups and the action at each
step. `ReplayedViews` re-executes it with `managym.replay.RecordWalker` as far
as a read asks, so opening a long game costs nothing up front and each step
forward adds one or two projections. [Guide](../../docs/academy.md#inspect-a-replay)

A played game has no state digests, so the walk checks itself against the
stored track: each step's acting-seat board and action description must equal
what was stored, and the final board and result must too. At the first
difference the walk stops; replayed views are used up to it and the stored
track after it, and the read's `views_stopped` says how far the view was
replayed. A game from another world or card set is not walked. The stored
boards stay the authority for what the person saw; replayed views are never
written back. Narration of the opponent's moves reads the stored track.

Access follows the recording. A reading from the person's seat contains
nothing that seat did not see. The opponent's view, for **Show both hands**,
exists only for an authoritatively complete game (`end_reason == "game_over"`).
Each `TraceEvent` records `committed_at`, its commit time in UTC; nothing
shows it yet.

### Analysis lifecycle

`POST /api/traces/{id}/analyses {"policy": <Policy ID>}` (participant only)
loads a fresh `PolicyRunner` in the person's seat and walks the game on a
daemon thread; `GET` lists the game's analyses with their decisions so far,
reading only their rows, so Review follows a running analysis with it.
`GET /api/traces/{id}/analyses/policies`, read once when Review opens,
re-executes the game to list the eligible policies, the default (the
opponent's own policy, else the recommended one), why none may run and the events an analysis of the
person's seat reads. Eligible means registered, admitted for the game's decks, with an
artifact and no Search composition: Search plays out from the walk's real
hidden cards. Admission takes seconds with many checkpoints and reads only the
registered policies and each seat's deck, sideboard and world, so
`PolicyAdmissions` admits each (registered policies, setups) once per process,
for every game with those decks, Review and Play alike; loading a runner still
admits it again. A game still being played is refused (`analysis_game_active`);
one analysis runs per game per process (`analysis_running`).

At each of the seat's deliberate decisions (automation is skipped) the runner
is called on the walk's observation and Env, its `BotDecision` kept as a
`DecisionAnnotation` plus the value semantic, its sampled action discarded.
The walk checks each step's actor and action description, not whole boards.
Each offer's estimate after it is read through the session's history-free
`evaluate` at the next decision on a clone: the seat's own value when it
chooses next, else the opponent's negated, in a world `determinize`d from the
seat's perspective. Closed decklists cannot be re-dealt, so those estimates
are absent. A manabot value belongs to the chooser, so no value is read from
a seat that is not choosing.

A runner error or divergence stops the analysis with its reason; saved
decisions stay. A `running` row whose process is gone reads as stopped.

`POST /api/traces/{id}/analyses/{analysis}/line {"event": <analysed event>}`
(participant only; `agent_line`) re-executes the game to that decision with a
fresh runner of the analysing policy in the seat and the game's own opponent
(`build_villain_policy`) in the other, plays the analysis's first choice on a
copy and continues with the policy's first choice and the opponent's own
choice to the end of the next turn (`LINE_TURNS`), the end of the game or
`MAX_LINE_STEPS`. A recurrent policy is asked at each of its seat's earlier
decisions so it has its history. Each step returns the seat's own
`decision_surface_view`; the opponent's actions are named only for a complete
game. The line is computed per request in the real game's world and never
saved. Key moments are derived in the client from the analysis
(`keyMoments`).

### Playing on from a decision

`new_game {"played_from": {"trace_id", "event"}, "player_token"}` starts a
Play game from one of the person's deliberate decisions (`actor` hero, not
`auto`) of their own game that is no longer being played; another reader's
game, an opponent's step or an active game is refused (`played_from_*`). The
new game takes the source's setup and seed (`setup_draft` plus the seed), and
`_play_recorded_prefix` replays the first `event` steps through the ordinary
Command path: the opponent's recorded actions through the offer-policy hook (a
recurrent opponent is still asked, so it keeps its history) and the person's
through `hero_command`, with every window surfaced so automated passes replay
as recorded, now as Commands. Each step must name the recorded action; the
first that does not stops the start with `played_from_diverged`. The prefix is
persisted once, like a play demo's, and the game is then an ordinary recorded
game against a fresh opponent policy. `Trace.played_from` names the source and
step (written only when set); `GameNumbers.played_from` copies the step so
`/api/traces/record` leaves the game out, and the recording index returns it
with the turn it starts on.

### Analysis while you play

The pilot's `show_analysis {"shown": true, "policy": <Policy ID or null>}`
request (`SUBMIT_LIVE_COMMAND`) answers `analysis_state` with the policy shown,
the policies that may read the game (`analysing_policies`, from
`PolicyAdmissions`) and why none may. The connection admits them off the game
lock before dispatching the request, so the first Show analysis of a game's
decks never holds play while checkpoints are admitted. Showing starts a `LiveAnalysis` on its own daemon
thread: it loads a fresh runner, walks the game so far with `seat_decisions`
(asking the runner at each earlier decision, as an analysis after the game
does, so decision memory matches), then applies each step the game commits and
reads each decision the game publishes to the person (`_publish_current_prompt`)
with `read_decision`. Steps and decisions reach the worker in the game's order
on one queue; the worker never reads the game's Env, so nothing it does holds
the game lock or a frame.

Each reading goes to the pilot's sockets as its own
`{"type": "analysis", "policy", "event", "revision", "annotation", "semantic",
"after"}` message (or `{"stopped": reason}`), scheduled on the server loop
without waiting. The client draws it only while its frame's revision matches.
Once a send completes the event is `shown`, and committing that decision stores
`analysis_shown: true` on its `TraceEvent`. The field is written only when true,
so a game played without analysis saves the same row as before. Frames carry
`analysis: {"policy"}` only while it is shown, so a reload reads the switch.
`show_analysis {"shown": false}` closes the worker; a new game or a closed
session closes it too. `GameNumbers.analysis_shown` counts the marked
decisions, the Review position index exposes `analysis_shown` per choice, and
`/api/traces/record` counts games with any (`with_analysis`).

### Numbers and record

When a played game is no longer being played, `game_numbers` walks it once
with `Behaviour` and `Totals` (`managym.statistics`), checking each step's
actor and action description as an analysis does, and saves the result in
`attempts.numbers_json`. A failed count never affects the saved game.
`GET /api/traces/{id}/numbers` returns it, counting a game saved before
numbers were on its first read; a game that does not re-execute here, a Watch
game or a game still being played reads `null`. `GET /api/traces` rows carry
`numbers` from the column and never re-execute.

`GET /api/traces/record` groups the caller's completed games by opponent
identity and both decks: games, wins, draws, losses, games with a decision made
while analysis was shown (`with_analysis`), and a 95% Wilson interval on wins
over games.


## Elemental roster

The shared deck browser assigns either seat from all ten exact Elemental setups
in the additive `elemental-world-v2` pack, including mirrors. Browse the main
lists and Lesson pools, click card artwork for a printed preview, then
use the assignment buttons to choose each seat. During play, **Choose decks**
prepares the next matchup; only **Start game** replaces the current attempt.
The [local MTG-183 launcher](../../docs/academy.md#playing-a-trained-challenger)
pins the latest published checkpoint for CPU play.
The decks' 40-card main lists, Lesson pools, local treatments and retained card-identity
provenance are in the schema-2 roster manifest at
`managym/academy/packs/elemental-world/v2/manifest.json`. Native compiled setups
validate every row. A roster manifest chooses a default pair from those rows;
it does not define a second setup authority. Earlier schema-1 pair packs remain
separate. Mixing an Elemental name with a setup outside its roster is rejected.

A recorded Elemental game carries explicit compiled pack and asset identity
when rebuilding roots. The ordinary checkpoint catalog exposes its ten exact
setups, and loaded policies retain that roster for cross-pair and mirror
admission. Temporary Avatar grants reach both player projections, the shared
protocol and the card's visible/accessibility label. These changes add no
human-only choice or extra information. See the
[Elemental world contract](https://github.com/loopflowstudio/manabot/blob/cac6fa27d/docs/rules/elemental/world.md) for evidence limits.

Optional resolving costs share typed payment progress and effect continuations.
See [optional payments](../../docs/engine.md#resolution-contracts) for authoring,
commitment and frozen-content boundaries.

## Isolated browser validation

Use unused backend and frontend ports for this checkout. Playwright can reuse an
existing server; another worktree's server is not evidence for this source.

```bash
ACADEMY_API_PORT=8123 ACADEMY_FRONTEND_PORT=5293 \
  npm --prefix academy/ui run test:e2e -- testing-house.spec.ts
```

For a built SPA, set an absolute `ACADEMY_FRONTEND_BUILD` for build and ASGI,
a separate `ACADEMY_TRACES_DIR`, and
`ACADEMY_PLAY_RECORD_ORIGIN=automated_validation`. Keep existing human-play
stores and sibling servers untouched. Shared-history proofs use separate
browser identities and cover private unfinished replays and participant feedback.
For release-config proofs, set an absolute `ACADEMY_RELEASE_PROOF_DIR` per
subprocess; both browser artifacts and its trace store live below that directory.
Finish frontend generation and builds before starting the proof servers.

Changes to persistence or Study addresses require presentation, Study runtime,
and protocol checks as well as attempt-store checks. SQLite is the trace
authority; historical JSON is read-only. `ed2` names a pre-command frame;
Command and offer validation belongs to its landmark. Preserve frozen `erd1`
evidence and test recorder failure through the WebSocket boundary.

### Ordering simultaneous triggered abilities

When several of your abilities are waiting, choose one ability to put on the
stack next. The shared native label describes that specific ability; inspect its
source for the full card text. Choosing commits its place in the order. If it
needs targets, finish those choices before continuing. Later placements sit above
earlier ones and resolve first if the stack proceeds without responses. The last
remaining position is forced. Reload restores the actual pending order or target
decision, or the current stack if placement has finished. A lost response does not
undo a submitted choice.

For editable file input and development construction rules, see
[decklist import](../../docs/engine.md#decks).
