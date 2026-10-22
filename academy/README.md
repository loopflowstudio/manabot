# academy

The Academy experience server: authoritative play, presentation, and
study. The Svelte client in [frontend/](../frontend/) renders what this
package decides; the client never invents rules meaning.

## Authority model

One versioned experience protocol drives direct play, replay, and decision
inspection as projections of the same authority. The server owns legality,
hidden information, and state; the client receives `ExperienceFrame`s and
`PresentationEvent`s and returns `Command`s bound to the exact frame that
offered them. Recovery envelopes make sessions reloadable — including fully
offline. Schemas and three-language conformance fixtures live in
[protocol/](../protocol/README.md).

## Modules

- **`server.py`**: FastAPI WebSocket server (`/ws/play`) — sessions,
  authoritative game loop, viewer-safe observation serialization
- **`presentation.py`**: semantic presentation events (combat, turn
  transitions, spell sequences) derived from engine events
- **`curated_pack.py`**: the frozen curated matchup asset pack
- **`study_protocol.py`**: viewer-safe study artifacts and decision evidence
- **`study_branch.py`**: source-bound historical forks, native structured
  execution, typed failure, and consuming exact-return receipts
- **`trace.py` / `attempts.py`**: viewer-safe trace projection and SQLite human
  attempts/feedback (`ACADEMY_TRACES_DIR/play.sqlite`, default `academy/traces/`).
  New records have one SQLite authority; legacy JSON remains readable.
  `attempt_players` identifies both seats with stable public IDs, types, recorded
  names, decks and bot versions. Table credentials and browser player credentials
  remain private authorization values. Shared history applies permissions in SQL
  before filtering/pagination; only completed shared replays are readable by
  nonparticipants. `ACADEMY_PLAY_RECORD_ORIGIN` distinguishes declared automated
  validation from human play; absent provenance is unknown.
- **`villain.py`**: opponent policies for the hero to face
- **`opponent.py`**: trained opponents — the configured `candidate.json`
  bundle and the checkpoint catalog (`GET /api/opponents`), both chosen by
  exact SHA-256 and verified before play

Keep training and belief dependencies lazy at their execution boundaries.
Ordinary play must import, advance and record without NumPy or Torch; type-only
belief imports belong under `TYPE_CHECKING`. Study validates the existing
`ed2` address against the pre-command frame, and validates the played Command
against its landmark separately. Retained `erd1` addresses still bind the
recorded offer and Command.

## Run it

```bash
./scripts/play          # certified one-command path from fresh checkout
./scripts/play --demo learn  # checked prefix into Learn, then ordinary play
./scripts/verify-clean-machine   # the clean-machine proof
```

The Learn demo opens `/?demo=learn`: choose **Take a Lesson**, **Discard and
draw**, or **Decline Learn**. The first two modes show card previews and a
Back control; only selecting a card commits the choice. **New Game** repeats
the same prefix. The demo fixes UR Lessons in seat 0 against Random GW Allies;
its [evidence and remaining integration work](../docs/rules/learn-lesson.md)
are separate from complete-game replay certification.

Long action lists have an optional **Find an action** filter. See the
[retained choice positions and local walkthrough](../docs/choice-navigation.md)
for target, discard, Learn, combat and optional-cost coverage.

Development server without the launcher:

```bash
uv run uvicorn academy.server:app --port 8000
```

The [local trained-challenger workflow](../docs/local-trained-challenger.md)
covers training, the built same-origin server, feedback, and read-only SQL.
Configure `ACADEMY_PLAY_CANDIDATE` on the server, never in player input.

Any other trained checkpoint is selectable from the Opponent menu. The table
lists every policy checkpoint under `MANABOT_RUNS_DIR` (default `./.runs`)
and in `ACADEMY_PLAY_CHECKPOINTS` — `.pt` files, run directories and
Tournament files, separated by `:` — with world, model shape, training step
and producer. manabot's `manabot.model.catalog` reads them with Torch's
weights-only loader and checks each world binding against the running engine;
an incompatible checkpoint stays listed, disabled, with the mismatch (for
example `trained in w5, engine is w6`), and the server refuses it with the same
sentence. The browser sends only a digest; the server re-hashes the bytes,
checks the selected decks against the checkpoint's binding (either seat), and
records the exact checkpoint identity as the trace's `opponent_checkpoint`.
Listing needs the training dependencies; a play-only install reports that the
catalog is unavailable and still serves the configured candidate.

Replacement admission must keep the old GameSession authority intact until the
new environment resets successfully. Invalid native deck setup can raise
PyO3's `PanicException` (a direct `BaseException`, with no importable Python
class); contain that failure only at disposable-environment construction, while
letting process interrupts escape. Do not broaden gameplay error suppression.

Tests: `uv run --extra dev pytest tests/academy`. The end-to-end browser
proofs (launch, offline reload, accessibility, release prompts) are under
[frontend/e2e/](../frontend/e2e/) and documented in
[docs/clean-machine-play.md](../docs/clean-machine-play.md) and
[docs/experience-proof.md](../docs/experience-proof.md).

## Watch and recorded bot decisions

Open **Watch** (`/replay?watch=1`) to select both seats, decks and a deal seed.
The configured checkpoint and random, greedy, heuristic and search anchors use
GameSession's normal Command authority. Step advances one native decision;
Play/Pause and speed control that same operation. Previous decisions read saved
outputs without rerunning the policy. Completed watched games open from Games
in this same table. Reloading retains the watch credential in the browser.

`watch.py` admits controllers and verified arena/Tournament game JSON or
`.jsonl.gz` traces; choose a game when a trace contains several. Import verifies
the original receipt, Commands and state digests without invoking a player.
The imported Academy replay has its own identity; the source digest is retained
without rewriting the arena evidence. Missing historical policy outputs are
marked unavailable. Local-update receipts expose only policy/search vectors
and an acting-view belief summary, never private sampled hands or branch audits.

`TraceEvent.bot_decision` records the actual policy/value/search outputs at
choice time. Compound checkpoints record their conditional root-sequence
factors and values; these are labeled separately from native microchoice
marginals. Anchors without critics have no value estimate. Watch's acting-view
inspection endpoint never returns the full mixed canonical replay. The separate
private analysis observation supplies **Show both hands**, enabled by default
for bot spectators, including during active watches. Active inspection requires
the watch credential; completed inspection remains shared. Switching it off
shows only the acting bot's view. Ordinary trace reads always strip the analysis
snapshot, and bot inputs and decision evidence retain their acting-view scope.

**Try another line** uses retained Study roots from either acting seat, executes
one exact semantic Command on an isolated branch, and returns to the recorded
decision. It does not advance the watched match. Roots and active controllers
have the ordinary table-session lifetime; saved completed inspection remains
available after the session expires, while a fresh Study fork requires retained
roots. Existing structured Study receipts and frozen `erd1` evidence keep their
original command bindings.


## Elemental roster

Both deck selectors include all ten exact Elemental setups from the additive
`elemental-world-v2` pack, including mirrors and either seat order. Their
40-card main lists, Lesson pools, local treatments and retained card-identity
provenance are in the schema-2 roster manifest at
`frontend/src/lib/packs/elemental-world/v2/manifest.json`. Native compiled setups
validate every row. A roster manifest chooses a default pair from those rows;
it does not define a second setup authority. Earlier schema-1 pair packs remain
separate. Mixing an Elemental name with a setup outside its roster is rejected.

A recorded Elemental game carries explicit compiled pack and asset identity
when rebuilding roots. The ordinary checkpoint catalog exposes its ten exact
setups, and loaded policies retain that roster for cross-pair and mirror
admission. Temporary Avatar grants reach both player projections, the shared
protocol and the card's visible/accessibility label. These changes add no
human-only choice or extra information. See the
[Elemental world contract](../docs/rules/elemental/world.md) for evidence limits.

Optional resolving costs share typed payment progress and effect continuations.
See [optional payments](../docs/rules/optional-payments.md) for authoring,
commitment and frozen-content boundaries.
