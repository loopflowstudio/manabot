# Install the rules engine

For an unpublished candidate, install the local wheels using
[Build and verify](development.md#build-and-verify). The PyPI commands below
apply after the paired release is published.

Install `managym` in a fresh directory outside the repository:

```bash
mkdir managym-example
cd managym-example
uv venv --python 3.13
uv pip install --python .venv/bin/python --only-binary :all: managym
uv run --no-project --python .venv/bin/python python -c 'import managym; print(managym.world_identity())'
```

The last command prints the installed world's version and card digest. The
`managym` distribution contains the compiled engine and its Python API. It
installs without Torch, Gymnasium, Etude or the training CLI. NumPy supports
native observation buffers; Pydantic supports the structured decision API.
No Rust compiler is needed when installing a wheel.

To play in the browser instead of from Python, install the `academy` extra and
run `uv run --no-project managym academy` ([running Academy](academy.md#running-academy)); it adds
FastAPI and Uvicorn and still needs no manabot or Torch.

Supported wheels target ordinary CPython 3.12 and 3.13 on Linux x86-64 with
glibc 2.28+ and Apple Silicon macOS 14+. To use 3.12, change the environment
creation command to `uv venv --python 3.12`. Free-threaded Python, Windows,
Intel Mac, Linux ARM and musl are outside this wheel matrix. If installation
reports no matching distribution, check Python, architecture and platform.
`--only-binary :all:` prevents an unexpected source build; building from the
source archive requires Rust and a C linker.

Save this as `play.py` in the example directory:

```python
import managym

game = managym.Env(seed=42)
observation, info = game.reset([
    managym.authored_deck_setup("tla-baseline-v1", key)
    for key in ("tla_red", "tla_white")
])
observation, reward, terminated, truncated, info = game.step(0)
print("chooser:", game.current_agent_index())
print("game over:", terminated or truncated)
```

Run `uv run --no-project --python .venv/bin/python play.py`. It starts a game
from two bundled decks and executes the first legal action, then prints the
next chooser and whether the game ended. Both seats' choices belong to the
caller. To play against a built-in without installing `manabot`, construct
`managym.Env(seed=42, player_policies=("agent", "heuristic"))`; `agent` is an
external human, model or tool, while `random` and `heuristic` choose natively.
The same tuple works with `managym.VectorEnv`. Either seat may be external.
For two built-ins, reset constructs the game and each `game.step()` advances
one decision; a vector uses `games.step([-1, ...])`. See
[per-seat players](engine.md#per-seat-players) for rewards,
terminal reset and Command stepping.
Forced single-action decisions are skipped by default. See the
[engine guide](engine.md#choosing-a-python-api) for complete games, batches,
reset behavior, Commands and hidden-information boundaries, and
[encoded observations](engine.md#encoded-observations) for reading the native
arrays by column name and joining card definition IDs.

## Collect in-game statistics

Give a capture a private statistics file to count every game, independently of
record sampling. It enables the native counters before reset and drains them
at game boundaries and close:

```python
from pathlib import Path
import managym
from managym.recording import ScalarCapture
from managym.statistics import CardPerformance, read_statistics

game = managym.Env(seed=42)
capture = ScalarCapture(seed=42, statistics_path=Path("statistics/worker.jsonl"))
players = [managym.PlayerConfig(str(seat), {"Mountain": 12}) for seat in (0, 1)]
capture.reset(game, players)
while not game.is_game_over():
    capture.step(game, 0)
capture.close()
performance = CardPerformance()
print(performance.from_statistics(read_statistics(Path("statistics"))))
print(performance.excluded_games)
```

The same `statistics_path` option works on `CapturedVector`. Use one file per
producer. Rows carry the card digest and the capture's `context` labels at the
game's end, so a table can be split by run, update or cell; they use mode 0600, flush at each game boundary
and fsync at close. The reader accepts a file or directory and ignores a torn
last line. A game that keeps playing after a raised failure still writes its
terminal row. Partial summaries from replaced or closed games are counted in
`excluded_games` and excluded from the table. These private every-game rows
never join audited evaluation populations automatically.

For low-level counters without a capture, enable statistics before the first
reset and own the drain yourself:

```python
import managym

game = managym.Env(seed=42)
game.enable_statistics()
players = [
    managym.authored_deck_setup("tla-baseline-v1", key)
    for key in ("tla_red", "tla_white")
]
observation, info = game.reset(players)
observation, reward, terminated, truncated, info = game.step(0)
print(game.statistics())
```

The typed summary describes this game's native seats, seed, generation,
completion and winner, alongside per-card counts. It counts opening and deck
copies, actual library draws (including redraws), nontrivial opportunities and
selections, and completed resolutions. Opportunities and selections count each
definition once per chooser and committed nontrivial decision, even when several
copies occur among its options. Tutors and returns to hand are not
library draws. A focused card identifies the choice's subject; it does not
establish that the card caused the eventual outcome. Rejected or stale Commands
contribute no committed choice. Missing historical exposure is unavailable,
not an observed zero.

Statistics default off. `enable_statistics()` must precede reset; enabling it
partway through a game fails rather than presenting incomplete counts as a full
game. `statistics()` reads the current summary without consuming it; disabled
or uninitialized statistics report unavailable. `take_statistics()` drains
summaries of completed or replaced generations. Replacing an unfinished game
retains a partial summary, never a fabricated result. The same enable/read/drain
methods apply to `VectorEnv`, with each summary identifying its stream and game
generation. Drain regularly: queued summaries accumulate until consumed.

Native counters require neither detailed recording nor replay and do not
enable card/exposure history capture. Direct Env calls perform no file writes;
a capture with `statistics_path` owns the drain and persists summaries. Counter storage grows
with encountered card definitions and physical copies, rather than decisions;
generated copies can still grow during a game. The Env's existing action trace
remains independent, so this is not a constant-memory promise for the whole
Env. Statistics and private recording can be enabled separately or together;
neither changes actions, seeds or legal choices.

These summaries include private deck facts; do not expose them as Academy viewer
data. Persist summaries yourself when needed. An interrupted process can lose its
in-memory counts; label the last saved summary partial and report the gap.
Counts alone cannot reconstruct a game; [record the games](#record-games) to
ask questions the counters do not answer. Live summaries also do not replace
replay verification for audited evaluation: manabot retains its replay-passed
eligibility rules and participant mapping when exporting Etude statistics.

## Record games

Wrap every mutation from the first reset onward in a capture, give it a file,
and each finished game becomes one row:

```python
from pathlib import Path

import managym
from managym.recording import ScalarCapture

game = managym.Env(seed=7)
capture = ScalarCapture(seed=7, games_path=Path("games.jsonl"))
players = [
    managym.authored_deck_setup("tla-baseline-v1", key)
    for key in ("tla_red", "tla_white")
]
capture.reset(game, players)
while not game.is_game_over():
    capture.step(game, 0, annotation={"value": 0.31})
capture.close()
```

```bash
uv run --no-project managym academy replay games.jsonl
```

Academy opens in **Replay** at the game's first decision. Each step shows the
board, the legal actions and the one chosen, with "Value 0.3100" beside it and
"not retained" for the probabilities this loop did not supply
([Open a replay directly](academy.md#open-a-replay-directly)).

A record holds what is needed to play the game again: the seed, both players'
setups, the world and build that played it, each committed action with the
state digest read before it, and how the game ended. It stores no boards.
Anything that shows or counts a game re-executes it with managym and checks
every digest, so a record is small and cannot disagree with the engine. The
same record serves training, evaluation and your own loop.

Keep the capture's seed and trivial-decision setting equal to the Env's, and
use the capture for every later mutation; a direct Env call leaves the record
behind. Call `capture.reset` again to start the next game in the same file.
Recording changes no action, seed, legal choice or outcome, and does not enable
statistics; add `statistics_path` to collect those independently.

| Option | Meaning |
|---|---|
| `games_path` | File that receives one row per finished game. Without it nothing is written at game end. Give each producer its own file. |
| `record_fraction` | Share of games to record, `1.0` by default. A game is selected at reset from its seed and stream, never from engine or policy randomness, so the same run selects the same games. An unselected game costs no extra native read. |
| `statistics_path` | Private native summaries for every game, independent of record selection. |
| `journal_path` | [Durable journal](#keep-a-durable-journal) of the game in progress. When supplied before reset, it records digests and annotations at the default fraction, without requiring a games file. A journal enabled mid-game keeps diagnostic inputs for that root and starts full recording at the next root. Set `record_fraction=0` for diagnostic cursors without digest reads. |
| `context` | Your own labels (run, update, attempt), kept with every row and never shown to an Academy viewer. |

`annotation` is optional per step and says what the chooser knew:
`probabilities` (one per legal action, in offer order), `value`, `update` and
`extra` for anything else JSON can hold. Academy shows the first three and
marks an absent one "not retained"; it never recomputes them.

For native batches, `managym.recording.CapturedVector` owns its `VectorEnv`
and takes the same options. Construct it with `num_envs` and `seed`, use
`reset_all(players)` and `step(actions)` or the buffered methods, and pass
`annotations=` with one entry (or `None`) per stream. Use `stream_offset` for
global stream identities across producers. A game recorded from a batch opens
exactly like a scalar one.

`capture.close()` writes each selected game still in progress as
**interrupted**, with the reason, and syncs the file. Rows written at game end
are flushed but not synced, so a process that is killed can lose its most
recent rows and its games in progress; the journal below is for that case.

### Read and re-execute a record

```python
from pathlib import Path

from managym.recording import read_records
from managym.replay import walk

record = next(read_records(Path("games.jsonl")))
replay = walk(record)
for decision in replay:
    offered = decision.observation.action_space.actions
    print(decision.index, offered[decision.chosen].action_type, decision.annotation)
print(replay.stop)   # complete, winner 0
```

`read_records` takes a file or a directory of them and yields one `GameRecord`
per game, including each stream of a vector snapshot or journal.
`read_record(path)` reads exactly one row or projected journal and keeps all its
streams together for failure reproduction. `walk` plays the game in a scalar `Env` and
yields each decision only after its digest matched. It ends with a typed
`stop`:

| Stop | Meaning |
|---|---|
| `complete` | Every action replayed and the final digest matched; the winner is the recorded one. |
| `interrupted` | The recording ended before the game did. The decisions before it are verified; there is no winner. |
| `diverged` | At this index the digest differed, the recorded action was not offered, or the engine raised. Nothing after it is produced. |

Statistics, the Arena-style export and Academy read games through `walk`, so
they agree on where a game stops. The bounded command-line replay also uses
this walker for game history:

```bash
uv run --no-project --python .venv/bin/python python -m managym.replay games.jsonl --seconds 30
```

Add `--statistics` to print the replayed game's `terminal`, `winner` and
per-card counts, reduced from the walker's verified card facts (the same
counts as the [native counters](#collect-in-game-statistics)); for a game that
stopped early these describe the replayed prefix, never the original unknown
outcome. Reproducing a failed or pending call prints the full native summary
instead, with seed, stream and generation.

A record opens only in the world that played it. A different world or card
content is refused, and the message names the recorded and the running
managym version, world and engine digest, so you know which build to install:

```text
record world or cards differ: recorded managym 0.3.2 world w17 cards 5be01c9a2f31… engine 9c0e…; running managym 0.3.4 world w18 cards e07a44d1c8b2… engine 42da…. Open it with managym 0.3.2.
```

The same world on a different engine build (another platform, or a patch that
left the world alone) opens, says the build differs, and is checked decision by
decision like any other; if the builds do not play the game alike it stops as
`diverged` with the same names. To keep a game readable across worlds, write
its [Arena-style log](#export-an-arena-style-log).

A record holds the seed and both decks, so it holds both hands. Files use mode
0600 and directories 0700. Share one as you would share both players' hidden
cards; Academy still shows each viewer only what that seat could see.

Rows written before this format (`native-failure-v1` receipts and journals)
read through the same functions. Private snapshot replays (`managym-snapshot-v1`
exported by `GameSnapshot.to_replay_json()`) also open directly through
`read_record`, `read_records`, `records_from_source`, `walk`, and
`managym academy replay`, executing any recorded boundary interventions natively.
`FailureEvidence` and `read_failure` are the
former names of `GameRecord` and `read_record`; they warn and are removed in
0.4.0.

### Keep a durable journal

A journal is the same record written earlier: before and after every native
call instead of once at game end. It is for a call that may never return.
`ScalarCapture.command` creates a diagnostic journal under its
`failure_directory` when none was supplied, so a pending Command is durable
before execution. This automatic journal does not enable game sampling or
digest reads. A failed call freezes that automatic journal as failure evidence;
later resets cannot overwrite it. An explicitly configured journal stays a live
cursor and also retains a separate immutable failure snapshot.

```python
capture = ScalarCapture(
    seed=42,
    journal_path=Path("private-games/game.jsonl"),
    context={"attempt": "local-example"},
)
```

The journal retains actual observed setups, known action prefixes and
pending/returned native-call boundaries. Cached intent is durable before native
decision-frame preparation. A pending call means no durable return
acknowledgment: it may not have entered native code, or may have returned before
the acknowledgment could be saved. Batch return does not prove each row
succeeded. Sampled reset inputs remain unknown until observed; previous decks
are never substituted as the new setup.

The command chooses failure reproduction when the record has an error, a
pending boundary or a search call. It reconstructs the original scalar, vector
or search execution and tries the retained failing call. This requires the same
native bytes and recording source; `--allow-changed-runtime` explicitly permits
a different build of the same world. Viewing through `walk` or Academy instead
uses the world/card admission above and never attempts a pending call.

Copy the file to a private location before inspecting a live writer, then read
it with `read_record` and replay it with the command above. Replay reconstructs
the known prefix and, for this diagnostic command only, can attempt a pending
indexed action with its retained fresh binding; Academy and `walk` never
attempt a pending call and show the game as interrupted there. Unknown setup or
binding is reported as unavailable, and the original interrupted outcome stays
unknown even if replay succeeds. Older manabot receipts keep their original
source-digest meaning; when legacy source verification requires manabot, use
its [replay command](training.md#inspect-and-replay-a-journal) in the original
environment. Replay does not restore a process or optimizer.

Each boundary requires serialization and fsync, which can reduce throughput;
no training measurement of that cost exists yet. Automatic atomic compaction
replaces superseded journal history when appended bytes reach the larger of
the last checkpoint size and 64 KiB, allowing one record's overshoot and a
temporary checkpoint. Storage still grows with active prefixes. A finished
game supersedes its journal; set `games_path` as well to keep finished games.

The reader warns on a torn final append and uses complete records; complete
corruption, sequence gaps or an incomplete first checkpoint fail inspection.
If pending intent cannot be persisted, the native operation is not entered.
A persistence error after execution does not mean the operation did not run.
Retain existing evidence and repair storage before starting a new attempt;
there is no automatic retry or game skipping. Local durability cannot protect
against host deletion: collect private files before removing the machine.

## Save and restore game positions

An immutable `GameSnapshot` can restore independent games on several threads,
even after the source `Env` is gone. Here two opening choices share the same
unseen library order.

```python
import json
from pathlib import Path

import managym
from managym.journal import DurableJournal
from managym.recording import read_record
from managym.replay import walk

game = managym.Env(seed=42, decision_memory=True, decision_memory_include_own=True)
players = [
    managym.PlayerConfig(f"p{seat}", {"Mountain": 8, "Gray Ogre": 4})
    for seat in (0, 1)
]
game.reset(managym.GameSetup(players, decklists="closed"))

snapshot = game.snapshot()
meta = snapshot.metadata()
print(meta.turn, meta.phase, meta.action_count, meta.terminal)
# 1 PhaseEnum.PrecombatMain 6 False

del game
branch_a = snapshot.restore(
    managym.RestoreOptions.shuffle_libraries(seed=101, forced_first_action=0)
)
branch_b = snapshot.restore(
    managym.RestoreOptions.shuffle_libraries(seed=101, forced_first_action=1)
)

# Atomically save both decks and the branch seed in an owner-only file.
replay_path = Path("branch-a.json")
DurableJournal(replay_path).replace(json.loads(branch_a.snapshot().to_replay_json()))

reopened = managym.GameSnapshot.from_replay_json(replay_path.read_text())
continuation = reopened.restore()
assert continuation.state_digest() == branch_a.state_digest()

# The same snapshot replay feeds managym.replay and Academy Watch directly.
replay = walk(read_record(replay_path))
for _ in replay:
    pass
print(replay.stop.status, replay.stop.index)
# interrupted 1
```

```bash
uv run --no-project --python .venv/bin/python python -m managym.replay branch-a.json --seconds 30
uv run --no-project managym academy replay branch-a.json
```

`game.snapshot()` accepts a decision boundary or a finished game and raises
before the first reset. Each restored game starts a fresh generation with
empty projection caches. The source keeps its profiler, statistics and
recording sinks. Use `snapshot.restore()` for exact continuation, or pass
`RestoreOptions` to change the branch at its root.

| Constructor | Effect at the saved boundary |
|---|---|
| `RestoreOptions()` / `RestoreOptions.exact(*, reseed=None, ...)` | Preserves full game authority state. Supply `reseed` to replace only future rules RNG. |
| `RestoreOptions.shuffle_libraries(seed, ...)` | Permutes only exchangeable slots in each player's library using a local seeded RNG, on open or closed decks. Preserves both hands, face-down exile, future rules RNG, and every inspected, revealed, or positioned library card until a genuine rules shuffle. Reusing `seed` across sibling branches gives Common Random Numbers. |
| `RestoreOptions.determinize(viewer, seed, ...)` | Redraws hidden opponent cards consistently with what the chosen seat can see. The constructor requires `viewer` to be 0 or 1. Restore requires an open opponent decklist. |

Each constructor also accepts two keyword arguments.

- `enable_decision_memory` defaults to the live capture setting. Set it to
  `False` to freeze the existing logs or `True` to append from this boundary.
  Both retain the prefix and the setting for own-seat records. Search forks
  (`Env.clone_env()` and `fork_exact()`) always freeze their logs.
- `forced_first_action` commits one legal index after sampling and setting
  capture. It returns the resulting game, including a terminal result.
  Restore rejects a terminal root, an out-of-range index or a choice changed
  by determinization. For a built-in seat it records the override for replay.

`snapshot.metadata()` returns a frozen `SnapshotMetadata` with `turn`, `phase`
(`PhaseEnum`), `life`, `board_counts` (live battlefield permanents by
controller, including tokens), `stack_len`, `action_count`, `action_kind`
(`ActionSpaceEnum | None`), `acting_player` (`int | None`), `terminal`, and
sorted unique `active_card_defs` (catalog `CardDefId`s on the battlefield or
stack). On a terminal snapshot, `terminal` is `True`, `action_count` is `0`,
and `action_kind` and `acting_player` are `None`. Reading metadata never builds
observation JSON or opens hidden zones.

`snapshot.to_replay_json()` exports the seed, both decks, chooser and memory
settings, committed external actions, and ordered branch interventions. It
retains fingerprints for issued semantic frames so Commands keep their object
bindings. `GameSnapshot.from_replay_json(raw)` checks the schema, catalog and
replay checksum, replays the inputs, and verifies the final state digest.

`read_record(path)` checks the snapshot replay and its intervention offsets before
simulation. `walk(record)` then yields the verified decisions and reports
the first detected divergence if the rules have changed.

Unrecorded operations such as `scenario_*`, hidden-world materialization,
structured submissions, and external `random_action_index()` calls still
allow in-memory restore. Export raises when it cannot reconstruct the prefix.

Keep snapshot replays private. Academy opens one as a **Recorded game** when
forced-decision skipping is enabled, both seats use `agent`, and decklist
disclosure agrees. The acting view hides the opponent’s hand. The spectator
response includes both hands. The board draws the bottom seat’s cards and
shows the top seat’s hand count. Branch seeds stay on the server.

## Ask questions of recorded games

```python
from pathlib import Path

from managym.recording import read_records
from managym.statistics import Behaviour, CardPerformance, reduce

cards, behaviour = reduce(
    read_records(Path("games.jsonl")), [CardPerformance(), Behaviour()]
)
for row in cards[:3]:
    print(row)
print(behaviour)
```

```text
CardRow(seat=0, card='b9344ef3…:153', name='Fire Sages', games=1, wins=1, opening_hand=4, opening_won=4, drawn=0, drawn_won=0, gih=4, gih_won=4, not_seen=0, not_seen_won=0)
…
[BehaviourRow(seat=0, land_opportunities=15, lands_played=13, attack_opportunities=12, attacks=12), …]
```

Rows hold counts, so tables from different files add up; `row.gih_win_rate`
and `row.not_seen_win_rate` give the rates (`None` without games). `card` is
the card content digest and definition, unique within a world; `name` is for
reading.

`reduce` re-executes each record once and feeds every reducer the same
verified decisions. Two reducers are built in:

- `CardPerformance`: per card, games played, opening-hand and drawn copies, and
  the win rate of games with the card in hand against games where it was never
  seen. Game-in-hand, opening-hand and drawn count copies; games played and
  not-seen count games (`gih-copies-v1`). These are associations, not causal
  card values.
- `Behaviour`: per seat, how often a land was played at a decision that offered
  one, and how often at least one attacker was declared when one could be.

A decision with one legal action is never shown to the chooser and so never
counts as an opportunity.

Your own question is a class with three methods. This one asks how often a
seat passed priority while a spell was castable:

```python
from managym import ActionEnum
from managym.statistics import Reducer


class HeldSpells(Reducer):
    def start(self, setup):
        self.able = self.passed = 0

    def decision(self, decision):
        actions = decision.observation.action_space.actions
        if any(a.action_type == ActionEnum.PRIORITY_CAST_SPELL for a in actions):
            self.able += 1
            self.passed += actions[decision.chosen].action_type == ActionEnum.PRIORITY_PASS_PRIORITY

    def end(self, result, capture):
        return [{"able": self.able, "passed": self.passed}]
```

`start(setup)` receives both players' setups. `decision(decision)` receives
one `RecordedDecision`: the chooser's observation, original positional index
or native Command (`commitment`), the legal semantic `frame` for a Command,
and optional annotation. `chosen` gives its aligned action index. The native
semantic executor currently accepts action-aligned Commands with empty
answers; compound-offer submissions use a separate API and cannot be recorded
through `ScalarCapture.command`. Rejected inputs never reach reducers.
`end(result, capture)` receives the winner or stop and the game's
card movements and resolutions, and returns that game's rows. `reduce` returns
one list of rows per reducer, in the order given. A judgement such as "attacked
when the attack was good" belongs in a reducer of your own, since the engine
has no opinion on good.

The same reducers run during scalar or vector play. Pass `reducers=[...]` to
`ScalarCapture` or `CapturedVector`, and call `capture.rows()` for tables from
completed or closed games:

```python
from managym.recording import CapturedVector

capture = CapturedVector(num_envs=32, seed=7, reducers=[Behaviour(), HeldSpells()])
# Reset and step through the capture, using ordinary or buffered vector calls.
# No games_path is needed: these questions cover every game, not a saved sample.
capture.close()
print(capture.rows())
```

Each vector stream receives a deep copy of your reducers and owns independent
per-game state. `start` resets that state at every native root, including
autoreset and same-seed replacement. Only committed, active-stream decisions
reach `decision`; rejected isolated calls do not. `end` receives the old game's
card facts before autoreset discards them, or an interrupted result on root
replacement/close. Per-game rows are combined and the original reducer's
`finish(rows)` aggregates them; do not depend on stream-local mutable state in
`finish`. Rows accumulate until the capture is released, so return compact
per-game summaries. Record annotations are available to live hooks even when
record sampling is off.

A catchable custom callback error stops the custom analysis and is available
as `capture.analysis_error`; `capture.rows()` raises `AnalysisFailure` instead
of returning a partial table. It does not fail a committed native step or rewrite
the game record. A strict vector batch failure leaves commitment coverage
unknown, so its custom table is unavailable. Native errors still follow the
call's existing strict/isolated behavior. Offline `reduce` also raises
`AnalysisFailure` when its question fails.

With no reducers registered, there is no analysis observation read, card/exposure
capture or Python callback. Enabled hooks materialize each active chooser's full
observation and retain native card/exposure history; this is additional work on
the buffered path, not a zero-cost extension. Callbacks execute synchronously
in the calling producer's Python thread; they must return promptly. Built-in
native summaries remain the cheaper option when they answer the question.

manabot evaluation also accepts named reducers on `play_cell`. Those hooks run
inside its existing verification replay, sharing native card facts with audited
card statistics. Its [evaluation statistics](evaluation.md#custom-statistics-for-every-evaluation-game)
table aggregates once across all deal blocks; another `reduce()` call is not
needed to collect the same questions.

A record that stops early contributes its verified decisions and no result;
`CardPerformance` leaves it out of win rates and reports how many games it left
out. Reducers describe the games they were given. When only a share of games
was recorded, the rows describe that sample; the [native counters](#collect-in-game-statistics)
are the statistics meant for every game. Rows include private deck facts and
never reach an Academy viewer. manabot's audited card evaluation adds its
replay-passed eligibility, participant mapping and intervals on top of these
counts ([card evaluation](evaluation.md)).

## Export an Arena-style log

A record replays only in its own world. To keep a game readable after the
engine changes, or to hand it to a tool that reads MTG Arena logs, export it:

```python
from pathlib import Path

from managym.export_arena import export_log
from managym.recording import read_records

record = next(read_records(Path("games.jsonl")))
Path("game-seat0.log").write_text(export_log(record, viewer=0))
```

The export re-executes the game once and writes what seat 0 saw, in the shape
Arena's client logs: the connection message with the viewer's deck, a
`GREMessageType_GameStateMessage` for each card movement with its turn number
(the first also names phase, step and active player), and the match result.
Its first line marks the log as synthetic. The pinned 17Lands client parses it
into the opening hand, drawn cards, turns and result; no mulligan is recorded,
so that client reports its count as unavailable
([reader and limits](arena-export.md#reader-and-limits)).

Cards carry the Arena id of their recorded printing, from a table shipped with
managym. `managym.export_arena.coverage()` reports, for the running world, how
many cards have one; tokens and printings with no Arena id are written with id
`0` and counted there. The basic lands are among them: their recorded printing
is Alpha's, which Arena never had.

The log is a lossy, one-seat projection. Against the record it does not hold:

- the seed and setups, so it cannot be re-executed or opened in Academy;
- annotations: probabilities, values, update numbers and `extra`;
- the other seat's hand, library order and anything else the viewer could not
  see (export the other seat separately with `viewer=1`);
- the legal actions that were offered and not taken;
- cards without an Arena id, beyond their count and zone movements.

Keep the record as well for as long as its world is installable. These logs are
synthetic; never submit them to 17Lands or any other service.

## Locate a stall

Hold one `Diagnostics` for the run and name each phase of your loop:

```python
from pathlib import Path

from managym.diagnostics import Diagnostics

with Diagnostics(Path("run/diagnostics"), captures=[vector]) as diagnostics:
    with diagnostics.operation("collection") as operation:
        vector.step_into_buffers(actions)
        operation.progress(len(actions))
    with diagnostics.operation("learning"):
        update(batch)
```

If a phase stops returning, `run/diagnostics` already holds `operation.json`
(the phase, pending or returned, when it began and its sequence number),
`escalations.jsonl` (an appended history of when diagnostics escalated and why),
`stacks.txt` (every Python thread's stack) and `PRODUCER.snapshot.json` for each
capture's cached games. Later calls write `PRODUCER.jsonl`, readable with
`read_record`. Nothing has to be switched on first.

| Level | What runs |
|---|---|
| `"standard"` (default) | The operation marker, one small write per phase and never per step. A watchdog thread. A stack dump on `SIGUSR1` and from the watchdog. |
| `"detailed"` | Also the durable per-call [journal](#keep-a-durable-journal) of every capture and the engine's profiler scopes, reported in `info["profiler"]`. |
| `"off"` | Nothing: no per-step read, no write, no thread. |

The watchdog raises the level to `"detailed"` and records the reason when no
operation returns within `stall_seconds`, or when the rate you report with
`operation.progress(decisions)` falls below `slow_fraction` of its trailing
rate. You can raise it yourself with `diagnostics.escalate("reason")`, or start
at `Diagnostics(..., level="detailed")`. Mark inference microsteps with
`operation("inference", per_step=True)`; only detailed diagnostics writes those
markers. Progress reported by an operation measures that phase alone, excluding
time spent learning or exporting between collections.

What a stall leaves depends on the call that hung. The buffered `VectorEnv`
calls, the rollout pool and flat-MC scoring release Python's interpreter lock
while the engine works, so the watchdog runs during them and writes the game
cursors and stacks itself. Scalar `Env.step` and `reset`, and the
object-returning `VectorEnv.step` and `reset_all`, hold the lock: during one of
those no Python thread runs, the marker still names the phase, and the stack
comes from `kill -USR1 <pid>`, which is handled without the lock. A journal
that was not on when a call hung cannot be written for that call afterwards;
the cursors are the games up to the last call that returned.

A pending marker means no return was recorded. It does not mean the call
failed, or that it entered the engine. Diagnostics skip no game and retry
nothing. Engine-side messages for people go to the `managym` logger, which you
configure like any other.

## Add training or migrate an existing project

Install `manabot` when you need training, evaluation or its optional clients:

```bash
uv pip install --python .venv/bin/python --only-binary :all: --torch-backend cpu manabot
uv run --no-project --python .venv/bin/python manabot --help
uv run --no-project --python .venv/bin/python python -c 'import managym; from manabot import managym as compatibility; assert compatibility.Env is managym.Env'
```

`manabot` requires exactly its own version of `managym`; resolution may update
the existing engine to that matching version. When the environment already has
`manabot` 0.2.0 or 0.2.1, which bundled the engine and owns `managym/__init__.py`,
uninstall it before installing `managym` or a newer `manabot`; removing it
afterwards deletes that file. Repair an affected environment by reinstalling
the engine (`uv pip install --reinstall-package managym 'managym==<version>'`
with uv). Keep a project lockfile, or pin
both to the same published version. A resolver conflict between the two is
fixed by aligning versions, not by using `--no-deps`. See
[training installation](development.md#install) for extras and Torch backends.

Engine-only projects should depend on `managym` and use `import managym`.
Projects using `from manabot import managym` or `manabot.managym` can keep those
imports when `manabot` is installed: they reach the same engine classes and
submodules, without a second extension or game state. Installing `managym`
alone does not provide the `manabot` namespace. Python training wrappers such
as `manabot.env.vector_env.VectorEnv` still require `manabot`; the native
batch class is `managym.VectorEnv`.

The two packages release together at the same version. Package upgrades do
not change frozen observation layouts. Retain the package versions, configuration
and seeds with collected data, and reproduce old runs in their original
environments.

Recorded search annotations can also supply `search_scores` in legal-action
order and a `belief` summary (`model`, `digest`, `condition_mass`,
`sampled_worlds`). Academy presents those public outputs. The annotation's
arbitrary `extra` payload and record context remain private; evaluation keeps
its full search receipts there for offline audit.
