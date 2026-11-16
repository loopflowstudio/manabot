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

Enable statistics before the first reset, then play through the ordinary API:

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

Statistics require neither detailed recording nor replay, perform no file
writes, and do not enable card/exposure history capture. Counter storage grows
with encountered card definitions and physical copies, rather than decisions;
generated copies can still grow during a game. The Env's existing action trace
remains independent, so this is not a constant-memory promise for the whole
Env. Statistics and private recording can be enabled separately or together;
neither changes actions, seeds or legal choices.

These summaries include private deck facts; do not expose them as Academy viewer
data. Persist summaries yourself when needed. An interrupted process can lose its
in-memory counts; label the last saved summary partial and report the gap.
Compatible private recordings can recover known-prefix counts by replay.
Add `--statistics` to the standalone replay command below to count its reconstructed
prefix; these describe the replay, never the original unknown outcome. Counts
alone cannot reconstruct a game. Live summaries also do not replace
replay verification for audited evaluation: manabot retains its replay-passed
eligibility rules and participant mapping when exporting Etude statistics.

## Record private replay inputs

Wrap every mutation from the first reset onward. This example retains a private
rolling journal while playing one decision, using only the `managym` install:

```python
from pathlib import Path

import managym
from managym.recording import ScalarCapture

game = managym.Env(seed=42)
capture = ScalarCapture(
    seed=42,
    journal_path=Path("private-games/game.jsonl"),
    context={"attempt": "local-example"},
)
players = [
    managym.authored_deck_setup("tla-baseline-v1", key)
    for key in ("tla_red", "tla_white")
]
observation, info = capture.reset(game, players)
observation, reward, terminated, truncated, info = capture.step(game, 0)
```

Keep the recorder seed and trivial-decision setting equal to the Env's. Use the
recorder for subsequent mutations too; direct Env mutations leave the retained
prefix behind. To collect statistics as well, call `game.enable_statistics()`
before `capture.reset`. Recording does not enable statistics implicitly.

For native batches, `managym.recording.CapturedVector` owns its `VectorEnv`.
Construct it with `num_envs`, `seed`, an explicit `journal_path`, and optional
`context`; use `reset_all(players)` and `step(actions)`, or its buffered methods.
Its `enable_statistics`, `statistics` and `take_statistics` methods keep global
stream identities too. Assign each producer a distinct
path and use `stream_offset` for global stream identities. The caller owns the
destination and lifetime; the engine does not discover a training run or upload
files. No destination means no durable rolling journal. Detailed recording is
opt-in; ordinary Env use adds no diagnostic reads or durable writes.

The journal retains actual observed setups, known action prefixes and
pending/returned native-call boundaries. Cached intent is durable before native
decision-frame preparation. A pending call means no durable return
acknowledgment: it may not have entered native code, or may have returned before
the acknowledgment could be saved. Batch return does not prove each row
succeeded. Sampled reset inputs remain unknown until observed; previous decks
are never substituted as the new setup.

Copy the file to a private location before inspecting a live writer:

```python
from pathlib import Path
from managym.recording import read_failure

evidence = read_failure(Path("private-games/copied-game.jsonl"))
print(evidence.boundary)
```

For bounded replay of the copy:

```bash
uv run --no-project --python .venv/bin/python python -m managym.replay private-games/copied-game.jsonl --seconds 30
```

Replay checks the world, native extension and engine-source identities. It
reconstructs the known prefix and can attempt a pending indexed action only
with its retained fresh binding. Unknown setup or binding is reported as
unavailable. The original interrupted outcome stays unknown even if replay
succeeds. Older manabot receipts keep their original source-digest meaning;
when legacy source verification requires manabot, use its
[replay command](training.md#inspect-and-replay-a-journal) in the original
environment. An explicit changed-runtime diagnostic override does not permit a
different world. Replay does not restore a process or optimizer.

These are privileged inputs, including hidden decks, not Academy viewer
recordings or policy observations. Files use mode 0600 and directories 0700.
Each boundary requires serialization and fsync, which can reduce throughput.
Automatic atomic compaction replaces superseded journal history when appended
bytes reach the larger of the last checkpoint size and 64 KiB, allowing one
record's overshoot and a temporary checkpoint. Storage still grows with active
prefixes. Successful games supersede earlier inputs; this is not a full game
archive. Final cursors and immutable failure receipts remain separate.

The reader warns on a torn final append and uses complete records; complete
corruption, sequence gaps or an incomplete first checkpoint fail inspection.
If pending intent cannot be persisted, the native operation is not entered.
A persistence error after execution does not mean the operation did not run.
Retain existing evidence and repair storage before starting a new attempt;
there is no automatic retry or game skipping. Local durability cannot protect
against host deletion: collect private files before removing the machine.

A journal can identify engine work, but cannot identify what its caller does
between calls. The runner must record inference, learning and other phases
outside native execution. Training provides those markers through
[`--training-game-logs`](training.md#diagnosing-stalled-training).

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
not change frozen observation layouts or make old checkpoints compatible
with a new world. Retain the package versions, world identity, configuration
and seeds with collected data, and reproduce old runs in their original
environments.
