# v0.2.1

Install the training library and its matching rules engine in a fresh directory:

```bash
uv venv --python 3.13
uv pip install --python .venv/bin/python --only-binary :all: --torch-backend cpu 'manabot==0.2.1'
uv run --no-project --python .venv/bin/python manabot --help
```

The last command lists the CLI commands without starting training. Wheels target
ordinary CPython 3.12 and 3.13 on Linux x86-64 (glibc 2.28+) and Apple Silicon
(macOS 14+). Use `--python 3.12` to select the other supported interpreter.
`manabot` is a pure-Python distribution requiring `managym==0.2.1`. The
separate `managym` distribution owns the compiled engine and `import managym`;
`manabot.managym` remains a compatibility alias to the same classes and submodules.
Engine-only consumers can install `'managym==0.2.1'` without Torch, Gymnasium
or Etude. See the [engine installation guide](https://github.com/loopflowstudio/manabot/blob/v0.2.1/docs/engine-python.md).
See the [installation guide](https://github.com/loopflowstudio/manabot/blob/v0.2.1/docs/development.md#install)
for extras, backend selection and dependency troubleshooting.

This research release includes API removals and world changes since 0.2.0.
Review the migration guidance before upgrading an existing collector or run.

## Read scalar and batched hands

`HandSnapshot` reports authoritative hand contents for an explicit native seat:

```python
from manabot import managym
from manabot.env.match import Match

games = managym.VectorEnv(num_envs=2, seed=42, opponent_policy="none")
games.reset_all(Match().to_rust())
hands = games.hand_snapshots([1, 0])

for slot, hand in enumerate(hands):
    print(slot, hand.player, hand.cards)
```

This prints seat 1's hand in slot 0 and seat 0's hand in slot 1 as card-name
counts, including duplicates. Each detached record carries `player`, `game_id`,
`seed`, `revision` and `cards`; an empty mapping means an empty hand.

- Scalar native and Gym `Env.hand_snapshot(player)` return the same record.
  Native `VectorEnv`, `CapturedVector` and Torch `VectorEnv` expose
  `hand_snapshots(players)`, with one seat per slot and results in slot order.
- Reads do not step, reset or consume randomness. They contain hidden
  information even with closed decklists and remain separate from viewer-safe
  observations, policy inputs and exported captures. Clients own how they use
  the contents.
- `current_game_generations()` on the native vector and its capture wrapper
  distinguishes resets and replacements, including resets to the same seed.
  Snapshot game IDs also distinguish scalar forks and materialized worlds.
  These IDs are process-local bookkeeping, not portable replay identities.
- Read before a vector step when retaining the current hands: completed slots
  reset before returning, so a later snapshot describes the replacement game.
  Scalar terminal hands remain available until explicit reset. After a failed
  step, discard that call's results and recover or reset before reading again.

The [hand snapshot guide](https://github.com/loopflowstudio/manabot/blob/v0.2.1/docs/engine.md#hand-snapshots)
defines identity, seed and error behavior. The
[scalar/batch API audit](https://github.com/loopflowstudio/manabot/blob/v0.2.1/docs/python-api-audit.md)
compares the APIs and records their remaining differences; general fixed-seat
batched observations and batched fork/search are not provided.

## Migrate collectors and reset callers

| Previous use | 0.2.1 behavior and migration |
|---|---|
| Gym `Env(auto_reset=...)` | The argument is removed and raises `TypeError`. Remove it, inspect `terminated` and `truncated`, then call `reset()` explicitly. Terminal steps return terminal observations and true done flags; `true_terminated` / `true_truncated` info keys are removed. |
| Vector `opponent_policy="passive"` | Removed; construction raises `ValueError`. Choose `"none"` to handle both seats or `"random"` for a native random opponent. The default is `"none"`. Retained passive-policy receipts require their original source to replay. |
| Readable `reset_all`, `step` or other undelegated native mutation through `CapturedVector` | Raises `AttributeError`. Use the wrapper's `reset_*_into_buffers` / `step_*_into_buffers` methods so replay prefixes track the actual game. Native read methods remain explicitly delegated. |
| Torch `VectorEnv.reset(seed=s)` with the current base seed | Every explicit seed restarts slot `i` at `s + i`, reproducing the deal. Use `reset()` without a seed to continue seed succession. |

Native and Torch vector steps still automatically reset finished slots. Their
rewards and done flags describe the completed game; returned observations
describe its replacement. Torch observation tensors are reused: clone values
retained across steps. Hand snapshots need no policy-input or checkpoint
migration by themselves.

## Choose dependencies and training devices

Public requirements allow `torch>=2.10.0` and `etude-sdk>=0.2.15`, including the
Etude requirements of optional extras. Consumer projects can resolve these
with their own dependencies; repository runtime groups retain exact backend
pins. Keep a project lockfile for reproducibility. A successful resolution does
not certify every future dependency version. Pandas is no longer a direct
dependency; notebook file generation remains in the core requirements.

Fresh local training recipes select CUDA first, then Apple MPS, then CPU.
Baseline requires CUDA or MPS. Apple Silicon supports float32 ordinary self-play;
MPS belief inputs, frozen opponents and process recovery remain unsupported,
and reduced precision remains CUDA-only. Evaluation and search use CPU for this
workflow. See [training](https://github.com/loopflowstudio/manabot/blob/v0.2.1/docs/training.md#run-training)
for explicit device selection and limitations. Release install checks exercise
CPU execution, not accelerator performance.

## Other changes since 0.2.0

- **Cards, decks and decisions:** cards are native typed definitions shared by
  play and learning. The catalog includes the TLA set, with Sagas, transforming
  faces, extra turns, continuous effects and opponent-control decisions.
  Imported decklists and open/closed disclosure use native admission. Forced
  single-action decisions are skipped by default in runners and Academy;
  simultaneous trigger ordering and card naming use shared native choices.
- **Policy inputs:** explicit v7–v9 inputs add shared effects, richer execution
  context and name identities. Fresh inputs remain v6; older selected layouts
  retain their contracts. New model configurations use catalog token rows for
  fact and shared-effect identities; checkpoint loaders preserve saved meanings.
- **Training and evaluation:** deck-source self-play can sample trophy-deck
  matchups, record exposure and recover from explicitly admitted observation
  capacity failures. Evaluation batches games and stores card evidence in
  Etude artifacts. Engine view caching, batched local-update rollouts and
  self-play metric indexing reduce repeated work; the linked guides and
  experiment reports retain the scope of each measurement.
- **Release workflow:** one tag builds both packages at the same version:
  four engine wheels, one pure-Python training wheel and two source archives.
  Both inventories are validated before publishing the engine, then training.
  After both PyPI uploads, the workflow creates a GitHub Release from these
  notes and adds both PyPI links. Uploads are not atomic across projects;
  completion checks the retained seven-file inventory and hashes. A complete
  engine upload with matching hashes can resume a missing training upload;
  partial files within either package or conflicting bytes stop for inspection.

The current world is open **w17**, versus w11 in 0.2.0. Package version numbers
do not override world, content or checkpoint bindings. Use an older run's
original source to reproduce it; do not upgrade its pinned environment or
rewrite frozen receipts. See [world compatibility](https://github.com/loopflowstudio/manabot/blob/v0.2.1/WORLDS.md)
and the [engine guide](https://github.com/loopflowstudio/manabot/blob/v0.2.1/docs/engine.md).
This release does not certify complete rules coverage or playing strength.
