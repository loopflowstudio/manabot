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
[per-seat controllers](engine.md#built-in-heuristic-opponent) for rewards,
terminal reset and Command stepping.
Forced single-action decisions are skipped by default. See the
[engine guide](engine.md#choosing-a-python-api) for complete games, batches,
reset behavior, Commands and hidden-information boundaries, and
[encoded observations](engine.md#encoded-observations) for reading the native
arrays by column name and joining card definition IDs.

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
with uv, `pip install --force-reinstall 'managym==<version>'` with pip). Keep a project lockfile, or pin
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
