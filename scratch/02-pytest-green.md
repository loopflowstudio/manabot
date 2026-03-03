# 02: Python Integration Tests Green

## Problem

The Rust engine compiles, passes `cargo test`, and the PyO3 binding surface
exists — but no one has run `pytest tests/env/ tests/agent/` against it.
These 16 tests are the contract between the game engine and the RL training
pipeline. Until they pass, the migration is theoretical.

The fixes already on this branch (negative action index, action space
preservation) were derived from code review. They might be right. They might
be subtly wrong. Running the tests is the only way to know.

## Approach

**Triage-first, fix bottom-up.** Run the full suite once to see all failures,
classify them, then fix from lowest-level (import/build) to highest
(engine behavior). No guessing at failures from code review — the previous
round of that already happened and we need to validate it.

### Phase 1: Environment bootstrap

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install maturin pytest torch numpy pandas gymnasium hydra-core wandb
uv pip install -e .
maturin develop --features python
python -c "import managym; print(managym.Env())"
```

If the import fails, nothing else matters. Fix build issues first.

### Phase 2: Full triage run

```bash
pytest tests/env/ tests/agent/ -v --tb=short 2>&1 | tee scratch/triage.log
```

Classify every failure into one of four buckets:

| Bucket | Example | Fix location |
|--------|---------|-------------|
| **Import/build** | `ModuleNotFoundError`, `ImportError` | `pyproject.toml`, module layout |
| **API mismatch** | Missing attribute, wrong type | `bindings.rs`, `convert.rs` |
| **Encoding mismatch** | Wrong shape, NaN, wrong values | `observation.rs`, `bindings.rs` |
| **Engine behavior** | Wrong game state, validation failure | Rust engine (`game.rs`, `flow/`, `state/`) |

### Phase 3: Fix in dependency order

Fix imports before API mismatches, API mismatches before encoding, encoding
before engine behavior. Each fix gets a `cargo test` + targeted `pytest -k`
before moving on.

### Phase 4: Full green run

```bash
maturin develop --features python
pytest tests/env/ tests/agent/ -v
python -c "import managym; assert int(managym.ZoneEnum.STACK) == 4"
```

All pass. Commit.

## Alternatives considered

| Approach | Tradeoff | Why not |
|----------|----------|---------|
| Fix tests to match Rust API | Faster, but breaks C++ compatibility | Wave constraint: zero test modifications |
| Test-by-test incremental | Lower risk per step | Slower — you don't see the full failure surface and may fix things that mask other issues |
| Compatibility shim layer | Absorbs API differences in Python | Adds a permanent translation layer. The binding surface already exists and should match directly. |

## Key decisions

### Run all tests before fixing anything

The previous iteration of this design predicted failure modes from code
review. Two of those predictions led to code changes. Those changes are
untested. Running the full suite first reveals whether the predictions were
right *and* surfaces failures that code review missed.

### Fix Rust code, never test code

The wave constraint is zero test modifications. Every mismatch is a bug in
the binding or engine code. If a test expectation seems wrong, that means
our understanding is wrong — read the test more carefully.

### Engine behavior fixes are in scope

If `test_observation_validation` fails because the Rust engine assigns wrong
`owner_id` to cards, or `test_play_land_and_cast_spell` fails because the
engine doesn't offer the right actions, those are engine bugs that need
fixing. This isn't just a binding polish task.

### VectorEnv is the hardest test

`test_vectorenv_tensor_outputs` spawns 7 subprocess workers via
`AsyncVectorEnv`. Each worker imports `managym` and runs a full game.
If the maturin-built module doesn't survive `fork()` or can't be imported
in subprocess contexts, this test fails in ways that are hard to diagnose
from the error alone. If it fails, test it in isolation with
`pytest -k test_vectorenv -s` and check subprocess stderr.

## Predicted failure surface (from code review)

Ranked by likelihood, with specific test → code path analysis:

### 1. Engine validation failures (HIGH)

`test_observation_validation` calls `obs.validate()` every step. Validate
checks that:
- `agent.id != opponent.id`
- Exactly one player has `is_agent = true`
- Every card in `agent_cards` has `owner_id == agent.id`
- Every card in `opponent_cards` has `owner_id == opponent.id`
- Every permanent in `agent_permanents` has `controller_id == agent.id`
- Every permanent in `opponent_permanents` has `controller_id == opponent.id`

If the Rust `Observation::populate_cards` or `populate_permanents` methods
partition incorrectly (e.g., by controller instead of owner for cards, or
vice versa), validation fails silently (returns `false`) and the test
asserts on it.

**Fix location:** `managym/src/agent/observation.rs` — `populate_cards`,
`populate_permanents`, and the `From<Observation>` conversion.

### 2. Player alternation (MEDIUM)

`test_agent_turns_distribution` needs `agent.player_index` to take at least
2 distinct values over a game. This requires the Rust priority system to
correctly alternate the "agent player" between turns. If `agent_player()`
always returns the same index, the test fails with
`"Expected at least 2 different agent indices"`.

**Fix location:** `managym/src/agent/env.rs` — how `agent_player` is
determined for each action space.

### 3. Play land / cast spell (MEDIUM)

`test_play_land_and_cast_spell` searches for `PRIORITY_PLAY_LAND` and
`PRIORITY_CAST_SPELL` actions. With the test decks (Mountain + Grey Ogre),
the agent's first main phase should offer a land play. After playing a
Mountain, if the agent has enough mana and it's still main phase, a spell
cast should be available.

If the Rust priority system doesn't generate these action types, or
generates them with different enum values, the test's `find_action` returns
`None` and the test passes vacuously (the assertions are inside
`if land_idx is not None`). So this test might *pass* even with a broken
engine — which is worse than failing. Watch for this.

**Fix location:** `managym/src/flow/priority.rs` — action generation.

### 4. Game termination (LOW-MEDIUM)

Multiple tests loop until `terminated`. If the Rust engine has a bug that
prevents games from ending (e.g., infinite priority cycling, combat damage
not applied), tests hang at the 2000/10000 step limit and fail with
`"Game should eventually terminate"`.

**Fix location:** `managym/src/flow/` — combat damage, state-based actions,
player death detection.

### 5. Info dict type check (LOW)

`test_initialization` checks `all(isinstance(v, (int, float, bool, dict))
for v in info.values())`. Rust `reset()` returns empty dict → `all()` on
empty is `True`. But if we accidentally add a string value to the reset
info dict, this fails. The step info dict *does* contain `winner_name`
(string) on game end, but `test_initialization` only checks reset info.

**Fix location:** Probably nothing to fix, but verify `empty_info_dict()`
in `env.rs`.

### 6. AsyncVectorEnv subprocess import (LOW)

Maturin-built native modules generally survive fork+import. But if the
Rust code uses thread-local state or the `Mutex<Env>` causes issues in
forked processes, `test_vectorenv_tensor_outputs` could fail with obscure
errors.

**Fix location:** Likely a non-issue. If it fails, check that
`managym._managym` is importable in a subprocess.

## Scope

- **In scope:** Python environment setup, maturin build, running all 16
  tests, fixing Rust binding and engine code to pass every test, profiler
  smoke test
- **Out of scope:** C++ removal (wave 03), performance benchmarking (wave 03),
  new card implementations, changes to any Python test file

## Done when

```bash
source .venv/bin/activate
maturin develop --features python
pytest tests/env/ tests/agent/ -v
python -c "import managym; assert int(managym.ZoneEnum.STACK) == 4"
```

All pass, zero test modifications.

Advances wave goal: *"Rust engine + binding tests are green (cargo test,
pytest tests/env/ tests/agent/)."*
