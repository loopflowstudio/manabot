# 02: Python Integration Tests Green

## Problem

The Rust/PyO3 managym backend compiles, passes `cargo test`, and has been
manually smoke-tested — but `pytest tests/env/ tests/agent/` has never run
against it. These 16 tests are the contract between the game engine and the
RL training pipeline. Until they pass, the migration is theoretical.

## Approach

Set up a working Python environment with maturin, run the tests, and fix
every failure on the **Rust/binding side only** — zero test modifications.

The binding surface already exists (877 lines of `bindings.rs`). This is an
integration debugging task, not a design task. The work is: discover what
the actual mismatches are, fix them, and verify.

## Alternatives considered

| Approach | Tradeoff | Why not |
|----------|----------|---------|
| Fix tests to match Rust API | Faster to unblock, but breaks C++ compatibility | Wave constraint: zero test modifications |
| Dict-backed collections in PyO3 | Matches original C++ API exactly | Stale — observation.py already updated for list-backed access. Tests use `cards[0].id` and `len(cards)`, not `.keys()` |
| Python IntEnum wrappers | Familiar Python enum semantics | PyO3 `eq_int` already provides `__int__`, `__index__`, and `==` comparison. Tests only use these operations |

## Key decisions

### 1. Collections are List, not Dict

The earlier design doc claimed dict-backed collections were needed. **This is
stale.** Current code:
- `observation.py:216` does `cards[:self.cards_per_player]` (list slice)
- `test_observation.py:188` does `observation.agent_cards[0].id` (list index)
- `bindings.rs` exposes `Vec<PyCard>` as Python `list`

All three agree. No conversion needed.

### 2. Negative action index handling

`test_managym.py:96` tests `env.step(-1)` and expects an error containing
"Action index". The Rust binding currently takes `action: usize`
(`managym/src/python/bindings.rs`) — PyO3 will raise `OverflowError` on
negative input before our code runs.

**Fix:** Change the PyO3 signature to accept `isize` (or `i64`), bounds-check
in Rust, and raise `AgentError("Action index -1 out of bounds: N")`.

### 3. InfoDict is empty on reset, sparse on step

- `test_env.py:67` checks `all(isinstance(v, (int, float, bool, dict)) for v in info.values())`
- Rust `reset()` returns `empty_info_dict()` → empty dict → `all()` on empty iterable is True ✓
- Rust `step()` returns `winner_name` (string) only on game end
- The Python `Env` wrapper adds `true_terminated` (bool), `true_truncated` (bool),
  `action_space_truncated` (bool) — the test only checks `isinstance(step_info, dict)` ✓

No fix needed for info dict types.

### 4. Reward pipeline

- Rust engine returns `1.0` / `-1.0` for win/loss, `0.0` otherwise
- `test_managym.py:137` asserts `−1.0 ≤ final_reward ≤ 1.0` — passes against raw engine ✓
- `test_env.py:42` uses `Reward(RewardHypers(trivial=True))` which returns `1.0` always ✓
- `test_env.py:95` checks `isinstance(reward, float)` — Rust returns `f64` → Python `float` ✓

No fix needed.

### 5. Enum indexing into zone_counts

`test_managym.py:160` does `obs.agent.zone_counts[managym.ZoneEnum.BATTLEFIELD]`.
`zone_counts` is a Python `list`, `ZoneEnum.BATTLEFIELD` has `__index__() → 2`.
Python lists call `__index__` for non-int subscripts. ✓

No fix needed.

## Predicted failure modes

From code review, ranked by likelihood:

| # | Failure | Root cause | Fix location |
|---|---------|-----------|--------------|
| 1 | `test_invalid_action_handling` — `-1` action (**confirmed mismatch**) | PyO3 `usize` rejects negative ints with `OverflowError` | `bindings.rs`: accept `i64`, bounds-check manually |
| 2 | Build/import failures | maturin config, module path, missing `__init__.py` re-exports | `pyproject.toml`, module layout |
| 3 | Action space corruption after invalid/failed step | `Game::step` consumes `current_action_space` and may not restore on errors | `game.rs`: restore action space before returning `Err` |
| 4 | Game behavior differences | Rust engine produces different game states than C++ (card ordering, timing) | Engine-level fixes in Rust |
| 5 | `test_agent_turns_distribution` | `player_index` encoding differs between engines | `observation.rs` or `bindings.rs` |

## Scope

- **In scope:** Environment setup, maturin build, running tests, fixing Rust
  binding/engine code to pass all 16 tests, CI workflow verification
- **Out of scope:** C++ removal (wave 03), performance benchmarking (wave 03),
  new features, test modifications

## Done when

```bash
source .venv/bin/activate
maturin develop --features python
pytest tests/env/ tests/agent/ -v
python -c "import managym; assert int(managym.ZoneEnum.STACK) == 4"
```

All pass. No test files modified. Advances wave goal: *"All pass, zero test
modifications. CI enforces this on every PR."*
