# Migrate hypers.py from dataclasses to Pydantic, eliminate schema.py duplication

## Context

Every config field exists twice — once as a Pydantic model in `schema.py`, once as a dataclass in `hypers.py`. `TrainingConfig.to_hypers()` bridges them via `model_dump()` → `**kwargs`. This creates field drift risk and unnecessary indirection. All hypers are immutable after init, so Pydantic models are a drop-in replacement.

## Approach

Make `hypers.py` the single source of truth using Pydantic `BaseModel`. Delete the duplicate classes from `schema.py`. Since `Hypers` will now *be* the validated Pydantic model, `TrainingConfig` and `SimulationConfig` (and their `to_hypers()` methods) become unnecessary.

## Findings from codebase exploration

- **13 `asdict()` calls** across presets.py, experiment.py, train.py, verify/util.py → all become `.model_dump()`
- **No field mutation** anywhere — all hypers are read-only after init
- **One `__post_init__`** in `Hypers` (validates max_cards_per_player, max_actions) → becomes `@model_validator(mode="after")`
- **One `field_validator`** for `target_kl` (coerces "inf" string) currently lives only in schema.py → moves to hypers.py
- **No `isinstance` checks** on any hypers class
- **No dataclass-specific behavior** beyond `asdict()` and `field(default_factory=...)`
- Files that import hypers for type annotations and field access (`env/match.py`, `env/observation.py`, `model/agent.py`, `sim/player.py`, `sim/sim.py`) need no changes — same field names, same access patterns

## Changes

### 1. `manabot/infra/hypers.py` — dataclass → BaseModel

- Replace `@dataclass` with `BaseModel` + `ConfigDict(extra="forbid")`
- Replace `field(default_factory=...)` with `Field(default_factory=...)`
- Replace `__post_init__` validation with `@model_validator(mode="after")`
- Add `target_kl` field validator (currently only in schema.py's `TrainConfig`)
- Drop `initialize()` no-op function

### 2. `manabot/config/schema.py` — delete

All duplicate sub-config classes (ObservationConfig, MatchConfig, TrainConfig, RewardConfig, AgentConfig, ExperimentConfig, SimConfig) and wrappers (TrainingConfig, SimulationConfig) are redundant once hypers.py is Pydantic.

### 3. `manabot/config/load.py` — use Hypers directly

- `load_train_config()` returns `Hypers` (via `Hypers.model_validate(config)`)
- `load_sim_config()` returns `SimulationHypers` + `ExperimentHypers`
- Remove imports of deleted schema classes

### 4. `manabot/config/presets.py` — `asdict()` → `.model_dump()`

- `get_training_base()`: `Hypers().model_dump()`
- `get_sim_preset()`: `SimulationHypers().model_dump()`

### 5. `manabot/config/__init__.py` — update exports

Remove schema re-exports, keep load/presets exports.

### 6. `manabot/verify/util.py` — `asdict()` → `.model_dump()`

- `build_hypers()`: `Hypers().model_dump()` and validate via `Hypers.model_validate(merged)`
- Remove import of `TrainingConfig`

### 7. `manabot/infra/experiment.py` — `asdict()` → `.model_dump()`

`_get_flattened_config()`: use `.model_dump()` on each sub-config.

### 8. `manabot/model/train.py` — `asdict()` → `.model_dump()`

Checkpoint serialization uses `asdict()` on agent hypers, observation hypers, and train hypers.

### 9. `manabot/infra/__init__.py` — remove `initialize` re-export if dropped

### 10. Passive consumers — no changes needed

`env/match.py`, `env/observation.py`, `model/agent.py`, `sim/player.py`, `sim/sim.py` use hypers for type annotations and field access. Same API surface.

### 11. Tests

- `tests/config/test_load.py` — update for `Hypers` return type instead of `TrainingConfig`
- `tests/verify/test_util.py` — likely no changes
- `tests/model/test_train.py` — likely no changes (constructs hypers with kwargs, which Pydantic supports)

## Verification

```bash
pytest tests/config/ tests/verify/ tests/model/ tests/env/ -v
python -c "from manabot.infra import Hypers; h = Hypers(); print(h.model_dump())"
python -c "from manabot.config.load import load_train_config; c = load_train_config('local'); print(type(c))"
```
