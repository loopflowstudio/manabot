# Rust VectorEnv

## Problem

Training throughput is bottlenecked at ~1,600 SPS by the Python environment
layer. The current `VectorEnv` wraps `gymnasium.vector.AsyncVectorEnv`, which
spawns N subprocesses. Each subprocess independently calls into Rust via PyO3,
encodes observations in Python, and serializes results back over IPC. The game
engine is fast; the orchestration around it is not.

This is step 1 of the SPS wave. It eliminates subprocess overhead by owning all
N games in a single Rust struct and stepping them in one PyO3 call.

**Wave goals advanced:**
- "Rust VectorEnv that owns N games and steps them in one call" (goal 1)
- Prerequisite infrastructure for goals 2-5

## Approach

### Rust: `VectorEnv` struct

`managym/src/agent/vector_env.rs` owns N `Env` instances. Single `step()` call
processes all envs sequentially, returns batch results. Handles auto-reset and
opponent auto-play internally to minimize Python round-trips.

```rust
pub struct VectorEnv {
    envs: Vec<Env>,
    player_configs: Vec<PlayerConfig>,
    opponent_policy: OpponentPolicy,
}

pub enum OpponentPolicy {
    None,       // Both players controlled externally
    Passive,    // Auto-play opponent: always pass priority
    Random,     // Auto-play opponent: uniform random
}

impl VectorEnv {
    pub fn new(
        num_envs: usize,
        seed: u64,
        skip_trivial: bool,
        opponent_policy: OpponentPolicy,
    ) -> Self;

    pub fn reset_all(
        &mut self,
        player_configs: Vec<PlayerConfig>,
    ) -> Vec<(Observation, InfoDict)>;

    /// Step all envs. Auto-resets terminated games. Auto-plays opponent turns.
    /// Returns post-reset obs for terminated envs, with terminal reward/flags preserved.
    pub fn step(
        &mut self,
        actions: &[i64],
    ) -> Vec<StepResult>;
}

pub struct StepResult {
    pub obs: Observation,
    pub reward: f64,
    pub terminated: bool,  // Terminal outcome from the step (pre-reset truth)
    pub truncated: bool,   // Truncation outcome from the step (pre-reset truth)
    pub info: InfoDict,
}
```

Each env gets a unique seed derived from `base_seed + env_index`. On auto-reset,
the seed increments to produce a fresh game.

**Opponent auto-play in Rust**: When `opponent_policy != None`, after each step
(and after reset), the VectorEnv loops through opponent turns internally using
the specified policy. `Passive` picks the pass-priority action when available
(falling back to action 0). `Random` picks uniformly from valid actions using
the env's RNG. This eliminates per-opponent-turn Python round-trips entirely.

**Auto-reset semantics**: Match the existing trainer-facing contract exactly.
When a game terminates:
1. Record terminal reward, terminated/truncated flags
2. Reset the env with fresh player configs
3. Skip opponent turns in the new game
4. Return post-reset observation with pre-reset reward
5. Return `terminated`/`truncated` directly from the pre-reset terminal transition

No dual done-flag contract (`true_terminated`) in the Rust path.

### PyO3 bindings

`managym/src/python/vector_env_bindings.rs` exposes `VectorEnv` to Python:

```python
class VectorEnv:
    def __init__(
        self,
        num_envs: int,
        seed: int,
        skip_trivial: bool,
        opponent_policy: str,  # "none", "passive", "random"
    ) -> None: ...

    def reset_all(
        self,
        player_configs: list[PlayerConfig],
    ) -> list[tuple[Observation, dict]]: ...

    def step(
        self,
        actions: list[int],
    ) -> list[tuple[Observation, float, bool, bool, dict]]: ...
    # Returns: (obs, reward, terminated, truncated, info)
```

### Python wrapper

`manabot/env/rust_vector_env.py` — thin wrapper implementing the same interface
as `VectorEnv` (`manabot/env/env.py:169`):

```python
class RustVectorEnv:
    def __init__(self, num_envs, match, observation_space, reward, device,
                 seed: int,
                 opponent_policy=None):
        policy_name = opponent_policy_to_str(opponent_policy)
        self._base_seed = seed
        self._rust_env = managym.VectorEnv(
            num_envs=num_envs,
            seed=seed,
            skip_trivial=True,
            opponent_policy=policy_name,
        )
        self._player_configs = match.to_rust()
        self.observation_space = observation_space
        self.reward = reward
        self.device = device
        self.num_envs = num_envs

    def reset(self, seed=None, options=None):
        if seed is not None and seed != self._base_seed:
            self._base_seed = seed
            self._rust_env = self._rebuild_rust_env(seed=seed)
        if options and "match" in options:
            self._player_configs = options["match"].to_rust()
        results = self._rust_env.reset_all(self._player_configs)
        return self._process_results(results)

    def step(self, actions: torch.Tensor):
        actions_list = actions.cpu().tolist()
        results = self._rust_env.step(actions_list)
        return self._process_step_results(results)
```

`_process_step_results` calls `ObservationSpace.encode()` on each raw
observation, stacks into batch tensors, and moves to device. This is the same
encoding path as today — just called from the main process instead of
subprocesses. Done tensors come directly from `terminated`/`truncated`.

### Trainer integration

`manabot/model/train.py` — add `use_rust_env` flag to `TrainHypers`:

```python
class TrainHypers(BaseModel):
    use_rust_env: bool = True  # Default to new path
    ...
```

`build_training_components` selects `RustVectorEnv` or `VectorEnv` based on
flag and threads `hypers.experiment.seed` into either path. Both paths return
identical interface. Old path preserved for A/B comparison during validation.

## Alternatives considered

| Approach | Tradeoff | Why not |
|----------|----------|---------|
| Keep AsyncVectorEnv, batch PyO3 calls | Less invasive | Still pays subprocess spawn + IPC cost. The fundamental bottleneck is N processes, not N calls. |
| Rust VectorEnv + Rayon from the start | Maximum throughput immediately | Premature — adds GIL/thread complexity before we've validated the basic path works. Sprint 04 adds this. |
| Move observation encoding to Rust in same sprint | Fewer sprints total | Encoding parity is a separate, tricky validation problem. Shipping VectorEnv independently lets us validate the stepping contract before adding encoding. |
| Opponent policy callback from Rust to Python | Supports learned opponent policies | Adds complexity and cross-boundary calls. Passive/random cover training needs. Learned opponents are for simulation, not training hot path. |

## Key decisions

**Opponent auto-play lives in Rust.** The current `SingleAgentEnv` calls opponent
policies in Python, which would mean per-opponent-turn PyO3 round-trips inside
the batch step. Passive and random policies are trivial to implement in Rust
(~20 lines each). This is a meaningful speedup: in a typical game, the opponent
acts roughly as often as the agent, so this halves the effective round-trip count
even before considering auto-reset.

**Auto-reset is atomic in Rust.** When a game terminates, the Rust VectorEnv
resets it, plays through opponent turns, and returns the ready-to-act observation
in a single call. No extra Python round-trip. The terminal reward and flags are
preserved in the return tuple.

**Done semantics are simplified.** Rust returns one set of done flags
(`terminated`/`truncated`) and they represent terminal truth for the action just
taken, even when the returned observation is post-reset.

**Reward remains hero-perspective across opponent auto-steps.** If termination
occurs during opponent auto-play, the returned reward is negated to match
`SingleAgentEnv` behavior today.

**Default to new path.** `use_rust_env: bool = True`. The old path is preserved
behind `use_rust_env=False` for parity testing, not as a safety net. We ship
with confidence.

**Per-env seeding.** Episode-0 seed for env `i` is `base_seed + i`. On each
auto-reset for that env, increment by `num_envs` (`base_seed + i + k*num_envs`
for episode `k`) to avoid collisions while staying deterministic and
reproducible for any `(base_seed, num_envs)` pair.

**No new Python file for opponent policies.** The Rust `OpponentPolicy` enum
replaces `PassivePolicy`/`RandomPolicy` for the training path. The Python
policies in `single_agent_env.py` remain for simulation use.

## Scope

### In scope

- `managym/src/agent/vector_env.rs` — Rust VectorEnv struct
- `managym/src/agent/opponent.rs` — Passive/Random opponent policies in Rust
- `managym/src/python/vector_env_bindings.rs` — PyO3 bindings for VectorEnv
- `manabot/env/rust_vector_env.py` — Python wrapper with trainer-compatible interface
- `manabot/model/train.py` — `use_rust_env` flag, constructor branch
- `manabot/infra/hypers.py` — `use_rust_env` field on `TrainHypers`
- `tests/env/test_rust_vector_env.py` — Unit tests for new path
- Opponent-turn terminal reward sign parity test (`SingleAgentEnv` vs Rust path)
- Observation parity spot-check (100 random states, compare old vs new encoding)
- Verification ladder passes with new env

### Out of scope (wave says "not here")

- Game engine optimizations
- New card implementations
- Decoupled actor/learner, multi-GPU
- Rust-side observation encoding (sprint 02)
- Zero-copy buffers (sprint 03)
- Rayon parallelism (sprint 04)
- Learned opponent policies in Rust

## Implementation plan

### 1. Rust VectorEnv + opponent policies (~400 LOC)

- `managym/src/agent/vector_env.rs`: VectorEnv struct, `new`, `reset_all`, `step`
- `managym/src/agent/opponent.rs`: OpponentPolicy enum, `select_action` for Passive/Random
- Wire into `managym/src/agent/mod.rs`
- Rust unit tests: step N envs, verify auto-reset, verify opponent auto-play

### 2. PyO3 bindings (~150 LOC)

- `managym/src/python/vector_env_bindings.rs`: PyVectorEnv class
- Register in `managym/src/python/mod.rs`
- Smoke test from Python: create, reset, step

### 3. Python wrapper + trainer integration (~200 LOC)

- `manabot/env/rust_vector_env.py`: RustVectorEnv class
- `manabot/env/__init__.py`: export RustVectorEnv
- `manabot/infra/hypers.py`: add `use_rust_env` to TrainHypers
- `manabot/model/train.py`: branch on `use_rust_env`

### 4. Tests + verification (~200 LOC)

- `tests/env/test_rust_vector_env.py`: tensor shapes, dtype, device, auto-reset semantics
- Observation parity test: run 100 games through both paths, compare encoded output
- Run verification ladder (steps 0-1) with `use_rust_env=True`

Total: ~950 LOC, single PR.

## Done when

```bash
# Rust checks pass
cd managym && cargo fmt --check && cargo clippy --all-targets --all-features -- -D warnings && cargo test

# Python tests pass
pytest tests/env/test_rust_vector_env.py -v

# Verification ladder passes with new env
pytest tests/verify/ -v --use-rust-env

# Observation parity: old path vs new path produce identical encoded tensors
pytest tests/env/test_rust_vector_env.py::test_observation_parity -v

# Training runs and produces learning curves within noise of old path
manabot train --preset simple --set train.use_rust_env=true
```

SPS improvement is expected (eliminating subprocess overhead) but the exact
number is measured in sprint 05. The bar here is correctness, not speed.
