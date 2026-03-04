# Verification Ladder

## Problem

The manabot training platform has never shown clean convergence. Before
investing in hyperparameter sweeps, attention tuning, or auxiliary
heads, we need proof that each layer of the system works in isolation.
Without this, a failing training run produces no diagnostic signal —
is the environment broken? The reward misspecified? The optimizer stuck?

The verification ladder answers these questions layer by layer, from
"does the environment even work?" up to "can the agent learn to win?"

Wave goal 4: *Build a verification ladder: trivial reward →
memorization → beat passive → beat random.*

## Approach

Five standalone scripts in `manabot/verify/`, each runnable as
`python -m manabot.verify.stepN_name`. Each step isolates one failure
mode, prints PASS/FAIL with the relevant metric, and optionally logs
to wandb. All steps use `attention_on=False`.

The scripts reuse the existing `Trainer`, `VectorEnv`,
`SingleAgentEnv`, and `Experiment` infrastructure rather than building
a parallel training loop. Each step constructs a `Hypers` dataclass
programmatically (no Hydra config files — these are verification
scripts, not training configs) and calls into the existing machinery.

In the same milestone, strip Hydra from manabot runtime config:
Typer CLI + Pydantic schemas + Python dict presets. This keeps one
configuration model across verification, training, and simulation.

### Step 0: Environment sanity (no training)

Pure simulation. Run 1000 games with `RandomPolicy` vs `RandomPolicy`
via the existing `Env` (not `SingleAgentEnv` — both players are
external). Verify ~50% win rate. Then run 1000 games with
`RandomPolicy` vs `PassivePolicy`, verify Random wins >95%.

This step also monitors all truncation counters. If any truncation
fires during 1000 games, the padding limits are too tight — bump them
before proceeding.

Implementation: Use `Env` directly with `auto_reset=True`, step both
players manually (like `sim.py:_simulate_game`), track outcomes with a
simple counter. No `Trainer`, no `Agent`, no torch.

**Pass:** Random vs Random 40-60%. Random vs Passive >95%. Zero
truncation events.

### Step 1: Trivial reward (optimizer sanity)

Set `reward.trivial=True` (+1.0 every step). Train for 500 updates
with `num_envs=4`, `num_steps=128`. The value function should converge
to `1 / (1 - gamma)` = 100 (gamma=0.99).

This isolates the PPO optimization loop from the game. If explained
variance doesn't rise, there's a bug in training independent of MTG.

Implementation: Construct `Hypers` with `reward.trivial=True`,
`agent.attention_on=False`, `total_timesteps = 4 * 128 * 500 =
256_000`. Create `Experiment(wandb=False)`, build `Agent`, `VectorEnv`
(with passive opponent — doesn't matter since reward is trivial), and
`Trainer`. Call `trainer.train()`. After training, evaluate the
final value predictions on a batch of observations and check
explained variance from the last logged value.

**Pass:** Explained variance > 0.8 after 500 updates.

### Step 2: Memorization (single trajectory)

`num_envs=1`, fixed seed, tiny deck (20 Mountains each), passive
opponent. The game is nearly deterministic — same sequence of plays
each time. Train until the agent memorizes it.

Implementation: `Hypers` with `num_envs=1`, `experiment.seed=42`,
`match.hero_deck={"Mountain": 20}`, `match.villain_deck={"Mountain":
20}`, `total_timesteps=1_000_000`, passive opponent.

After training, run evaluation games with the trained agent and
measure win rate.

**Pass:** Wilson 95% confidence lower bound for win rate > 95% after
1M steps (`num_games=200`).

### Step 3: Beat passive (generalization)

Full setup: `num_envs=16`, Mountains + Grey Ogres deck, passive
opponent. The agent should learn to play lands, cast creatures, and
attack.

Implementation: `Hypers` with `num_envs=16`, standard deck, passive
opponent, `total_timesteps=5_000_000`. Train with periodic checkpoint
evaluation at 1M, 3M, and 5M timesteps (`num_games=200` each).

Track additional metrics at each checkpoint:
- Mean episode length
- Attack action frequency (hero ATTACK actions / hero actions)
- Explained variance (`trainer.last_explained_variance`)

**Pass:** Wilson 95% confidence lower bound for win rate > 90% at 5M,
mean episode length at 5M is at least 10% lower than at 1M, attack
action frequency at 5M is at least 10% higher than at 1M, explained
variance > 0.5.

### Step 4: Beat random (stretch)

Same as step 3 but `opponent_policy="random"`,
`total_timesteps=10_000_000`. Harder because random occasionally plays
creatures and blocks.

**Pass:** Wilson 95% confidence lower bound for win rate > 60% after
10M steps (`num_games=200`).

## Shared infrastructure

### `manabot/verify/__init__.py`

Empty — just makes it a package.

### `manabot/verify/util.py`

Shared helpers used across steps:

```python
def build_hypers(**overrides) -> Hypers:
    """Build Hypers with verification defaults and overrides."""

def run_evaluation(agent, obs_space, match, reward, num_games=100) -> dict:
    """Run evaluation games, return {win_rate, win_ci_lower, mean_steps, attack_rate, ...}."""

def print_result(step_name: str, passed: bool, metrics: dict):
    """Print PASS/FAIL with metrics."""
```

`run_evaluation` creates a single `Env` (no vectorization), loads the
agent in eval mode, plays `num_games` games, returns stats. This
avoids pulling in the `sim.py` machinery which has wandb dependencies
and multi-threading complexity.

### Metric collection during training

Steps 1-4 need to read metrics from training. The `Trainer` already
logs explained variance and other stats to wandb. For verification
scripts (which run with `wandb=False`), we need the metrics to be
accessible programmatically.

Approach: After `trainer.train()` completes, the last-computed
explained variance is available as `trainer.last_explained_variance`
(a new attribute set in the training loop). This is a single float
stored at the end of each update — minimal change.

For episode-level stats (win rate, CI, episode length, attack rate),
run `run_evaluation` at fixed checkpoints.

For step 3 trend checks, use fixed checkpoint evaluations at 1M, 3M,
5M timesteps. This avoids adding new trainer-side metric APIs and keeps
pass/fail logic deterministic and script-local.

## Alternatives considered

| Approach | Tradeoff | Why not |
|----------|----------|---------|
| Hydra configs per step | Reusable, composable | Verification scripts are one-off with very specific requirements; Hydra adds ceremony without benefit. Programmatic `Hypers` construction is more explicit and self-documenting. |
| Reuse `sim.py` for evaluation | Already built | Pulls in wandb/threading/model-loading complexity. Verification needs a 30-line evaluation loop, not a 700-line simulation framework. |
| Single monolithic verification script | Simpler to maintain | Can't run steps independently, can't iterate on one step without rerunning earlier ones. |
| Test framework (pytest) | Familiar | Steps 2-4 take 10-30 minutes. They're not unit tests — they're training runs with pass/fail criteria. pytest timeouts and reporting don't fit well. |
| Trainer rolling trend metrics | Cleaner long-term API | Adds more trainer instrumentation complexity now. Checkpoint evals are simpler and enough for verification gating. |

## Key decisions

**No Hydra, no wandb for verification.** These are diagnostic scripts.
They construct `Hypers` programmatically and print PASS/FAIL. Wandb
support can be added later as an opt-in flag if useful, but it's not
required.

**Evaluation uses raw `Env`, not `SingleAgentEnv`.** The evaluation
loop in `util.py` steps both players explicitly (trained agent as hero,
passive/random as villain), matching the `sim.py` pattern. This avoids
`SingleAgentEnv`'s auto-stepping which could mask bugs in the trained
agent's behavior.

**`Trainer.last_explained_variance` as the only training API change.**
One new attribute. No callback system, no metrics registry. The
verification scripts read it after `train()` returns.

**Win-rate gating uses confidence bounds, not raw point estimates.**
Each pass/fail win-rate check uses Wilson 95% lower confidence bound
with `num_games=200` to reduce false passes/fails from sampling noise.

**Step 3 trends use fixed checkpoint evaluations.** Trend checks are
computed from evaluation metrics at 1M and 5M timesteps (with 3M as an
intermediate sanity point), avoiding new trainer callback APIs.

**Step 0 uses `Env` directly, not `SingleAgentEnv`.** Both players are
external (RandomPolicy or PassivePolicy). This validates the raw
environment without the single-agent wrapper. Also validates truncation
limits against actual game statistics.

**All steps run on CPU.** `device="cpu"` in all configs. GPU is not
required and would make the scripts less portable.

**~1000 LOC total across all files.** Each step script is ~100-150
lines. `util.py` is ~150 lines. Compact and readable.

## Scope

**In scope:**
- `manabot/verify/__init__.py`
- `manabot/verify/util.py`
- `manabot/verify/step0_env_sanity.py`
- `manabot/verify/step1_trivial_reward.py`
- `manabot/verify/step2_memorization.py`
- `manabot/verify/step3_beat_passive.py`
- `manabot/verify/step4_beat_random.py`
- One new attribute on `Trainer` (`last_explained_variance`)
- Unit tests for `util.py` helpers (in `tests/verify/`)
- `manabot/config/schema.py` (Pydantic train/sim schemas)
- `manabot/config/presets.py` (Python dict presets: local/simple/attention/sim)
- `manabot/config/load.py` (deep-merge + `--set key.path=value` parsing)
- `manabot/cli.py` (Typer commands for train/sim)
- `manabot/model/train.py` and `manabot/sim/sim.py` entrypoint migration to Typer/Pydantic
- Dependency cleanup: remove `hydra-core`/`omegaconf`, add `typer`/`pydantic`

**Out of scope:**
- Wandb integration in verification scripts
- Attention mechanism (all steps use `attention_on=False`)
- Auxiliary heads (wave item 05 — comes after verification ladder)
- Rust engine changes
- CI wiring for verification steps (deferred; run manually this wave)

## Done when

```bash
python -m manabot.verify.step0_env_sanity   # PASS (< 2 min)
python -m manabot.verify.step1_trivial_reward  # PASS (< 5 min)
python -m manabot.verify.step2_memorization   # PASS (< 15 min)
python -m manabot.verify.step3_beat_passive   # PASS (< 30 min)
python -m manabot.verify.step4_beat_random    # Results documented
```

Steps 0-3 pass. Step 4 is attempted and results documented even if it
doesn't pass — knowing how close we get is valuable signal.

Hydra is removed from runtime entrypoints:
- `manabot train --preset simple --set train.total_timesteps=128`
- `manabot sim --preset sim --set sim.num_games=10`
- existing `python manabot/model/train.py` and `python manabot/sim/sim.py`
  continue to work as thin wrappers (no Hydra import/dependency)

Wave goals advanced: **Goal 4** (verification ladder) fully addressed.
Provides foundation for **Goal 5** (auxiliary heads) by establishing
baseline metrics.
