# 04: Verification Ladder

Build the test suite that proves each layer of the system works before
adding complexity. Each step isolates a different failure mode. Never
move to the next step until the current one passes.

## Finish line

Four verification steps implemented as runnable scripts with clear
pass/fail criteria. Steps 0-3 all pass.

## Current state

**Code is written.** All five scripts exist in `manabot/verify/`
(`step0_env_sanity.py` through `step4_beat_random.py`) plus shared
helpers in `util.py` and unit tests in `tests/verify/test_util.py`.

**Config migration is done.** Hydra replaced with Typer CLI
(`manabot/cli.py`) + Python dict presets (`manabot/config/presets.py`)
+ deep-merge loader (`manabot/config/load.py`). Hypers classes in
`manabot/infra/hypers.py` are now Pydantic `BaseModel`s (no more
`schema.py` or `asdict()` indirection). `train.py` and `sim.py`
updated to use new config.

**Step 0 does not pass yet.** Running `python -m manabot.verify.step0_env_sanity`
fails on two fronts:
1. Heavy card/permanent truncation events (observation padding limits
   too tight for actual game states)
2. Random-vs-random hero win rate far from the expected 50%

This likely indicates either aggressive observation caps in
`ObservationSpaceHypers` (20 cards, 15 permanents per player) or
action-selection robustness issues when the action space is truncated.

**Steps 1-4 are blocked on step 0.** The ladder is sequential — no
point running training steps until the environment is validated.

## Known issues and assumptions

- Verification rollout/evaluation loops catch `PyAgentError` from
  `env.step(action)` and try action `0` once before treating the game
  as aborted. Assumption: continuing with a counted non-win is better
  than crashing the whole verification script.
- `build_hypers(match={...})` treats `hero_deck`/`villain_deck`
  overrides as full replacements (not deep merges) so step 2 can run
  Mountain-only decks.
- The win-rate asymmetry in random-vs-random may stem from the Rust
  engine's turn structure or from how `Env` maps player actions. Needs
  investigation.

## Remaining work

1. **Fix truncation:** Bump `ObservationSpaceHypers` card/permanent
   limits or investigate why truncation is so frequent with the test
   deck. The limits (20 cards, 15 permanents) were set from deck
   profiles but never stress-tested at scale.
2. **Fix win-rate symmetry:** Investigate why random-vs-random hero
   win rate deviates from 50%. Could be turn-order advantage, action
   space asymmetry, or a bug in the env wrapper.
3. **Get step 0 to pass**, then run steps 1-4 sequentially.
4. **Document step 4 results** even if it doesn't pass.

## Configuration

All steps use `attention_on=False`. The attention mechanism is a
complexity multiplier that can be re-enabled after basic learning is
proven.

## The ladder

### Step 0: Environment sanity

Run 1000 games: RandomPlayer vs RandomPlayer. Verify ~50% win rate
(within statistical bounds). Run 1000 games: RandomPlayer vs
PassivePlayer. Verify RandomPlayer wins ~100%.

This step also monitors all truncation counters. If any truncation
fires during 1000 games, the padding limits are too tight — bump them
before proceeding.

Implementation: Uses `Env` directly with `auto_reset=True`, steps both
players manually, tracks outcomes with a simple counter. No `Trainer`,
no `Agent`, no torch.

**Pass criteria:** Random vs Random win rate between 40-60%. Random vs
Passive win rate > 95%. Zero truncation events.

### Step 1: Trivial reward

Set reward to +1.0 every step. Train for 500 updates with `num_envs=4`,
`num_steps=128`. The value function should converge to
`1 / (1 - gamma)` = 100 (gamma=0.99).

**Pass criteria:** Explained variance > 0.8 after 500 updates.

### Step 2: Memorization

`num_envs=1`, fixed seed, tiny deck (20 Mountains each), passive
opponent. The game is nearly deterministic.

**Pass criteria:** Wilson 95% confidence lower bound for win rate >
95% after 1M steps (`num_games=200`).

### Step 3: Beat passive

Full setup: 16 envs, Mountains + Grey Ogres deck, passive opponent.
Fixed checkpoint evaluations at 1M, 3M, and 5M timesteps
(`num_games=200` each).

**Pass criteria:**
- Wilson 95% confidence lower bound for win rate > 90% after 5M steps
- Mean episode length at 5M is at least 10% lower than at 1M
- Attack action frequency at 5M is at least 10% higher than at 1M
- Explained variance > 0.5

### Step 4: Beat random (stretch goal)

Same setup but against RandomPlayer, `total_timesteps=10_000_000`.

**Pass criteria:** Wilson 95% confidence lower bound for win rate >
60% after 10M steps (`num_games=200`).

## Key decisions

- **Evaluation uses raw `Env`, not `SingleAgentEnv`.** Both players
  stepped explicitly to avoid masking bugs.
- **`Trainer.last_explained_variance`** is the only training API
  addition. One attribute, no callback system.
- **Win-rate gating uses Wilson 95% confidence bounds**, not raw point
  estimates. `num_games=200` per evaluation.
- **All steps run on CPU.** `device="cpu"` everywhere.
- **CI wiring deferred.** Scripts run manually this wave.

## Available instrumentation

Rollout health counters wired into trainer and wandb:
- `rollout/truncated_episodes` (+ `_total`)
- `rollout/action_space_truncations` (+ `_total`)
- `rollout/card_space_truncations` (+ `_total`)
- `rollout/permanent_space_truncations` (+ `_total`)

The env emits per-step `info` flags (`action_space_truncated`,
`card_space_truncated`, `permanent_space_truncated`) which
`SingleAgentEnv` OR-accumulates across opponent steps and the trainer
rolls up into the counters above.

## Done when

Steps 0-3 pass. Step 4 is attempted and results documented.

```bash
python -m manabot.verify.step0_env_sanity   # PASS (< 2 min)
python -m manabot.verify.step1_trivial_reward  # PASS (< 5 min)
python -m manabot.verify.step2_memorization   # PASS (< 15 min)
python -m manabot.verify.step3_beat_passive   # PASS (< 30 min)
python -m manabot.verify.step4_beat_random    # Results documented
```
