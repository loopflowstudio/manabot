# 04: Verification Ladder

Build the test suite that proves each layer of the system works before
adding complexity. Each step isolates a different failure mode. Never
move to the next step until the current one passes.

## Finish line

Four verification steps implemented as runnable scripts with clear
pass/fail criteria. Steps 0-3 all pass.

## Configuration

All steps use `attention_on=False`. The attention mechanism is a
complexity multiplier that can be re-enabled after basic learning is
proven.

## The ladder

### Step 0: Environment sanity

Run 1000 games: RandomPlayer vs RandomPlayer. Verify ~50% win rate
(within statistical bounds). Run 1000 games: RandomPlayer vs
PassivePlayer. Verify RandomPlayer wins ~100%.

This validates the environment is symmetric and the passive opponent
actually loses.

**Pass criteria:** Random vs Random win rate between 40-60%. Random vs
Passive win rate > 95%.

### Step 1: Trivial reward

Set reward to +1.0 every step (the `trivial` reward mode already exists).
Train for 500 updates (`attention_on=False`). The value function should
learn to predict `1 / (1 - gamma)` ≈ 100 (with gamma=0.99).

This isolates the optimization machinery from the game. If explained
variance doesn't rise, there's a bug in the training loop independent
of MTG.

**Pass criteria:** Explained variance > 0.8 after 500 updates.

### Step 2: Memorization

`num_envs=1`, fixed seed, tiny deck (20 Mountains for each player),
`attention_on=False`. The game plays out nearly identically each time.
Train against passive opponent.

The agent should achieve near-100% win rate because it can memorize the
single game trajectory.

**Pass criteria:** Win rate > 95% after 1M steps.

### Step 3: Beat passive

Full setup: 16 envs, Mountains + Grey Ogres deck, `attention_on=False`,
train against passive opponent. The agent should learn to play lands,
cast creatures, and attack.

**Pass criteria:**
- Win rate > 90% after 5M steps
- Mean episode length decreases over training
- Attack action frequency increases over training
- Explained variance > 0.5

### Step 4: Beat random (stretch goal)

Same setup but against RandomPlayer. Harder because random occasionally
plays creatures and blocks.

**Pass criteria:** Win rate > 60% after 10M steps.

## Implementation

Each step is a script in `manabot/verify/` that:
1. Configures the environment and training for that step
2. Runs training (or simulation for step 0)
3. Logs metrics to wandb (or stdout)
4. Prints PASS/FAIL with the relevant metric

```bash
python -m manabot.verify.step0_env_sanity
python -m manabot.verify.step1_trivial_reward
python -m manabot.verify.step2_memorization
python -m manabot.verify.step3_beat_passive
python -m manabot.verify.step4_beat_random
```

## Constraints

- Each step should be runnable on CPU (no GPU requirement)
- Steps 1-3 should complete in under 30 minutes on a laptop
- Step 4 may take longer and is a stretch goal for this wave

## Done when

Steps 0-3 pass. Step 4 is attempted and results are documented even
if it doesn't pass — knowing how close we get is valuable signal for
the next wave.
