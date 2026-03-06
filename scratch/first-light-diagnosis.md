# First Light: Policy Diagnosis

2026-03-06. Branch: `jack-heart.first-light.20260305_1906`

## Summary

Training against random opponent makes the agent **worse than an untrained policy**.
The root cause is that PPO learns to stop playing lands, which collapses the entire
gameplay chain: no lands -> no mana -> no creatures -> no attacks -> no wins.

## The Causal Chain

```
play land -> tap for mana -> cast creature -> declare attacker -> deal damage -> win
```

The agent's decision on "play land vs pass priority" is the critical bottleneck.
Everything downstream depends on it.

## Key Finding: attacked_when_able is Vacuous

During declare attackers, the engine only offers DECLARE_ATTACKER actions — there is
no "pass" or "done declaring" option. The agent is forced to attack with every creature.
`attacked_when_able = 100%` at all checkpoints, including untrained. This metric is
not informative.

## Untrained Baseline (200 games, stochastic eval vs random)

```
python3 -c "
from manabot.env import Match, ObservationSpace, Reward
from manabot.infra import Hypers
from manabot.model.agent import Agent
from manabot.verify.util import STANDARD_DECK, run_evaluation

hypers = Hypers(
    match={'hero_deck': STANDARD_DECK, 'villain_deck': STANDARD_DECK},
    train={'opponent_policy': 'random'},
    agent={'attention_on': False},
)
obs_space = ObservationSpace(hypers.observation)
match = Match(hypers.match)
reward = Reward(hypers.reward)
agent = Agent(obs_space, hypers.agent)
agent.eval()

m = run_evaluation(agent, obs_space, match, reward, num_games=200,
    opponent_policy='random', deterministic=False, seed=42)
for k in sorted(m):
    v = m[k]
    print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')
"
```

Results:
```
  win_rate:          0.7200
  landed_when_able:  0.4403
  cast_when_able:    0.5639
  attacked_when_able:1.0000
  attack_rate:       0.3898
  mean_steps:        198.24
  could_land:        4545
  could_spell:       6459
  could_attack:      8162
  win_ci_lower:      0.6541
```

An untrained stochastic policy wins 72% against random — just by playing lands ~44%
of the time and casting ~56% of the time.

## Initial Logit Analysis

```
python scripts/diagnose_initial_policy.py
```

At initialization, logits for all actions are near-zero with tiny differences:

```
Valid actions (with raw logits):
  [0] priority_play_land        logit=-0.0005  features=[1. 0. 0. 0. 0. 0.]
  [1] priority_play_land        logit=-0.0005  features=[1. 0. 0. 0. 0. 0.]
  [2] priority_play_land        logit=-0.0005  features=[1. 0. 0. 0. 0. 0.]
  [3] priority_play_land        logit=-0.0005  features=[1. 0. 0. 0. 0. 0.]
  [4] priority_pass_priority    logit=-0.0003  features=[0. 0. 1. 0. 0. 0.]
```

- Deterministic (argmax): always picks pass (logit -0.0003 > -0.0005). This is
  why all previous deterministic evals showed 0% for everything.
- Stochastic: samples roughly proportional — 13/20 land, 7/20 pass in a sample.
  The 4 land actions collectively get ~80% probability mass.

## Training Curve: 5M steps vs random (50-game evals, stochastic)

```
step       win%    land%   cast%   attack%  steps
─────────────────────────────────────────────────────
baseline   72.0    44.0    56.4    100.0    198.2
  205k      8.0    20.9    93.1    100.0    157.7
  410k      0.0    10.2    98.9    100.0    128.1
  614k     70.0    17.8    96.5    100.0    205.0
  819k      2.0    23.7    95.4    100.0    199.6
 1024k     22.0    18.9    91.6    100.0    198.1
 1229k     18.0    18.0    97.7    100.0    180.0
 1434k     84.0    20.1    98.9    100.0    130.8
 1638k      8.0    18.3    98.6    100.0    126.2
 1843k     42.0    22.8    95.0    100.0    192.5
 2048k     42.0    17.9    97.4    100.0    197.9
 2253k     50.0    17.7    98.7    100.0    203.1
 2458k     38.0    12.4    97.4    100.0    160.6
 2662k     16.0    12.6    98.3    100.0    157.3
 2867k     54.0    18.2    97.6    100.0    195.2
 3072k     42.0    17.6    98.8    100.0    244.6
 3277k      6.0    13.0    99.0    100.0     93.2
 3482k     28.0    13.6    98.8    100.0    181.7
 3686k     56.0    19.8    92.5    100.0    151.9
 3891k      8.0    21.9    94.3    100.0    147.6
 4096k     10.0    17.3    91.6    100.0    140.6
 4301k     52.0    16.5    97.2    100.0    140.6
 4506k     12.0    17.3    97.6    100.0    147.8
 4710k     58.0    19.3    97.1    100.0    185.8
 4915k      8.0    16.3    98.3    100.0    169.9
```

Column key:
- `land%` = landed_when_able (played a land when one was available)
- `cast%` = cast_when_able (cast a spell when mana was available)
- `attack%` = attacked_when_able (always 100% — forced by engine)

## Final eval (200 games after 5M training steps)

```
  win_rate:           0.1800
  landed_when_able:   0.2012
  cast_when_able:     0.9636
  attacked_when_able: 1.0000
  explained_variance: 0.4345
  mean_steps:         200.55
  could_land:         11444
  could_spell:        3299
  could_attack:       3794
  win_ci_lower:       0.1329
```

## Diagnosis

1. **Training halves land play rate.** Baseline is 44%, trained is ~18-20%.
   The agent actively learns to pass instead of playing lands.

2. **Win rate noise masks the regression.** 50-game evals swing 0%-84%.
   The 200-game final eval shows 18% — far below the 72% untrained baseline.

3. **cast_when_able is misleading.** It rises from 56% to ~97%, but
   `could_spell` drops from 6459 to 3299. The agent casts when it can,
   but it rarely can because it doesn't play lands.

4. **Credit assignment failure.** Terminal-only reward (+1 win, -1 loss)
   over ~200 steps cannot assign credit to "play land on turn 1."
   PPO's advantage estimates are too noisy to distinguish "play land"
   from "pass priority" — both appear in winning and losing trajectories.

5. **Deterministic eval was hiding everything.** Prior runs showed 0%
   across the board because argmax always picks pass (see logit analysis).
   Switching to stochastic eval revealed the actual policy dynamics.

## Reward Structure

```python
# manabot/env/match.py:86-101
class Reward:
    def compute(self, raw_reward, _last_obs, new_obs):
        if not new_obs.game_over:
            return raw_reward   # 0 for all intermediate steps
        return win_reward if new_obs.won else lose_reward  # +1 / -1
```

Pure terminal reward. No intermediate signal for land plays, casting, or board state.

## Possible Directions

1. **Reward shaping** — intermediate rewards for playing lands, casting creatures,
   maintaining board advantage. Directly addresses the credit assignment gap.

2. **Entropy bonus tuning** — current ent_coef=0.01 may not sustain enough
   exploration. Higher entropy could prevent the policy from collapsing to "always pass."

3. **Curriculum** — train against passive first (where passing still wins by decking),
   then graduate to random. The passive opponent provides a gentler gradient.

4. **Action space redesign** — 4 identical "play land" actions split probability
   mass. Aggregating by type or making land features distinguishable could help.

## Changes Made This Session

Files modified:
- `manabot/infra/metrics.py` — added `land_available`, `spell_available` to actions table
- `manabot/verify/util.py` — added `_land_available()`, `_spell_available()`, `_is_land_action()`,
  `_is_spell_action()` helpers; `run_evaluation()` now returns `landed_when_able`, `cast_when_able`,
  `could_land`, `could_spell`, `land_plays`, `spell_casts`; default `deterministic=False`
- `manabot/verify/step2_memorization.py` — `deterministic=False`
- `manabot/verify/step3_beat_passive.py` — `deterministic=False`
- `manabot/verify/step4_beat_random.py` — `deterministic=False`
- `manabot/model/train.py` — periodic eval logs `landed_when_able`, `cast_when_able`; `deterministic=False`
- `scripts/diagnose_initial_policy.py` — diagnostic script for initial policy behavior

## Raw Logs

### Periodic eval log lines
```
2026-03-06 09:02:10 - Eval @ update 100 (step 204800): win_rate=8.00%, landed_when_able=20.94%, cast_when_able=93.10%, attacked_when_able=100.00%, mean_steps=157.7
2026-03-06 09:03:16 - Eval @ update 200 (step 409600): win_rate=0.00%, landed_when_able=10.23%, cast_when_able=98.90%, attacked_when_able=100.00%, mean_steps=128.1
2026-03-06 09:04:24 - Eval @ update 300 (step 614400): win_rate=70.00%, landed_when_able=17.80%, cast_when_able=96.49%, attacked_when_able=100.00%, mean_steps=205.0
2026-03-06 09:05:30 - Eval @ update 400 (step 819200): win_rate=2.00%, landed_when_able=23.66%, cast_when_able=95.43%, attacked_when_able=100.00%, mean_steps=199.6
2026-03-06 09:06:35 - Eval @ update 500 (step 1024000): win_rate=22.00%, landed_when_able=18.92%, cast_when_able=91.56%, attacked_when_able=100.00%, mean_steps=198.1
2026-03-06 09:07:41 - Eval @ update 600 (step 1228800): win_rate=18.00%, landed_when_able=18.04%, cast_when_able=97.68%, attacked_when_able=100.00%, mean_steps=180.0
2026-03-06 09:08:45 - Eval @ update 700 (step 1433600): win_rate=84.00%, landed_when_able=20.12%, cast_when_able=98.87%, attacked_when_able=100.00%, mean_steps=130.8
2026-03-06 09:09:50 - Eval @ update 800 (step 1638400): win_rate=8.00%, landed_when_able=18.25%, cast_when_able=98.64%, attacked_when_able=100.00%, mean_steps=126.2
2026-03-06 09:10:55 - Eval @ update 900 (step 1843200): win_rate=42.00%, landed_when_able=22.79%, cast_when_able=95.03%, attacked_when_able=100.00%, mean_steps=192.5
2026-03-06 09:12:01 - Eval @ update 1000 (step 2048000): win_rate=42.00%, landed_when_able=17.88%, cast_when_able=97.43%, attacked_when_able=100.00%, mean_steps=197.9
2026-03-06 09:13:05 - Eval @ update 1100 (step 2252800): win_rate=50.00%, landed_when_able=17.65%, cast_when_able=98.73%, attacked_when_able=100.00%, mean_steps=203.1
2026-03-06 09:14:09 - Eval @ update 1200 (step 2457600): win_rate=38.00%, landed_when_able=12.44%, cast_when_able=97.40%, attacked_when_able=100.00%, mean_steps=160.6
2026-03-06 09:15:12 - Eval @ update 1300 (step 2662400): win_rate=16.00%, landed_when_able=12.60%, cast_when_able=98.34%, attacked_when_able=100.00%, mean_steps=157.3
2026-03-06 09:16:17 - Eval @ update 1400 (step 2867200): win_rate=54.00%, landed_when_able=18.16%, cast_when_able=97.63%, attacked_when_able=100.00%, mean_steps=195.2
2026-03-06 09:17:22 - Eval @ update 1500 (step 3072000): win_rate=42.00%, landed_when_able=17.57%, cast_when_able=98.77%, attacked_when_able=100.00%, mean_steps=244.6
2026-03-06 09:18:24 - Eval @ update 1600 (step 3276800): win_rate=6.00%, landed_when_able=13.04%, cast_when_able=98.97%, attacked_when_able=100.00%, mean_steps=93.2
2026-03-06 09:19:28 - Eval @ update 1700 (step 3481600): win_rate=28.00%, landed_when_able=13.63%, cast_when_able=98.80%, attacked_when_able=100.00%, mean_steps=181.7
2026-03-06 09:20:32 - Eval @ update 1800 (step 3686400): win_rate=56.00%, landed_when_able=19.81%, cast_when_able=92.45%, attacked_when_able=100.00%, mean_steps=151.9
2026-03-06 09:21:36 - Eval @ update 1900 (step 3891200): win_rate=8.00%, landed_when_able=21.90%, cast_when_able=94.32%, attacked_when_able=100.00%, mean_steps=147.6
2026-03-06 09:22:39 - Eval @ update 2000 (step 4096000): win_rate=10.00%, landed_when_able=17.31%, cast_when_able=91.62%, attacked_when_able=100.00%, mean_steps=140.6
2026-03-06 09:23:43 - Eval @ update 2100 (step 4300800): win_rate=52.00%, landed_when_able=16.47%, cast_when_able=97.21%, attacked_when_able=100.00%, mean_steps=140.6
2026-03-06 09:24:46 - Eval @ update 2200 (step 4505600): win_rate=12.00%, landed_when_able=17.32%, cast_when_able=97.62%, attacked_when_able=100.00%, mean_steps=147.8
2026-03-06 09:25:50 - Eval @ update 2300 (step 4710400): win_rate=58.00%, landed_when_able=19.26%, cast_when_able=97.14%, attacked_when_able=100.00%, mean_steps=185.8
2026-03-06 09:26:54 - Eval @ update 2400 (step 4915200): win_rate=8.00%, landed_when_able=16.32%, cast_when_able=98.26%, attacked_when_able=100.00%, mean_steps=169.9
```
