# First Light: Policy Diagnosis — Reference

Historical analysis that motivated reward shaping. Key findings folded into
`wave/first-light/README.md` strategy section and item 05.

## Reproduce baseline numbers

```bash
# Untrained baseline: 200 games, stochastic eval vs random
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

Expected: ~72% win rate, ~44% landed_when_able, ~56% cast_when_able.

## Initial logit analysis

```bash
python scripts/diagnose_initial_policy.py
```

Expect near-zero logits for all actions, pass slightly higher than land.
Stochastic sampling gives ~80% land probability (4 land actions vs 1 pass).
