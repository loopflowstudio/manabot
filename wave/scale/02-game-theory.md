# 02: Game-Theoretic Training

## Finish line

Training incorporates imperfect-information game theory. The agent
converges toward unexploitable play rather than cycling through
exploitable strategies.

## Changes

### 1. NFSP integration (Neural Fictitious Self-Play)

NFSP maintains two networks:
- **Best response network**: trained via PPO (our existing agent) to
  beat the current opponent mixture
- **Average policy network**: trained via supervised learning on the
  agent's own action history. Represents the agent's "average" strategy
  over training.

The opponent plays the average policy (with some probability) rather
than always the current best response. This drives convergence toward
Nash equilibrium in two-player zero-sum games.

```python
class NFSPAgent:
    def __init__(self):
        self.best_response = Agent(...)  # PPO-trained
        self.average_policy = Agent(...)  # Supervised on action history
        self.reservoir = ReservoirBuffer(max_size=1_000_000)
        self.eta = 0.1  # anticipatory parameter

    def act(self, obs):
        if random() < self.eta:
            return self.best_response.act(obs)
        else:
            return self.average_policy.act(obs)
```

### 2. Exploitability estimation

Approximate exploitability by training a fresh best-response agent
against the current policy for a fixed number of steps, then measuring
win rate. If the best-response wins >60%, the current policy is
exploitable.

This is expensive (requires training a new agent), so run it
periodically (every 10M steps) as a health check, not continuously.

### 3. Policy averaging (lightweight alternative to NFSP)

If full NFSP is too complex, simpler alternative: maintain an
exponential moving average of the policy weights. Use the EMA policy
as the opponent in some fraction of training games.

```python
ema_policy = copy.deepcopy(agent)
for p_ema, p in zip(ema_policy.parameters(), agent.parameters()):
    p_ema.data = 0.99 * p_ema.data + 0.01 * p.data
```

This is essentially Polyak averaging applied to the opponent, and it
provides some of NFSP's stability benefits with much less code.

## Done when

- Strategy cycling eliminated (cross-play matrix shows clear ordering)
- Exploitability estimate decreases over training
- Win rate vs reference opponents remains monotonically improving
