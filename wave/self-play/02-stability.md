# 02: Self-Play Stability + Cycling Detection

## Finish line

Self-play training runs for 50M+ steps without strategy cycling.
Automatic detection and mitigation when cycling is detected.

## Changes

### 1. Cycling detection

Track win rate of current policy against each opponent in the pool
over a sliding window. If win rate against a previously-beaten opponent
drops below a threshold, flag it as potential cycling.

```python
# Cycling signal: win rate vs opponent X was >70%, now <50%
if rolling_win_rate[opponent] < 0.5 and peak_win_rate[opponent] > 0.7:
    log.warning(f"Potential cycling detected against {opponent}")
```

### 2. Anti-cycling mitigations

When cycling is detected:
- Increase opponent pool sampling weight toward the opponent being
  lost against (prioritized experience)
- Optionally add entropy bonus boost to encourage strategy diversity
- Consider freezing the current policy as a new pool member and
  reverting to the last "stable" checkpoint

### 3. Policy-mapping configuration

Explicit config for which policies are trainable and which are frozen.
Inspired by RLlib's `policies_to_train` concept:

```yaml
training:
  hero_policy: trainable
  villain_policy: pool  # or: frozen, random, passive
  pool_sample_strategy: uniform  # or: prioritized, recent
```

### 4. Cross-play evaluation matrix

Periodic evaluation: every checkpoint in the pool plays against every
other checkpoint. Produces a win-rate matrix logged to wandb. Reveals
whether the population is actually improving or just cycling through
equivalent strategies.

## Done when

- 50M steps of self-play training without detected cycling
- Cross-play matrix shows clear improvement ordering (newer > older)
- Elo monotonically increasing against reference set
