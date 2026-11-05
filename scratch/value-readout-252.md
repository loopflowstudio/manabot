# MTG-252 — critic readout comparison: implementation status (2026-11-05)

Brief: Task MTG-252 directive. Study document: `experiments/value-readout-252.md`
(draft protocol; nothing frozen or launched; no compute authorized).

## Done in this change
- `ModelConfig.value_aggregation` gains `attention_readout` (one learned query,
  PMA-style cross-attention over final valid objects → existing critic MLP).
  Built last in `ActorCritic.__init__`, so it shares every masked-mean draw.
  `ValueAggregation` now has one owner (`manabot.infra.config`).
- Witnesses in `tests/model/test_value_aggregation.py`: order, padding to 250,
  copy counts, singleton among 40 copies, seat swap, uniform replication
  (masked/readout exactly blind, token sees it), readout actor/init sharing.
- Diagnostics: `value_scale` per damped update; `gradient_norm_critic_readout`
  on sampled optimizer steps.
- `experiments/value_readout_252.py`: single-axis arm derivation from v10 regime
  and static costs (params, attention pairs, init sharing).

## Delete — do not maintain
- None. `builders.ValueAggregation` duplicate literal removed in favor of config's.

## Remaining (needs decisions before any execution)
- Arm count, seeds, update count, budget, thresholds (table in study doc).
- Learning rule depends on the open damped-learning review for the full-TLA
  successor (MTG-213 comment 81ec3f17); freezing on v10's rule may go stale.
- Protocol JSON with source/world/shape pins after landing; arm regimes committed then.
- Inference ms/decision and peak memory measured at execution, not here.

Check: `uv run --extra dev pytest tests/model/test_value_aggregation.py tests/training/test_value_readout_252.py tests/training/test_numerical_health.py -q` passed (75) after compress.
