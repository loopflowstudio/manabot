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
- Diagnostics: `value_scale` per damped update; on sampled optimizer steps
  `gradient_norm_critic_readout` (critic-only: `value_head`, `value_readout`) and,
  in the token arm only, `gradient_norm_value_token` (shared actor+critic).
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
- Shared `masked_mean` baseline with sparse-attention Task690c9b: coordinate
  one baseline arm rather than crossing axes; not yet arranged.
- Ataraxos fidelity: primary paper not reread; the study doc claims none.

## Overlaps (rechecked 2026-11-05)
- Open PR424 (MTG-243) conflicts with this branch in `manabot/model/actor_critic.py`
  (its other conflicts exist against main already). Whichever lands second
  rebases; preserve its catalog naming/action identity.
- Open PR383 also edits `damped_policy.py`/`selection.py`; no textual conflict
  with this branch.

Check: `uv run --extra dev pytest tests/model/test_value_aggregation.py tests/training/test_value_readout_252.py tests/training/test_numerical_health.py -q` passed (77) after the critic-readout diagnostic fix.

## Concept review (2026-11-05)

Concept accepted: one axis, `agent.value_aggregation`; each arm names what
summary the critic MLP reads. Its one finding (`value_token` counted as
critic-only) is fixed above. Remaining choices are the execution items listed
under Remaining; they block freezing/launch, not this change. Checked: the
readout arm's actor path equals baseline at init and its params are critic-only.
