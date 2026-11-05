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

Second pass (2026-11-05, after `dc35969a8`): `_CRITIC_READOUT` excludes the
token, `gradient_norm_value_token` is reported separately, the study doc
matches. No concept, name or API change; nothing new to implement.

## Direction (Jack Heart, 2026-11-05)

- Base recipe: MTG-183 Elemental (`elemental-long-183-v1`, ten decks, uniform
  per-game pairings) re-pinned to the current world, not v10 TLA trophy.
  Applied: `value_readout_252.base()` re-pins world, resolves policy input v6
  with 183's capacities (60/40/64); 202 objects; costs table updated.
- Evaluation: (a) random Elemental deck pairings against the heuristic, plus
  head-to-head between the arms' final policies. Study doc protocol updated
  (mtg186_eval "rest"-style schedule; pinned SHA at freeze).
- Still open: arms/seeds/updates/budget/thresholds/schedule size; measure
  w17 cost per update before budgeting.
- Fairness (Jack Heart): pairing seed fixed across arms. Training pairings
  are a pure function of per-game native seed (run seed + 10000 + s + k·streams),
  so shared run seeds give identical per-stream pairing/deal sequences;
  `runs(seeds)` builds the arm × seed grid from one list; witness
  `test_pairings_depend_on_the_run_seed_not_the_arm`. Eval uses one fixed
  schedule seed for all arms/seeds.
- Budget (Jack Heart, 2026-11-05): 3 arms × 3 seeds × 150 updates, USD20 cap
  incl. evaluation. Training: cohort mtg252 on 07c490-l4, 2 h / USD1.35 per run.
  Evaluation (random-pair vs heuristic + head-to-head) still to be built; ~USD7 left.
- Launch (2026-11-05 20:39 UTC): quote $11.95 for 9 runs at fe25e6e18; first
  launch refused (RunPod reports no L4 stock anywhere; nothing rented). A
  background retry checks every 5 min for up to 3 h, same cohort/cap. Raised
  `07c490-l4` max 8 → 9 for this cohort; set it back to 8 once mtg252 is up.
- Evaluation: build on 07c490's unmerged `learning_signal` Design/Pairing
  (random non-mirror Elemental pairs vs heuristic, fixed pairing seed) once it
  lands; add a 252-specific design (own reserved deals) and a final-policy
  head-to-head matchup. Not needed until training finishes.
