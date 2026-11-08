# MTG-252 — critic readout comparison: implementation status (2026-11-05)

## Resumed conversation (2026-11-08)

The older notes below predate completed depth-2/update-300 and
depth-4/masked-mean/update-150 evaluations. The native Claude transcript is
`/Users/jack/.claude/projects/-Users-jack-src-manabot-mtg-252-compare-learned-value/4265433e-e023-42b4-93b4-1d05d7c08a66.jsonl`.

At 11:34:17 UTC the participant wrote: "continue the depth 4 runs, and add
depth 4 with a value token you get to to update 300?" At 11:35:16 UTC the
budget question's answer was "Raise cap to $45 (Recommended)" for three
seeds per arm. The turn was interrupted after partially editing
`experiments/value_readout_252.py`; neither new batch had launched.

Recovered result at update 150: depth-4 masked mean scored .512 versus
depth-2's .492 against the heuristic; paired difference +.020
[-.023, +.058]. Head-to-head depth-4 score .563 [.488, .658]. These are the
previous session's retained report, not newly recomputed results. Its
scripts/data are under
`/private/tmp/claude-501/-Users-jack-src-manabot-mtg-252-compare-learned-value/4265433e-e023-42b4-93b4-1d05d7c08a66/scratchpad/`.

Fresh Etude 0.2.15 reads confirm all three depth-4 parents succeeded at 150.
Whole-rental billing plus estimates: training USD31.7143, evaluations
USD0.9124 + 1.1054 + 1.0835 = 3.1013; total USD34.8156, remaining USD10.1844.
This supersedes the prior USD32.90 budgeting estimate. Run status cost omits
some rental time: the third evaluation shows USD.22 there but USD1.0835
on its machine. Raw machine inventory receipt: `/tmp/mtg252-machines-20261108.txt`.
The first depth-4 parent's learner record reports 8,862.50 seconds for 150
updates. Do not budget another 150 updates from the prior USD.97 estimate.

Continuation exports are being verified/staged under `/tmp/mtg252-depth-parents/`
and `r2://manabot/mtg252/continue/<parent>/`. Latest code completes the
depth/readout selection and evaluation path; no new training launched yet.

In this resumed conversation the budget reply was "Raise cap to $55; complete
both arms". Allocation: USD6 masked mean (3 × 3.1 hours, USD2 each), USD11.10
value token (3 × 6 hours, USD3.70 each), USD3 evaluation. Prior cost plus these
ceilings is USD54.9156. All three saved parent regimes differ from the proposed
continuations only in absolute updates (150 → 300) and the two watchdog fields.
Focused tests: 16 passed; Ruff lint and formatting passed. Setup is pushed at
`18e4ae9a8`; the subsequent documentation commit records the raised cap.

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
- Training launched 2026-11-05 21:10–21:17 UTC, all nine runs on RunPod A40
  (`mtg252-a40`, $0.59/h; L4 and Vast had no usable stock), one three-arm
  cohort per seed (mtg252-s<seed>-mtg252-a40-*), $4.05 cap each, $12.15 total,
  revision fe25e6e18. Runs: s25201 c296951dc634/c5a1a009338c/e2118d949d3a,
  s25202 d70f68c8b612/cf95f6d220f1/4c0fc018fc8b, s25203
  4782034c5274/1488ddda75ca/295d9d24de78 (masked/token/readout). Windows were
  sized from L4 pace; A40 pace unmeasured — check first updates' seconds.
