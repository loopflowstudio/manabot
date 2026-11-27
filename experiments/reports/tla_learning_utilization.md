# Learning utilization per collected transition on TLA trophy `w160l8`

Exploratory. Screen three single-factor increases to optimizer transition
utilization (`advantage_quantile=0.50`, `advantage_quantile=0.0` / no filtering,
and `epochs=2` under damped-KL) against the incumbent `tla_trophies_colab`
`w160l8` recipe across two paired training seeds to see whether higher
utilization improves held-out TLA trophy playing strength per collected
transition, per active wall-clock hour, and per USD without destabilizing
training.

## Verified baseline accounting

Inspection of `experiments/tla_trophies_colab.py`, `experiments/tla_trophies.py`,
`manabot/training/loops/self_play.py`, and `manabot/training/updates/targets.py`
confirmed two codebase facts before designing arms:

1. **Both seats are already trained in `control`.** `SelfPlay.learning_seats`
   defaults to `"both"`, and `tla_trophies.py` does not override it. In
   `tla_trophies_colab_500k_v2`, `w160l8@40902` collected 95.49M learner
   transitions across 2,000 updates (~47.7k transitions/update from ~257
   complete games/update ≈ 186 decisions/game across both seats). No separate
   both-seats arm is needed.
2. **The 25% utilization comes from `advantage_quantile=0.75` +
   `filter_scope="actor_critic"` + single-epoch updates.**
   `DampedPolicyLearning` retains the top `1 - advantage_quantile = 25%` of
   `|advantage|` rows (~12.5k / 50k) for both actor and critic and makes one
   pass (`width = streams = 256`, ~64 retained rows per Adam step). Adding an
   optional `epochs` field (default `1`, excluded when `1`) to
   `DampedPolicyLearning` and `update_damped_iteration` enables multi-epoch
   damped-KL passes guarded by the existing collection-time `behavior_logs`,
   `collection_kl=0.1`, and `clip=0.2`.

## Design

- **Module:** [`experiments/tla_learning_utilization.py`](../tla_learning_utilization.py).
- **Base recipe:** `tla_trophies_colab` `w160l8` (`d=160`, 8 layers, 8 heads,
  6.03M params, `focus_film=True`, `stream_norm=True`), complete 7,859-pool TLA
  trophy training split with closed decklists, policy input v11,
  `learning_seats="both"`, `streams=256`, `transitions=64`, `rows=512`,
  `workers=4`, `threads=2`, float32 CUDA, bucketed shapes, live tokens, raw
  checkpoints, `tau_scale=0.1`, `IterationSchedule(steps=2000, reference_steps=10000)`.
- **Horizon & checkpoints:** `400` updates (~103k games, ~19M collected learner
  transitions per run), checkpointing every `50` updates (`u0..u400`, ~1.2–1.8h
  spacing for tight equal-time matching). Keeping `schedule_steps=2000` matches
  `tla_trophies_colab_500k_v2`'s LR/tau curve over `u0..u400` and allows any
  winning run to resume directly to `u2000` without changing its schedule.
- **Seeds:** two independent paired seeds per arm (`40901`, `40902`; `40902`
  matches `tla_trophies_colab_500k_v2`'s `w160l8@40902`).

| Arm | Change from `control` | Retained rows / update | Optimizer steps / update | Exposures / collected row | Mechanism |
| --- | --- | ---: | ---: | ---: | --- |
| `control` | None (`q=0.75, min=0.01, epochs=1`) | ~25% (~12.5k) | ~188 | 0.25× | Incumbent baseline |
| `keep-50` | `advantage_quantile=0.50` | ~50% (~25k) | ~188 | 0.50× | 2× breadth (larger effective minibatch ~128 rows/step) |
| `keep-all` | `advantage_quantile=0.0, min_advantage=0.0` | 100% (~50k) | ~188 | 1.00× | Remove advantage filtering (4× breadth, ~256 rows/step) |
| `epochs-2` | `epochs=2` | ~25% (~12.5k) | ~376 | 0.50× | 2× depth (2 damped-KL passes over top-25% rows) |

- **Evaluation (paired held-out TLA trophy deals, 873 disjoint held-out pools,
  `matchup_mode="cross"`, 4 games/deal):**
  - Against the first-64 heuristic: 32 deals (128g/seed) at `u0..u400` plus 128
    extra deals at `u400` (640g/seed = 1,280g/arm).
  - Against itself at half updates (`u100, u200, u300, u400`): 32 deals.
  - Head-to-head vs `control` at equal updates (`u100, u200, u300, u400`): 32
    deals across all seed pairs (512g/arm per point) plus 128 extra deals at
    `u400` (2,048g/arm total).
  - Head-to-head vs `control` at equal active training time (`training_seconds`):
    for each seed, pairs each treated arm's nearest checkpoint in
    `training_seconds` against same-seed `control` at `u100, u200, u300, u400`
    within overlapping active time (32 deals per milestone, plus 128 extra deals
    = 640g/seed at the final overlapping equal-time point).
- **Diagnostics to inspect:** `time/collection`, `time/learning`,
  `collection_kl`, `reference_kl`, `numerical/clip_fraction`,
  `numerical/gradient_norm_before`, `value_scale/explained_variance`,
  `combat/attack/gap`, `combat/block/gap`, and `combat/*/saturated`.
- **Assumptions & confounds:**
  - *Strongest confound:* two-slot Colab L4 packing (`colab-l4-s2`) introduces
    modest co-tenant wall-clock noise depending on which arm shares the GPU;
    training all 8 runs as 4 balanced two-slot pairs and checking per-update
    `time/collection` vs `time/learning` isolates GPU vs host time.
  - *Horizon assumption:* 400 updates tests early-to-mid learning acceleration
    (where `w160l8` climbed from `10.9%` to `79.7%` vs heuristic in
    `tla_trophies_colab_500k_v2`), not `u2000` asymptotics. Any mechanism that
    clearly beats `control` at equal active training time and USD without KL or
    combat-lock degradation earns a follow-up finding run (or `u2000` extension)
    before changing the default recipe; `filter_scope="actor"` (critic-only
    100% coverage) is deferred unless `keep-all` improves critic calibration
    while hurting actor play.
- **Cost cap:** ≤ **$25** total (8 runs × `$3.00` per-run ceiling = `$24.00`
  max; expected actual spend ~`$11–$15` across 4 `colab-l4-s2` sessions at
  `$0.194/hr` with held-out evaluation on owned `mini-r2` at `$0.00`).
