# Does BF16 learn the same as float32 at w384l8? — MTG-324

## Question and prediction (committed before any scored run)

`Execution.precision="auto"` trains a CUDA model at width 256 or above in BF16
(`BF16_MIN_HIDDEN_DIM`). MTG-317 and its w256l8 step run measured BF16 1.66×
faster than float32 at w256l8 and 2–2.7× at w384l8 on an L4 ([`cuda_precision.md`](cuda_precision.md)), but its replay could
not tell same from different learning. **Question:** trained from the same two
seeds on the frontier recipe at w384l8, does BF16 reach the same strength and
follow the same learning trajectory as float32 at matched updates?
Matched update *M* is the last update all four runs checkpointed: 300 on
schedule, earlier if a float32 run stops at its watchdog.
**Prediction:** at *M* the BF16 arm's mean score against the heuristic is
within 0.05 of the float32 arm's; BF16 scores between 0.45 and 0.55 in the
head-to-heads; and BF16's mean policy loss, value loss, clip fraction and
collection KL over the second half of updates 1–*M* each lie inside the float32 seeds' range
widened by that range's width on each side, with no non-finite, rejected or
skipped step. **Refuted** if BF16 scores more than 0.08 below float32 against the
heuristic at *M*, if BF16 scores under 0.42 in the
head-to-heads, if any of those four trajectory means falls outside the widened
float32 range, or if any BF16 numerical counter is nonzero. A result inside
neither the prediction nor the refutation is **unresolved**. **Decision:** if the
prediction holds, the ladder stays at 256; if it is refuted or unresolved,
`ladder_precision` returns float32 everywhere. **Cost cap:** USD 34 for the
experiment (training launched with `--max-usd 32`, evaluation capped at USD 2),
within the Task's USD 35 cap and the approved budget of about USD 30.

## Answer — 2026-11-13

**BF16 is rejected by the numerical gate.** Run `2be568a65518` (seed 32401)
completed 140 updates, then rejected attempted update 141 with
`behavior_log_probability_mismatch`. Its retained stage record says
`rejected_updates: 1`, `nonfinite_failure: 0`. The pre-registered rule rejects
any BF16 numerical rejection; no strength score can override that condition.
`Execution.precision="auto"` therefore resolves to **float32 at every width
and on every device**. The measured speed crossover at width 256 remains
recorded, but does not enable BF16. Explicit BF16 and TF32 remain available.

![Completed updates in the four scored training runs](data/precision_ladder/updates.png)

The other three runs paused at their time limits and exported normally (the
launcher reports those jobs as succeeded). The bars show where runs ended,
not a controlled throughput comparison: their durations and stopping reasons
differ.

| Precision | Seed | Etude run | Completed updates | Stop |
|---|---:|---|---:|---|
| float32 | 32401 | `07f84834e617` | 188 | time limit |
| float32 | 32402 | `aa9518f6fc4b` | 170 | time limit |
| bf16 | 32401 | `2be568a65518` | 140 | numerical rejection at 141 |
| bf16 | 32402 | `6aa5ba4d5fac` | 290 | time limit |

The last scheduled checkpoint retained by all four runs is **M = 75**
(checkpoints 20, 40 and 75 are common; the failed run never reached 150).
The paid strength evaluation was not launched after the terminal numerical
rejection made the gate decisive. There is **no measured strength difference**
and no claim that the four learning trajectories stayed within the float32
band. This result establishes failure of the admission rule, not the root
cause of the probability mismatch or a general BF16 learning deficit.

Evidence: each run's original `training/run.json`, the failed run's retained
traceback, the final launcher statuses and the chart are published under
`experiments/data/precision_ladder/` in `experiments/data.manifest.json`.
Restore them with:

```sh
uv run --extra artifacts python -m manabot.data fetch experiments/data/precision_ladder
```

## Cost

**USD 24.35**, summed from the launcher's four final run charges; evaluation
cost USD 0. All four rentals have dated termination confirmation. The earlier
w256 speed run and its startup losses are recorded separately in
[`cuda_precision.md`](cuda_precision.md#w256l8-step-run-mtg-324-2026-11-12).
The experiment stayed below its USD 34 cap and the Task's USD 35 cap.

## Limits

- Two seeds per arm; only BF16 seed 32401 hit the guard. Attribution of that
  failure requires a separate investigation.
- No arm completed the scheduled 300 updates, and no strength evaluation was
  performed. Long-run learning equivalence remains unknown.
- All four runs used bucketed shapes and **no live-token attention**. The
  later automatic live-token default does not change these frozen runs, and
  this A/B admits no BF16/live-token combination.
- The only learning width was 384. Width 256 has speed and fixed-weight
  measurements, not a learning comparison.

<details>
<summary>Design and launch history</summary>

## Design

`experiments/precision_ladder.py`. Two arms, `precision-ladder-w384l8-float32-v1`
and `precision-ladder-w384l8-bf16-v1`, each run under seeds 32401 and 32402 on
its own L4. Both arms are `manabot.recipes.builders.sampled_self_play` over the
ten-deck Elemental roster (two-seat damped self-play, 256 streams, four
producers, capacity recovery) at width 384, 8 layers and 8 heads with the
model's default feed-forward (MTG-317's w384l8), bucketed shapes on, and
precision written explicitly. The damping horizon is 300 updates in both arms,
so learning rate and τ match at every update. `require_single_axis` checks that
only the id and `execution.precision` differ.

A run seed fixes the initial weights and every native game seed, so the two
arms of a seed start from the same weights and meet the same pairings and deals
until their policies diverge.

Strength: each run's checkpoints at updates 20, 40, 75 and 150 and at its
end (300 on schedule) play the frozen heuristic over the fifty held-out Elemental pairings
of 07c490 (`learning_signal.ELEMENTAL_PAIRS`, four cross legs each, 200 games
per checkpoint). Each BF16 run also plays each float32 run at the last update
both checkpointed (four head-to-heads, 800 games; 4,800 games in all). Trajectories are the
per-update `policy_loss`, `value_loss`, `numerical/clip_fraction` and
`collection_kl` in each run's metrics, read in the report's ML panels.

Budget: the watchdog allows 300 updates at an assumed 150 s per float32 update
(the A40 measured 100.6 s at a384l8, [`training-update-time.md`](training-update-time.md)),
12.8 h per run. Etude reserves USD 7.85 per run at the L4's USD 0.60/h ceiling,
USD 31.40 for four, plus USD 2 for one CPU evaluation pass. A float32 run
slower than assumed stops at its watchdog with its exports; the comparison then
uses *M*, the last update every run reached. MTG-317's w160l8 capture update
took 197 s on an L4, so float32 at w384l8 may be slower than 150 s and stop
short of 300.

## Length and its limits

The scored comparison is 300 updates per run on schedule, two seeds per arm,
the length the approved budget (about USD 30) covers. A 64-update version was
queued first under the earlier USD 8 budget and cancelled before any machine
was rented, once the larger budget was approved; it produced no data. Limits:

- The watchdog, not the schedule, may set the length. If float32 runs slower
  than 150 s per update, its runs stop early and *M* falls below 300; the
  answer states *M*.
- The A/B trains at w384l8 only. The ladder applies BF16 from 256, where the
  speed and fixed-weight numerics were measured but learning was not.
- Two seeds per arm give a narrow float32 band; a band from two points can
  miss real spread, in either direction.
- 300 updates is still short of frontier length; slow-accumulating BF16
  effects (optimizer-state rounding, value-head drift) beyond it are untested.

## Status

Dated line, 2026-11-12: RunPod listed L4s at USD 0.69/h, above the 0.60
ceiling, so the scored run moved to on-demand `mtg324-l4-ab` (ceiling USD 0.70/h)
with the watchdog shortened to 10.9 h per run (`--hours 10.9`), keeping USD 7.85
per run and USD 31.40 for four. The horizon stays 300 updates; at 150 s per
update the window covers about 250, so *M* is likely below 300. The question is
unchanged.

Dated line, 2026-11-12: the queued cohort `mtg324-precision-ab-300c` started
three runs; BF16 seed 32402 was lost when RunPod refused capacity. It was
re-queued alone (`--arms bf16 --seeds 32402`) from a later commit that adds
only that filter to `runs()`; its regime and job config are identical.

Dated line, 2026-11-13: fetched all four final training records after their
windows ended. The numerical rejection decides the gate, so no paid evaluation
was launched. Main's automatic layout settings were merged into the implementation;
precision uses their existing resolution and authoring path. The experiment arm
now writes `live_tokens=False` explicitly to retain its declared layout.

</details>
