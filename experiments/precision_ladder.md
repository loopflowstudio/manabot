# Does BF16 learn the same as float32 at w384l8? — MTG-324

## Question and prediction (committed before any scored run)

`Execution.precision="auto"` trains a CUDA model at width 384 or above in BF16
(`BF16_MIN_HIDDEN_DIM`). MTG-317 measured BF16 2–2.7× faster than float32 at
w384l8 on an L4 ([`cuda_precision.md`](cuda_precision.md)), but its replay could
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
prediction holds, the ladder stays at 384; if it is refuted or unresolved,
`ladder_precision` returns float32 everywhere. **Cost cap:** USD 34 for the
experiment (training launched with `--max-usd 32`, evaluation capped at USD 2),
within the Task's USD 35 cap and the approved budget of about USD 30.

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
- Two seeds per arm give a narrow float32 band; a band from two points can
  miss real spread, in either direction.
- 300 updates is still short of frontier length; slow-accumulating BF16
  effects (optimizer-state rounding, value-head drift) beyond it are untested.

## Status

Scored run queued 2026-11-12 on this Task's own on-demand `mtg324-l4`
(`--max-usd 32`), waiting for L4 stock at or under USD 0.60/h.

## Answer

Pending.
