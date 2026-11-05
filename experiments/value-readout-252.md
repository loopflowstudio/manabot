# Critic readout comparison — MTG-252

**Status: draft protocol. Nothing is frozen, scored or launched.** Budget,
arm count, seed count and numeric thresholds are open decisions listed below;
no compute is authorized by this document or the software that supports it.

## Question

The TLA pilot's critic reads a masked mean over its final object board. With
up to 386 object slots (two players, 128 cards and 64 permanents per seat),
many similar permanents each carry a proportional share of that summary. Does a
learned summary — a value token that attends alongside the objects, or a single
learned query over the final board — learn a better critic and policy at equal
cost, without erasing count information the game depends on?

## Arms

All arms are the frozen v10 recipe
([`tla-trophy-213-v10.json`](regimes/tla-trophy-213-v10.json)) with only
`agent.value_aggregation` changed. [`value_readout_252.py`](value_readout_252.py)
derives them and rejects any other resolved difference: input v7, capacities,
first-64 action support, depth 2, width 64, four heads, scalar critic, damped
learning with quantile .75 and floor .01, lambdas, batch closure and execution
are unchanged. No sparse attention, grouping, capacity change, failure policy
or value-loss change enters this comparison.

| Arm | Summary the critic MLP reads | Added parameters | Attention pairs per observation (v10) |
| --- | --- | ---: | ---: |
| `masked_mean` (baseline) | Mean over valid objects after the per-object projection | 0 | 297,992 |
| `value_token` | One trainable neutral-owner token, appended before object attention | 64 | 299,538 |
| `attention_readout` | One learned query cross-attending over the final valid objects | 16,704 | 298,378 |

The value token takes part in both attention blocks, so it also changes the
object outputs that score actions: it is not a critic-only intervention. The
attention readout runs after object attention and only the critic reads it;
actor logits at initialization are identical to the baseline's, though the
value loss still trains the shared encoder in both learned arms. A flat
N×D-to-D readout was not added: it fixes capacity and ties weights to slot
positions. A within-board GRU would add order dependence and is not game memory.

Under one initialization seed the readout arm shares every baseline parameter.
The token arm does not: its draw precedes the v7 continuation, fact and
shared-effect encoders and shifts their initial values. Changing that order
would change the initialization of recipes that already use a token, so it is
recorded here as a known difference rather than repaired.

Ataraxos motivates the token, but this repository's
[fidelity table](../docs/learning.md#fidelity-to-the-sources) records the value
token as a local choice, not a recovered source detail. Claim no fidelity
without rereading the primary paper.

## Prior evidence

Both prior studies ran on frozen w4 with one attention layer and a scripted
opponent; neither transfers as current-world evidence.

- [Value-token screen](value-token-screen.md): endpoint means historical 39.67%,
  masked 42.00%, token 34.67% over three seeds; no arm met promotion or
  rejection. Token received roughly a third to a half of masked mean's optimizer
  exposures, with 31 empty-filter updates.
- [Pooling × floor follow-up](pooling-filter-followup.md): removing the .01
  floor raised token exposures by 102–154% without a reliable score change; the
  token-minus-masked floor interaction was −1.33 points [−12.67, 11.00].

The follow-up already answered whether removing the floor rescues the token on
w4. This study keeps the floor and measures whether it binds on the current
world; any floor intervention needs its own declared contrast.

## Witnesses (software, no training)

`tests/model/test_value_aggregation.py` exercises each arm's real attention
stack and readout:

- object order and 1–250 rows of padding leave the value unchanged;
- one, two, eight and forty copies of a permanent beside single objects give
  four distinct values;
- a unique permanent among forty identical copies still moves the value;
- swapping which seat controls every object changes the viewer's value;
- duplicating *every* object leaves masked mean and the attention readout
  exactly unchanged, while the token sees it. Softmax attention and mean pooling
  read counts only relative to other objects; real boards always hold single
  player objects, so the copies witness is the operative case.

None of these witnesses promises duplicate invariance or measures learning.

## Diagnostics retained per update

Damped updates now log, for every collected row, `value_scale`: collection
prediction and lambda-target mean/standard deviation, explained variance of the
target by the prediction (bootstrapped targets, not final outcomes), advantage
standard deviation, the selection threshold actually applied and whether the
.01 floor set it. Sampled optimizer steps add `gradient_norm_critic_readout`, the
pre-clip norm over critic-only parameters. Existing rows, retained rows,
actor/critic exposures, empty-filter skips and selection groups are unchanged.

## Proposed protocol (to be frozen before scoring)

1. **Source and world.** Pin the landed commit, `managym.world_identity()`,
   observation shapes and native identities in a protocol JSON, as v10 does.
   The arms' regimes are committed after that pin; v10's executed evidence is
   not rerun or relabeled.
2. **Seeds.** New paired initialization/collection seeds, identical across
   arms. Policy and RNG trajectories diverge after the first update; pairing
   controls initialization and deals, not trajectories.
3. **Training.** The v10 stage at a fixed update count with milestone exports.
   On the v10 L4 run, 16 updates took 2,399 stage seconds, so 100 updates is
   roughly four L4-hours per run: nine such runs are about 37 hours, or about
   USD22 at that rental's USD0.59/hour, before evaluation and setup.
4. **Evaluation.** Each arm's checkpoints play the matched first-64 heuristic
   and its own initialization over a common set of paired deals with both
   seat/deck legs, scheduled by `tla_pilot_v10.comparison_schedule`. Failed and
   missing slots stay in place.
5. **Analysis.** Paired differences against `masked_mean` by seed and deal
   block, with intervals over seeds and deals; per-collected-transition,
   per-optimizer-exposure and per-wall-second curves on overlapping support;
   value fit (explained variance against lambda targets, and against final
   outcomes on evaluation games); actor entropy and KLs; inference
   milliseconds per decision and peak memory measured at ordinary batch sizes
   on the evaluation host.

## Open decisions

| Decision | Options | Note |
| --- | --- | --- |
| Arms | 2 (masked, token) or 3 (+ readout) | The readout separates "learned summary" from "token changes the actor" |
| Seeds | 3 or more | Three seeds left every earlier interval wide |
| Updates per run | Fixed count, e.g. 25 or 50 with milestones | Sets the budget almost alone |
| Budget | Not set | No MTG-213 or 5da5f4 allowance is inherited |
| Learning rule | v10 damped rule as is | The full-TLA successor's learning-rule review may change it; freezing before that review risks a stale baseline |
| Thresholds | Promotion and rejection rules | Earlier screens used ±5 points with all-seed sign agreement |

Acceptance may be negative or inconclusive. No outcome changes a default
without a separate decision.
