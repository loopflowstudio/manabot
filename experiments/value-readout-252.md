# Critic readout comparison — MTG-252

**Status: training authorized, scoring protocol not frozen.** Jack Heart
(2026-11-05) authorized 3 arms × 3 seeds (25201–25203) × 150 updates inside a
USD20 cap covering training, evaluation and retries. Training launches as Etude
cohort `mtg252` (9 runs, at most USD1.45 each, USD13.05 reserved). Thresholds and
evaluation schedule size are still open and are fixed before any score is read.

## Question

The current critic reads a masked mean over its final object board. With
up to 202 object slots on the Elemental recipe (two players, 60 cards and 40
permanents per seat), many similar permanents each carry a proportional share of that summary. Does a
learned summary — a value token that attends alongside the objects, or a single
learned query over the final board — learn a better critic and policy at equal
cost, without erasing count information the game depends on?

## Arms

All arms are the MTG-183 Elemental recipe
([`elemental-long-183-v1.json`](regimes/elemental-long-183-v1.json)) re-pinned
to the native world (w17 at this writing), with only `agent.value_aggregation`
changed. That recipe trains two-seat current-self on the ten Elemental decks
with uniform per-game pairings (`elemental-uniform`). Re-pinning resolves its
saved observation at the current fresh policy input (v6, 512 facts) with 183's
capacities: 60 cards, 40 permanents, 64 actions, overflow an error.
[`value_readout_252.py`](value_readout_252.py) derives the arms and rejects any
other resolved difference: depth 2, width 64, four heads, scalar critic, damped
learning with quantile .75 and floor .01, lambdas, batch closure and execution
are unchanged. The frozen 183 file and its evidence are not edited or relabeled. No sparse attention, grouping, capacity change, failure policy
or value-loss change enters this comparison.

| Arm | Summary the critic MLP reads | Added parameters | Attention pairs per observation |
| --- | --- | ---: | ---: |
| `masked_mean` (baseline, 1,084,034 parameters) | Mean over valid objects after the per-object projection | 0 | 81,608 |
| `value_token` | One trainable neutral-owner token, appended before object attention | 64 | 82,418 |
| `attention_readout` | One learned query cross-attending over the final valid objects | 16,704 | 81,810 |

The value token takes part in both attention blocks, so it also changes the
object outputs that score actions: it is not a critic-only intervention. The
attention readout runs after object attention and only the critic reads it;
actor logits at initialization are identical to the baseline's, though the
value loss still trains the shared encoder in both learned arms. A flat
N×D-to-D readout was not added: it fixes capacity and ties weights to slot
positions. A within-board GRU would add order dependence and is not game memory.

Under one initialization seed the readout arm shares every baseline parameter.
The token arm does not: its draw precedes later input encoders and shifts
their initial values. Changing that order
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
pre-clip norm over critic-only parameters (`value_head`, `value_readout`), and in
the `value_token` arm `gradient_norm_value_token`, the token's own pre-clip norm.
The token is an unmasked key in object attention, so actor logits read it and
its gradient mixes policy and value loss; it is shared, not critic-only.
Existing rows, retained rows, actor/critic exposures, empty-filter skips and
selection groups are unchanged.

## Proposed protocol (to be frozen before scoring)

1. **Source and world.** Pin the landed commit, `managym.world_identity()`,
   observation shapes and native identities in a protocol JSON. The arms'
   regimes are committed after that pin; 183's executed evidence is not rerun
   or relabeled.
2. **Seeds.** One list of new run seeds, shared by every arm
   (`value_readout_252.runs`). The run seed fixes initialization and the
   collection seed. Native game `k` on stream `s` is seeded
   `run seed + 10000 + s + k × streams`, and its Elemental pairing and deal
   are a function of that seed alone, so under one run seed every arm meets
   the same pairing and deal sequence on every stream whatever its policy
   does (`test_pairings_depend_on_the_run_seed_not_the_arm`). Policy and RNG
   trajectories still diverge after the first update, and an arm whose games
   run longer or shorter may sit at a different point of that sequence at a
   given update.
3. **Training.** The re-pinned 183 stage at a fixed update count with
   milestone exports. Cost per update must be measured on the w17 base before
   a budget is set; 183's w13 timings and v10's TLA timings do not transfer.
4. **Evaluation against the heuristic.** Each arm's milestone and final
   checkpoints play the heuristic over one common schedule of random
   Elemental deck pairings: ordered pairs drawn with a fixed seed from the
   10 × 10 grid (mirrors included), each deal played with the model on the
   play and on the draw, as in `mtg186_eval`'s "rest" schedule. One fixed
   schedule seed draws the pairings and deals once; every arm and seed plays
   that identical schedule, pinned by SHA-256 in the protocol. Failed and missing slots stay
   in place.
5. **Head-to-head.** Final policies play each other directly, each arm
   against each other arm within a seed, on that same fixed schedule with
   both seat legs. Arm A is model A in every game of a pairing.
6. **Analysis.** Paired differences against `masked_mean` by seed and deal
   block, with intervals over seeds and deals; head-to-head scores per arm
   pair with intervals over seeds and deals; per-collected-transition,
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
| Learning rule | 183's damped rule as is | Same rule as v10; a later learning-rule review may change it |
| Thresholds | Promotion and rejection rules | Earlier screens used ±5 points with all-seed sign agreement |
| Shared baseline | Own `masked_mean` runs, or one baseline shared with the sparse-attention study | Sharing needs identical base recipe, seeds, source and schedule; neither study crosses the other's axis |
| Schedule size | Deals per heuristic evaluation and per head-to-head pairing | Evaluation must cost a fraction of training |

Acceptance may be negative or inconclusive. No outcome changes a default
without a separate decision.
