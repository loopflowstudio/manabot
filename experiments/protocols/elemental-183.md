# MTG-183: one long Elemental seed, rated against its own checkpoints, for $50

Jack Heart, 2026-10-23. Question: is there a point where the ten Elemental decks
have saturated against the heuristic but newer checkpoints still beat their
past selves? If so, the heuristic stops being informative there and a
checkpoint-population rating has to take over. This run starts that rating
system. It is one seed, not a seed-spread claim, and the rating is pinned to
the Elemental world: no cross-world Elo claim.

Motivating evidence (MTG-181 seed 18101, a local side check outside its frozen
protocol, copied in `scratch/mtg181-h2h.md`): against the heuristic 0.21 at
update 0 and 0.79 at update 200; newer vs older on the 10×10 deck grid, 1,000
games each, 200 v 0 0.966, 400 v 200 0.571, 600 v 400 0.539.

## Freeze

The git commit that adds this file, `elemental-183-identity.json`, the job
specs, the regime and the evaluation code is the runnable source. Every
window is compiled from that exact clean, published commit with
`manabot deploy compile` and run by one `manabot deploy cohort` service; no
later change enters it. The identity record binds world w13, the ten
`elemental-world-v2` decks with their Lesson pools, content/observation/action
digests, the Darwin native extension used for local evaluation, `uv.lock`, the
regime, job specs, and the evaluation code. Linux pod builds and native digests
are retained per TrainingRun and are not claimed identical to Darwin bytes.

Source differences from MTG-181's frozen source (`0e9c4427`), none chosen for
effect: the base is current main `003644bd`, which MTG-181 had merged; its grid
evaluator, arena leg selection and `User-Agent` repair are copied unchanged,
except that the grid evaluator can play a declared subset of its schedule.

## Recipe

`experiments/regimes/elemental-long-183-v1.json` is MTG-181's
`elemental-two-seat-181-v1` with only its ID changed: 64-wide two-layer
four-head scalar masked-mean additive policy, Adam, damped-policy learning with
absolute-iteration LR/tau schedules over a 100,000-update horizon, EMA .999,
256 streams, at least 64 decisions per stream per update, both seats learning,
independent uniform ordered deck pairs at game roots including mirrors, CUDA
float32, one producer, MTG-176 game-error recovery at its engineering defaults.
Training seed **18301**.

## Allocation and cost

A RunPod job is limited to twelve hours (single STS session), so the seed
trains as cohort **`mtg183-elemental-long`**: seven windows on one L4 each
(MTG-181's machine: live price at most $0.60/hour plus $0.02/hour storage,
6 vCPU, 48 GB), setup 900 s, checkpoint 120 s, upload 300 s, cleanup 120 s, and
CUDA numerical validation before learning in every window.

| Windows | Lifetime | Spending limit (incl. $0.10 guardian proof) |
|---|---:|---:|
| `mtg183-elemental-long-w1` … `-w6` | 11.9 h each | $7.48 each |
| `mtg183-elemental-long-w7` | 6.0 h | $3.82 |

Reserved total **$48.70 of the $50 inclusive cap**; the remaining $1.30 covers
deletion latency and observation uncertainty. Window 1 trains from
initialization. Each later window is a learner-state continuation with
`iteration="last-complete"`: after its predecessor pauses at its allocation
cutoff, publishes verified final raw/EMA/Adam/run exports and confirms provider
cleanup, the cohort service binds those exact bytes and admits the next window.
Raw policy, Adam, EMA, absolute update clocks and cumulative counters continue;
collector, game and RNG streams restart at each boundary (in-flight games at a
cutoff are abandoned, not learned from). This is the documented segmented
continuation, not exact CUDA process recovery, and not byte-reproducible
training. Each window's setup and validation cost about 20 minutes of rental.

The cohort service runs on this laptop under launchd (the Mac mini used by
MTG-165 was unreachable on 2026-10-23). It only schedules: every window's own
guardian and supervisor own its pause, export and deletion deadlines, so a
sleeping or offline controller can delay the next window but cannot extend a
rental. Cohort scheduling ends six days after admission.

Failures stop the cohort: a failed window, incomplete exports, unknown cleanup
or an exhausted reservation admits no successor, and no failed window is
replaced (no remaining reservation could hold one). Cost is the conservative
intent-to-observed-deletion estimate from job receipts, not an invoice. No
top-up, extension or additional seed.

**Deadline stops (MTG-182 is not merged).** Each window ends at its allocation
cutoff, which abandons every in-flight game. Under this source the game-error
window may count those abandoned games and the run's `stop_reason` may read
`game_error_rate_exceeded`. A window whose non-completed game attempts are all
`outcome=deadline` at its final update is read as an ordinary deadline pause,
and the cohort continues if the window is `paused` with complete exports (as
MTG-181's seed 18102 was). Any `crashed` attempt of another outcome, or a stop
before the cutoff, is a real game-error stop. Quarantined or abandoned games
never count as training or evaluation results.

## Exports

Every window exports a raw checkpoint at every multiple of **50 absolute
updates**, plus each window's final update. Every export is retained in the
job store. Rated checkpoints (the only ones evaluated):

- every 50 updates through 400, every 200 through 2,000, every 400 after;
- the **endpoint**: the last window's final update, admitted once the cohort is
  settled.

Window boundary exports and other cadence exports are retained but unrated.
Each continuation window's initial export repeats its predecessor's endpoint
and is not a separate checkpoint.

## Evaluation (local CPU, $0)

All games run on this machine through the replay-checked arena `play_cell`,
one torch thread per player, stochastic policies, 600-second and
10,000-command game limits, ten worker processes, at most one hour per
(cell, seat) group. Score: win 1, draw ½, loss 0, over valid games only
(terminated, not truncated, no failure, replay passed). Invalid games are
counted and shown, never replaced; a failed group counts its games invalid and
is not replayed.

**Against the heuristic.** MTG-181's frozen 10×10 grid (rows: model deck; columns:
heuristic deck; mirrors included; seat per game from seed 18190001; schedule
`ecc487e4…264`). Updates **0, 100, 200, 400, 800, 1,600, 3,200, 6,400** and the
endpoint play the full grid, 50 games per cell (5,000 games). Every other rated
checkpoint plays the declared subset: the first 10 games of every cell of the
same schedule (1,000 games), so all heuristic scores are paired by deal.

**Cross-checkpoint.** A newer checkpoint (row deck) plays an older one (column
deck) on the 10×10 grid, 10 games per cell, 1,000 games per pair, the newer
checkpoint's seat drawn per game from seed **18390001**, deals
1,839,000,000 + 1,000·cell + index (schedule `27f22397…f60`, disjoint from the
heuristic deals). Every pair plays the same deals. Per-deck and per-seat
results are recorded. The pairing graph (`pair_design`):

- **consecutive**: every rated checkpoint against the previous one;
- **reference**: every later rated checkpoint against updates **200** and **2,000**;
- **endpoint**: the endpoint against every earlier rated checkpoint;
- **held-out**: rated[i] against rated[i+2] for even i when not already a pair;
  these never enter the rating fit.

With an endpoint near 8,600, that is 34 rated checkpoints, 107 fit pairs and
14 held-out pairs (121,000 games), plus 9 full and 25 subset heuristic grids
(70,000 games).

`experiments.elemental_long_follow` plays these as exports arrive, oldest newer
checkpoint first, heuristic grid before pairs, and never rents or changes
training. It yields while MTG-181's grid follower is still running on this
machine. MTG-181's production grids took about 80 minutes per 5,000 games, so
expect about 19 hours for the heuristic grids and about 32 hours for the pairs;
everything but the endpoint's units (about 9 hours) overlaps training. The follower stops at cohort admission
+ 9 days; anything unevaluated then is reported as missing.

## Rating and analysis

Elo-scale Bradley-Terry with `manabot/arena/rating.py`: the seat-aware
Gaussian-MAP fit (400-Elo prior SD, one seat-0 advantage parameter), every
valid game of the fit pairs and every valid heuristic game, the heuristic as
anchor player fixed at 1,000. Deck effects are not modelled; every pair and
grid plays all 100 ordered deck cells equally. Uncertainty is the paired-deal
block bootstrap (2,000 replicates, seed 18301; a block is one deal, shared
across all pairs or all heuristic grids that play it).

The report gives:

- the full payoff matrix over rated checkpoints and the heuristic, with
  per-pair scores, Wilson intervals and fit residuals (Pearson and deviance);
- intransitivity: every triangle A > B > C with C > A among played pairs,
  where each edge's interval excludes 0.5;
- held-out prediction: predicted vs observed scores for held-out pairs,
  their log loss against a 0.5 baseline and calibration;
- rating stability: ratings refit on each prefix of rated checkpoints (only
  the games among them and the heuristic) against the full fit, which shows
  how far ratings move when new checkpoints join;
- the heuristic curve and the rating curve on one shared update axis, overall
  and per model deck.

**Saturation point.** On the heuristic curve, U_h is the earliest rated update
whose paired difference from the endpoint (endpoint − U_h, deal bootstrap,
subset deals for subset checkpoints) has a 95% interval that includes zero or
lies below it, and after which no later rated checkpoint is significantly above
the endpoint. Saturation **exists** when, in addition, the endpoint's rating
exceeds the rated checkpoint at U_h with a bootstrap interval above zero, and the
endpoint v U_h pair score's Wilson interval lies above 0.5. Per deck, the same
rule uses that deck's row on the heuristic grid and the newer checkpoint's row
in pairs against U_h(deck). Decks are ranked by U_h and by their remaining
self-play gain after it. If no such point exists in this run, the report says so
with the curves.

**Design choices recorded for later rating work:** games per pair, the pairing
graph, the heuristic anchor, the single seat parameter, and the stability
measurement above. A future population rating can fix existing ratings and fit
only new checkpoints against them; this run measures how much a full refit
would move them instead of choosing that policy.

## Predictions, fixed before allocation

- Throughput: about 110–130 updates per training hour, as MTG-181; the
  endpoint lands between 7,000 and 9,500 updates. No illegal action,
  non-finite value or replay mismatch; game-error rate far below 10% apart from
  deadline abandonment.
- Heuristic: overall score rises from about 0.21 at update 0 past 0.79 by 200,
  and U_h lies between 400 and 3,200, with the endpoint between 0.80 and 0.92.
- Self-play: every consecutive pair through 1,000 scores above 0.5; the endpoint
  beats update 2,000 with a score of at least 0.55; the endpoint's rating is at
  least 100 Elo above U_h's. Saturation exists.
- Decks: saturation updates differ by deck; at least one deck's heuristic row
  is still rising at 1,600 updates.
- The rating predicts held-out pairs better than 0.5 (lower log loss), and
  no triangle is significantly intransitive.

Refutation, a failed window, no saturation point and no useful effect are
reportable outcomes. Kill criteria are the job's own: non-finite values,
illegal offered action, identity mismatch, game-error rate above 10% before a
cutoff, or exhausted time/cost allowance. Stop rather than tune or substitute.

## Amendment, 2026-10-24 02:10 UTC, before any evaluation result

Writing the report's synthetic test showed that the saturation rule above is
degenerate at this sample size: with 5,000 paired heuristic games, a remaining
gain of 0.004 is already significant, so a heuristic score that keeps creeping
upward never "saturates". The primary rule is therefore a practical margin:
U_h is the earliest rated update whose remaining gain (endpoint − U_h) has a
95% upper bound below **0.03**, after which no later rated checkpoint is
significantly above the endpoint. In practice only full-grid checkpoints can
meet that bound (a 1,000-game subset's interval is too wide). The strict rule
is still reported beside it. The existence test (endpoint rating above U_h,
endpoint v U_h above 0.5) is unchanged. No game had been played when this was
written; training window w1 was in setup.

## Amendment, 2026-10-24 19:00 UTC: the seed continues on Etude after w2

Jack Heart's migration retired the cohort controller on 2026-10-24 at about
18:45 UTC, so w3–w7 above never start. Jack Heart decided (about 18:55 UTC) to
continue the same seed past w2 on Etude within the remaining cap: one learner
lineage, no new seed. Results so far are the follower's evaluation units of w1
and w2 exports; no training decision below depends on them.

**What stays.** Seed 18301, the regime's learning recipe (EMA, Adam, damped
policy with absolute-iteration schedules, streams, decision floor, seats,
deck curriculum, game-error policy), the 50-update export cadence on the same
absolute coordinates, the rated set, the heuristic grid, the cross-checkpoint
schedule and pairing graph, the rating, the analysis and the predictions above.

**Segment w3.** One Etude training run (`manabot.experiment.job`, built by
`experiments.elemental_long.segment_job`) continues w2's final export:

- Parent: w2's last complete update, read from its own `run.json` (expected
  near 2,550). `stage_parent` uploads that run record and its published
  raw/EMA/Adam exports to the Etude machine storage and refuses any byte that
  differs from what the record published. The worker refuses a parent that ends
  at another update or used another seed.
- Machine: one L4, rented by Etude with CUDA numerical validation before
  learning, one attempt (`max_attempts=1`).
- Dollars: `max_usd` = $50 − w1's and w2's observed cost (job receipts, the
  same conservative estimate as above) − $1.30 reserve.
- Hours: `max_usd` ÷ the quoted L4 hourly price at launch, rounded down to a
  tenth of an hour; the regime watchdog leaves 30 minutes for setup and the
  numerics gate and 10 minutes for finalization.
- Update target: the parent update + 0.9 × w2's measured updates per training hour ×
  (hours − 0.7), rounded down to a multiple of 50. Milestones: every multiple
  of 50 after the parent. The regime's `updates` changes only the endpoint:
  damped-policy rates follow the absolute iteration, so the learning schedule
  is the 100,000-horizon one above.
- These four numbers, the parent's digests and keys, and the source commit are
  written to `experiments/protocols/elemental-183-w3.json` and committed before
  allocation. The source is a published commit on main that contains Etude's
  training job with learner continuation (#362 and this Task's change).

Collector, game and RNG streams restart at the boundary, as at every earlier
window boundary. MTG-182 is on main, so w3's in-flight games at a cutoff are
recorded as run interruptions, not game errors.

**Endpoint.** The endpoint is w3's last tracked export once Etude reports the
run done, whatever its outcome. A run that stops short of its target (time,
dollars, or a failure after some exports) still ends the seed there; its
outcome is reported. A run that fails before its first export leaves w2's
final update as the endpoint. No retry, extension or further segment.

**Follower.** The cohort follower keeps evaluating w1/w2 exports until w2
has ended and its ledger holds w2's final cadence export. It is then stopped
and restarted from the w3 source commit on the same directory and ledger,
naming the Etude tracking storage and the w3 run; completed units are kept.
It admits w3's tracked exports by absolute update, settles when the run is
done, and stops two days after w3's deadline plus the endpoint's units
(about 9 hours); anything unevaluated then is reported as missing. The report
shows Etude's run outcome in place of the retired cohort phase.

## Amendment, 2026-10-25 00:00 UTC, before any evaluation result: heuristic v2, Etude eval, R2

Jack Heart, 2026-10-24: move the remaining work to R2 and the new Etude code,
and anchor the heuristic curve and the rating on **heuristic v2** (`639ab7770`,
`manabot/arena/heuristic.py` on main). No evaluation game had been played when
this was written: the follower's ledger held 46 admitted exports and 51 units,
all queued behind MTG-181's follower.

**Anchor.** Every heuristic game is played against v2. The heuristic agent's
identity is the digest of `heuristic_source_paths()`, so a game against another
heuristic version is never pooled with these; the report reads only the
current one. Heuristic scores here are not comparable with MTG-181's, the
ladder's or MTG-165's (v1). The prediction above (0.21 at update 0 rising past
0.79 by 200) was made against v1; against the stronger v2 the curve is expected
lower, and that prediction is not scored. The other predictions stand.

**Evaluation.** `experiments.elemental_long_eval` replaces the follower
(`experiments.elemental_long_follow`, deleted). It plays the same design —
rated set, frozen grid and cross-checkpoint schedules (digests unchanged),
full/subset grids, pairing graph, game and group wall bounds, and the rule that
a failed or timed-out group counts its games as invalid and is never replayed —
as one Etude eval run (`etude.eval.match.evaluate`, experiment
`mtg183-elemental-eval`, run `mtg183-eval`) on this Mac's CPU at $0. One Etude
match is one row of a grid or pair schedule (player 0's deck, 20 groups); its
`games` table holds one row per game and player. Every match's tables are
delivered to R2 (`r2://manabot/evaluations/`); the report reads them through
`etude.eval.query.statsdb`. Order, yielding to MTG-181's follower and the stop
rule are as before. Checkpoints come from the cohort follower's ledger (w1/w2,
read-only) and w3's Etude run. The rating fit, uncertainty, saturation rule and
analysis are unchanged.

**Cutover.** The cohort follower stops once its ledger holds w2's final export,
and in any case before MTG-181's follower exits; it never plays a game.
`elemental_long_eval` then starts from a pinned checkout of this change.

**Storage.** w3's tracking and checkpoints go to R2 (Etude SDK 0.2.1). After w2
ends and w3 has staged w2's learner state, w1/w2's retained objects move from
S3 to R2; checkpoint agents are identified by digest, not location, so moving
their bytes does not change any match.

## Amendment, 2026-10-25 02:30 UTC, before allocation: segments of one rental each

w2 ended at its deadline (01:47 UTC) at update 2,534 (`paused`); its receipts
are $7.2358 (training $7.2198, guardian $0.0161), so w1 and w2 together cost
$14.4611. Etude rents a machine for at most 43,000 seconds (its 12-hour
storage-session ceiling), so the single w3 run above cannot fit one rental,
and `max_attempts=1` gives it no second one. The continuation is therefore a
chain of Etude segments, as w1–w7 were planned under the cohort: w3, w4, …
Each continues its predecessor's final export through the same
`stage_parent` / `segment_job` path, with the same recipe, seed and cadence;
collector, game and RNG streams restart at every boundary.

For each segment, with `remaining` = $50 − every earlier receipt − $1.30:

- Hours: the lesser of 11.6 (43,000 s less the on-demand's 900 s setup
  allowance, rounded down to a tenth) and `remaining` ÷ the quoted L4 price
  − 0.25 h, rounded down to a tenth. Etude prices a rental as its price ×
  (hours + 0.25 h of setup).
- Dollars: `max_usd` = the lesser of `remaining` and the on-demand's $0.60
  ceiling × (hours + 0.25), rounded down to a cent.
- Update target: the parent update + 0.9 × w2's measured rate × (hours − 0.7),
  rounded down to a multiple of 50. w2's rate is 1,209 updates in its stage's
  42,033 s: 103.55 updates per hour.
- A segment's spec (`experiments/protocols/elemental-183-w<N>.json`) is
  committed before it is allocated; the next one is sized from receipts only
  after its predecessor has ended. A segment that is not worth its setup
  (under 2 hours) is not launched.

The chain ends when the remaining dollars allow no further segment, or when a
segment ends without a final export (failure before its first export, or
refusal); the endpoint is then the last tracked export of the last segment
that produced one. No segment is retried. The evaluator admits every
segment's exports by absolute update and settles when the chain ends.
