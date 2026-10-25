# MTG-181: two-seat self-play on ten Elemental decks, three seeds for $10

Jack Heart, 2026-10-22/23. Question: does the two-seat recipe learn on all ten
accepted Elemental decks? This is a basic trial: three training runs that
differ only in training seed. It is not a one-seat comparison (MTG-165 settled
that), a world-balance study or a cross-world strength claim. Launch gate
(#354, #355, #356) verified merged on main before freezing.

## Freeze

The git commit that adds this file, `elemental-181-identity.json`,
`elemental-181-job.json`, the regime and the evaluation code is the runnable
source. Every job is compiled from that exact clean commit with `manabot deploy
compile` and submitted with `manabot deploy submit`; no later change enters it.
The identity record binds world w13, policy input 1, rules decision 6 /
possible-world 2, all ten `elemental-world-v2` decks with their Lesson pools,
content/observation/action/experience digests, the Darwin native extension
used for local evaluation, `uv.lock`, the regime, job spec and evaluation code.
Linux build/native digests are retained per TrainingRun on the pods and are not
claimed identical to Darwin bytes.

## Recipe

`experiments/regimes/elemental-two-seat-181-v1.json` is MTG-174's frozen recipe
(`elemental-uniform-174-v1`): the 64-wide, two-layer, four-head scalar
masked-mean additive policy (FiLM off), Adam, every MTG-159 damped-policy
(Ataraxos move) learning parameter, absolute-iteration LR/tau schedules, EMA
.999, 256 streams, at least 64 decisions per stream per update, independent
uniform ordered deck pairs at game roots including mirrors, CUDA float32, one
producer. The intended change is **both seats learn** (`learning_seats: both`):
every decision is a learner decision and each stream finishes its current game
before the update, as delivered by MTG-165 and the consolidated `manabot train`
path (#356).

Compatibility changes, none chosen for effect:

- world w10 → w13 and pack `elemental-world-v1` → `elemental-world-v2`: the
  current accepted roster after the mana, hexproof and rules repairs. The
  representative authored match is BG vs GR; the curriculum samples the full
  roster regardless.
- schema 1 → 2, and the gradient tag `ataraxos_move` → `damped_policy`, its
  current name. Hyperparameters are identical.
- Source repair: #356's `Agent` → `ActorCritic` rename also rewrote the HTTP
  `User-Agent` header to `User-ActorCritic`; RunPod's REST API then refused
  inventory reads with HTTP 403. The header is restored (with a regression
  test) before allocation. No learning or evaluation code changed.
- MTG-176 game-error recovery on with its engineering defaults: quarantine and
  regather crashed games, fail the run above a 10% settled-game error rate over
  a 100-game window after 100 games, 40,000 actions per game. Quarantined games
  never enter learning or results.

Seeds **18101, 18102, 18103**; jobs `mtg181-elemental-two-seat-<seed>`. Within
a run, absolute update targets stay at 100,000; the allocation deadline stops
training far earlier and exports the endpoint. A local CPU smoke of this regime
(32 streams, 3 updates) collected about 7,000 learner decisions per update, so
256 streams should take about 56,000 learner decisions per update.

## Allocation and cost

`elemental-181-job.json`: one L4 per seed at a live price of at most
$0.60/hour plus $0.02/hour storage, **5.0 hours** lifetime, $3.15 job limit,
setup 900 s, checkpoint 120 s, upload 300 s, cleanup 120 s, CUDA numerical
validation before learning (five-minute ceiling inside the job). Each job's
projected allocation is $3.10; with a guardian proof each (about $0.02), the
three seeds project **$9.36** of the **$10 inclusive cap**, leaving $0.64 for
deletion latency and observation uncertainty. The three jobs are submitted
concurrently from this commit. No paid calibration or paid evaluation: the
smoke and evaluation calibration ran locally at $0.

A seed whose submission or training fails is reported with its charges and is
not replaced: no second allocation fits the remaining cap. Cost is the
conservative intent-to-observed-deletion estimate from job receipts, not a
provider invoice; any prior MTG-181 rental would count (none exists). The
remote supervisor and guardian own pause, export and deletion deadlines.
No top-up, extension or continuation.

## Evaluation

Jack Heart's design (2026-10-23), replacing the brief's per-pair cohorts.
Each evaluated checkpoint plays a **10×10 grid against the heuristic**: rows
(x) are the model's deck, columns (y) the heuristic's deck, all ten decks on
each axis including mirrors, **50 games per cell, 5,000 games per checkpoint**.
Each game has its own deal; whether the model is on the play (seat 0 takes the
first turn) is drawn per game from one recorded RNG seed, **18190001**, giving
2,452 play and 2,548 draw games. `experiments/elemental_grid.py` generates the
schedule; its digest `ecc487e4…264` is pinned in code and the identity record.
Every checkpoint and seed plays the same 5,000 games.

Games run on local CPU through the replay-checked arena (`play_cell`, one
torch thread per player, stochastic policy, 600-second and 10,000-command game
limits), ten worker processes, one per (cell, seat) group, at most one hour per
group. Score: win 1, draw ½, loss 0, over valid games only (terminated, not
truncated, no failure, replay passed). Invalid games are counted and shown,
never replaced; a failed group counts its games invalid and is not replayed.
Cells, rows, columns, play/draw and overall report the mean with a 95% Wilson
interval. Intervals describe game noise for one checkpoint, not variation
between training seeds; the seed spread is reported separately.

The heuristic is `manabot.arena.heuristic.HeuristicPlayer`, the anchor Jack
Heart named. Its targeting polarity covers 7 of the 94 roster cards; other
targeted spells fall back to v1's seat-ordered target, while its mana, attack
and block rules apply everywhere. It is a legal, fixed anchor, not a strong one.

**Cadence:** initialization (update 0), every raw export at a multiple of
**200 absolute updates** (jobs export every 200 updates), and the actual
endpoint once the job is terminal with confirmed cleanup. No other checkpoint
is evaluated or selected afterwards. A local calibration of the full grid on
the smoke run's untrained initialization completed 27 of 200 groups in six
minutes with eight workers while tests shared the machine, projecting 45–55
minutes per checkpoint (about 2.7–3.3 seconds per game per worker). Expected
throughput (below) gives 3–6 evaluations per seed, 9–18 checkpoint hours of
local CPU; `experiments.elemental_two_seat_follow` plays them oldest update
first while training runs and finishes after it. The follower stops at
allocation + 48 hours; anything unevaluated by then is reported as missing.
It never rents, cancels or changes training.

The endpoint grid's 5,000 replay-checked games are each seed's load, replay,
legality and completion proof (ordinary checkpoint loading admits world and
ABI identity). No held-out deals are reused for selection: nothing is selected.

## Report

`experiments.elemental_two_seat_follow` rebuilds, after each evaluation, an
Etude metric run per seed (`mtg181-elemental-grid`, clocks `updates` and
`learner_transitions`: overall, play/draw, per model deck and per heuristic
deck), beside each pod's own Etude training run (`self-play`), and renders
Etude's report pages plus `index.html`: training counters, cost, seed spread
by update, each checkpoint's grid with per-cell rate and interval, and
row/column/overall aggregates. The final report records per seed: learner and
environment decisions separately, games, collection and learning time,
throughput, quarantined/crashed games and error rate, inclusive cost, curves,
and the spread across seeds.

Comparison with MTG-159's Allies-vs-Lessons $10 baseline is within-world only
(each run against its own anchor and relative to its initialization) or per
dollar. Different worlds, decks and anchors: no cross-world strength claim.

## Predictions, fixed before allocation

- Throughput: 4–9M learner decisions per hour of training (L4, two-seat);
  about 4.4 hours of training per seed reaches 300–700 updates and 18–40M
  learner decisions.
- No illegal action, non-finite value or replay mismatch; game-error rate far
  below 10%.
- Learning: each seed's endpoint overall score exceeds its initialization by at
  least 0.10, and the 95% intervals do not overlap. Endpoint overall score
  against the heuristic is between 0.35 and 0.70; across-seed endpoint range
  below 0.10.

Refutation, safe early stop, a failed seed and no useful effect are reportable
outcomes. Kill criteria are the job's own: non-finite values, illegal offered
action, identity mismatch, game-error rate above 10%, or exhausted time/cost
allowance. Stop rather than tune or substitute.

## Amendment, 2026-10-23 17:40 UTC, before the retry's allocation

The three jobs were submitted concurrently from `9e47a7d6`. The first attempt
stopped before any rental for all three: a shell quoting slip truncated the
worker role ARN, so STS refused issuance ("no rental created"). The same job
IDs were then resubmitted, as documented for uncertain submissions.
`mtg181-elemental-two-seat-18102` and `-18103` were accepted and are training.
For `mtg181-elemental-two-seat-18101`, RunPod answered the **guardian probe's**
create request with HTTP 500. That claim stays ambiguous; no pod with its name
remained after the probe's 90-second self-deletion deadline, and no training
rental was ever requested under that ID. Its conservative charge (the probe's
reservation, at most $0.05) counts against the cap.

The "not replaced" rule above assumed a failed training allocation, which would
leave no room under the cap. Here no training was allocated, and the brief
permits retried segments inside the inclusive cap. Seed 18101 is therefore
retried once, unchanged, as `mtg181-elemental-two-seat-18101-r2`: same regime
bytes, seed, job spec, cadence and evaluation. Projected inclusive total:
two accepted jobs $6.20 + their guardian proofs + the ambiguous probe + the
retry $3.10 and its proof ≈ **$9.41** of $10. The retry's source commit adds
only this amendment and the follower's job-ID mapping; learning and evaluation
code are unchanged. If the retry fails, seed 18101 is reported and not retried
again.

## Results

`experiments/data/mtg181/report.md` (fetch with `manabot.data fetch
experiments/data/mtg181`). All three seeds learned: 0.206 → 0.851 mean overall
against the v1 heuristic (endpoint range 0.015), ≈ $9.12 inclusive. 18102's
`game_error_rate_exceeded` stop is a deadline cut, not game errors. The trial
scripts are retired; read them at
`739fb251d:experiments/elemental_two_seat{,_follow,_report}.py`. The report adds
the endpoints against main's strengthened heuristic.
