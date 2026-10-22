# Two-seat Allies/Lessons: $50 scale-ladder experiment

**Interim report, updated 2026-10-22. The training/cost snapshot is from
2026-10-21 23:57 UTC; the early heuristic evaluation is now complete. Training
is running, and the $50 endpoint has not been reached.** This report will
retain each attempt and its costs, including failures, as the experiment proceeds.

Both-seat learning is working in the real w6 engine: all **20,909,111**
environment decisions in the verified snapshot were retained as learner decisions,
and the 250- and 500-update raw checkpoints were exported. This establishes
collection, optimization and checkpoint export. It does not establish stronger play
or a gain in learning per dollar.

## Question and recipe

Does retaining both players' decisions improve the cost/strength curve of the
Allies/Lessons scale ladder? Jack Heart selected the ladder in
[PR #333](https://github.com/loopflowstudio/manabot/pull/333), then raised this
experiment from $10 to $50 and selected the Mac mini as controller.

The [frozen recipe](regimes/scale-ladder-allies-lessons-both-v2.json) keeps seed
15902, the width-64/two-layer semantic policy with scalar masked-mean value,
256 cross-balanced streams, Ataraxos move learning, float32 CUDA, one learner
CPU thread, EMA .999 and absolute-iteration learning-rate/tau schedules. It uses
an L4 admitted at $0.59/hour, plus $0.02/hour storage allowance. Raw checkpoints
are exported every 250 updates. The declared tiers are $10 within one day and
$50 within seven days. See the [protocol and budget amendment](protocols/two-seat-ladder-002.md).

The substantive differences from one-seat v1 extend beyond row retention.
Two-seat collection takes at least 64 combined decisions per stream, then
finishes its current game. Terminal seat trajectories are packed without
padding; credit stops at every seat/game boundary. After computing targets,
Ataraxos groups consecutive rows into batches of at most 256 decisions.
The old control has fixed one-seat windows and bootstrapped partial-game tails.
The schedule still counts updates, so it is not matched by learner decisions.

## Verified training snapshot

Counters below come from immutable generation 414 of job
`mtg165-two-seat-002`, not an extrapolation from quotas. A later provider/status
read at the observation time above confirmed the same job still running at
712 updates. The snapshot counters stop at 704.

| Measure | Snapshot value |
|---|---:|
| Completed collection/update iterations | 704 |
| Complete games | 211,389 |
| Learner decisions | 20,909,111 |
| Environment decisions | 20,909,111 |
| Retained optimizer exposures after advantage filtering | 5,227,546 |
| Optimizer steps | 81,988 |
| Run wall time | 3.961 hours |
| Collection time | 1.041 hours |
| Learning time | 2.863 hours |
| Recorded nonfinite values / rejected optimizer steps | 0 / 0 |
| Actual training device | `cuda:0` |

Retaining every decision does not mean every row produces a gradient: the
unchanged advantage filter selected about
25.0% of rows for optimizer exposure.
Learning accounts for 73.3% of recorded collection-plus-learning
time. The observed run throughput is 1,466
learner decisions/second; this is one run on one rented host, not a matched
throughput comparison or evidence of a 2× speedup.

| Update | Learner decisions | Environment decisions | Raw checkpoint SHA-256 |
|---|---:|---:|---|
| 250 | 7,595,063 | 7,595,063 | `2bd07fa24a524536471c3e4d0a03e9b5a42fc19201f007db6d1bb6b490126897` |
| 500 | 14,913,525 | 14,913,525 | `9ca62d8abaf5993c840aae0e62d1935dec167a61d12d40bfb8c8dbfac83c26de` |

## Early heuristic result: update 250

Completed 2026-10-22T00:07:15.513952+00:00 on the existing Mac mini CPU, without
another rental: 25 declared deals, seeds 1590100000–1590100024, all eight
seat/deck legs, **200 games**. Every game was terminal and replay-verified, with
zero recorded integrity failures. The evaluation took
425.84 seconds. All referenced compressed trace
hashes were independently checked after copying the evidence from the mini.

The two-seat checkpoint scored **51.5% [48.5, 54.5]** against heuristic. The
one-seat v1 checkpoint at the same 250 updates scored 47.5% [44.5, 50.5].
The descriptive paired difference is **+4.0 percentage points [−1.0, +8.5]**;
the interval includes zero. Scores below are percentages with 95% intervals;
deltas are percentage points. Matchups name the trained policy's deck first.

| Matchup | One seat, update 250 | Both seats, update 250 | Paired delta |
|---|---:|---:|---:|
| LvA | 22 [12, 32] | 24 [12, 38] | 2 [-10, 14] |
| AvL | 82 [72, 92] | 74 [62, 86] | -8 [-18, 0] |
| LvL | 38 [28, 48] | 60 [52, 68] | 22 [8, 36] |
| AvA | 48 [40, 56] | 48 [40, 56] | 0 [-8, 8] |
| mean | 47.5 [44.5, 50.5] | 51.5 [48.5, 54.5] | 4 [-1, 8.5] |

LvA is Lessons versus Allies; AvL reverses those decks; LvL and AvA are mirrors.
Each cell has 50 games from 25 deals; the mean has 200 games. The apparent mean
gain is concentrated in Lessons mirrors, while Allies versus Lessons declines.
Lessons versus Allies, the primary tracking matchup, changes from 22% to 24%
with a paired interval spanning −10 to +14 points.

Both evaluations use the same engine/content/ABI fingerprints, heuristic source,
25 deals, eight legs and one-pass CPU inference envelope. Bootstrap analysis
resamples whole deals, keeping both variants and all seat/deck legs paired:
4,000 draws, seed 17. The paired-difference analysis is descriptive and post hoc,
not a method-level hypothesis test; matchup intervals are unadjusted for
multiple comparisons. There is still only one training seed.

**This is not a cost-matched win.** At these checkpoints, two-seat training used
7,595,063 learner decisions versus 4,096,000, and its training wall coordinate
was 5,222.94 seconds versus 1,530.62 seconds: 1.85× the learner rows and
3.41× the wall time. Environment decisions were 7,595,063 versus 8,304,667.
Complete-game credit, row order, schedule coordinates and physical training hosts
remain confounded. These results establish a playable early checkpoint with a measured anchor
score, not improved strength per dollar or a completed $10/$50 tier.

[Scored rows, intervals, identities and receipts](data/two-seat-ladder/250-update-heuristic.json)
retain both 200-game cohorts and their exact input hashes. The original one-seat
result is the completed `heuristic-0` cohort retained for #333; its earlier
spawn-failure attempt is not substituted for successful games.

## Costs and remaining allocation

These are estimates from provider receipt rates, plus the declared storage
allowance. They are not invoices. The live rental's estimate ends at the report's
observation time; it continues accruing cost.

| Attempt or allowance | State | Dollars |
|---|---|---:|
| 001: earlier PPO pilot, bootstrap failed before training | Deleted; cleanup confirmed | 1.5313 |
| 002: guardian proof | Deleted; cleanup confirmed | 0.0163 |
| 002: L4 training rental to observation time | Running | 2.4625 |
| Estimated incurred total, including earlier failure | | **4.0101** |
| Controller allowance, reserved separately | Not a measured charge | 0.20 |
| All admitted/queued reservations plus prior cost and controller allowance | Admission bound | **49.9813** |
| Authorized experiment ceiling | | **50.00** |

The controller and bounded dashboard-report companion are live on the Mac mini.
They retain progress independently of this laptop. The controller adopted 002
with its original lease;
003–007 each have an 11.9-hour allocation and 008 has 4.9 hours. Their reservations
include guardian allowances. After a clean allocation pause, final raw/EMA/Adam
and run exports must verify and provider cleanup must be confirmed before the
next segment can start. Models, Adam moments, EMA and absolute counters continue;
collector/game/RNG streams restart. No continuation has yet run in this report.
Failures, missing final exports, uncertain cleanup or excess cost stop admission.

The overall deadline is **2026-10-28 19:53:48 UTC**. Reservations are ceilings,
not a promise that every dollar will be spent. Later estimates and cleanup
receipts must replace these live estimates without dropping prior attempts.

## What the comparison can establish

The one-seat v1 small tier reported 151.6M learner decisions, heuristic score
0.70 as Lessons versus Allies and mean 0.69 over four matchup cells, at its $10
checkpoint ([frozen v1 report](https://github.com/loopflowstudio/manabot/blob/32acac9019b3d94a10173827f8d6782c87e18fa0/experiments/scale-ladder.md)).
The present two-seat run has not reached a comparable spend or evaluation endpoint.
The early same-update paired interval includes zero; there is no current
evidence of an overall strength or cost-efficiency advantage for two seats.

The strongest confound is the changed collection/credit boundary and row order,
combined with update-based schedules that see different numbers of decisions.
This is also only one training seed, and its L4 host need not have v1's CPU.
A score interval over deals measures this checkpoint's evaluation uncertainty;
it does not establish a method-level gain or training-seed robustness. A matched
one-seat/two-seat comparison with equal games, seeds and cost would discriminate
row-retention benefits from the other changes.

## Evidence and final-report requirements

- [Machine-readable observation](data/two-seat-ladder/2026-10-21-interim.json)
  contains all counters, cost bases, exact snapshot references and checkpoint hashes.
- Worker source: `fabb96db2bb88f98891aba66cc9d12e65d700545`; world: `w6`; training seed: `15902`.
  TrainingRun: `b855d41288c542ffb8af8c337db81186`. The source and native runtime fingerprints are
  retained in its immutable run record.
- Generation 414 was fetched through the normal remote-job reader, which
  verified its bundle. Run JSON SHA-256:
  `cb1c5ba67f5da2f849bf638ca35e4d557711bb4b342d83e02805b313b3c612f5`. The private S3 manifest reference is in the data file;
  no public bucket access is required or enabled.
- The final report remains pending: settle every admitted segment or stop reason,
  confirm cleanup and all-attempt costs, bind the endpoint raw/EMA/Adam artifacts,
  evaluate the selected tier checkpoints against heuristic, report Lessons versus
  Allies first and the full matchup matrix with deal uncertainty, then update
  this report without implying that an interrupted run reached $50.


## Live Etude monitoring

The Mac mini now runs a separate serial Etude Eval observer, pinned to Manabot
`33054b347`, alongside the existing budget controller. Its local-network report
is [mini-heart.local:8766](http://mini-heart.local:8766/). The page and JSON exports
refresh every 30 seconds during evaluation and retain timestamps, coverage,
failures, checkpoint digests and the last observed training updates.

Each exported absolute 250-update raw checkpoint gets the same 25 development
deals × eight deck/seat legs against the heuristic. The arena retains command
traces and verifies terminal replay; an invalid deal fails its Etude match.
Etude holds the benchmark, writes match records and Parquet tables, and resamples
whole deals for its score intervals. These remain repeatedly inspected development
monitors. Its default bootstrap is 2,000 draws/seed 0, so interval endpoints may
differ from the earlier monitor's 4,000 draws/seed 17; the original evidence above
remains unchanged.

LaunchAgent `manabot.mtg165.etude-evaluation` uses an isolated evaluation checkout
and environment on the Mini. Each checkpoint has a one-hour process allowance;
failed or interrupted attempts are retained without automatic replay. The observer
cannot rent, extend or restart training. The read-only report server serves only
public report files. Detailed evidence stays under
`/Users/jack/.manabot/mtg165-two-seat-50/evaluation/` on the Mini. The training
worker and cohort controller keep their frozen source identities and lease.


Activation retained two interrupted evaluation attempts: the first exposed an
empty-score-table reporting bug, now covered by a regression test; the second
was stopped to remove macOS Background scheduling, while preserving its completed
Etude matches. An explicit resume reads those completed records rather than
replaying them. Training and paid rental deadlines were unaffected. At the latest
activation check, the original GPU job was running at 831 updates with a fresh
controller heartbeat. The complete 200-game Etude result was still pending.
