# Two-seat Allies/Lessons: $50 scale-ladder experiment

**Interim report, 2026-10-21 23:57 UTC. Training is running; the $50
endpoint and its strength evaluation have not been reached.** This report will
retain each attempt and its costs, including failures, as the experiment proceeds.

Both-seat learning is working in the real w6 engine: all **20,909,111
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

A CPU evaluation of update 250 against heuristic is running on the
existing Mac mini: 25 declared deals, seeds 1590100000–1590100024, with all eight
seat/deck legs (200 games). It uses the same anchor and deal-bootstrap analysis
as v1. There is no additional evaluation rental. No win rate is claimed before
all games complete and their saved results are checked.

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

The controller is live on the Mac mini. It adopted 002 with its original lease;
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
There is no current evidence that two seats outperform one.

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
