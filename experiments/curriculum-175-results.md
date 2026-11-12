# Curriculum adapters: results

Final report, 2026-11-08. The study closed with the evidence below; no further
training or evaluation is planned. Four curricula chose which deck matchups the
learner played during two-seat Elemental-v2 self-play, all from the same
initial weights:

- **U**, uniform: MTG-183's existing seed-18301 window, reused rather than retrained.
- **M**, moderate-matchup preference: favours pairings whose estimated win rate is near 33/67 or 67/33.
- **E**, value-target error: favours pairings where the value prediction misses its bootstrapped target most.
- **ME**: an equal mixture of the normalized M and E distributions.

The [protocol](protocols/curriculum-175.md) defines selection, the fixed
evaluation deals and the analysis. Every number here is one training seed
against heuristic v2 or another arm, scored with equal weight per ordered deck
cell and native seat. No invalid or missing game appears in any cohort below.

## Summary

- **No curriculum beat uniform.** Against heuristic v2 at update 800, U scored
  67.75%, E 65.00% and M 64.25% (400 development games each). Head to head at
  update 800, U beat M 54.75% and E 53.00%; M beat E 52.00% (400 games per pair).
- **M and E finished level on the held-out deals.** At their cost-matched
  checkpoints, M (update 700) scored 64.65% and E (update 750) 63.50% over 2,000
  games each. M−E is +1.15 points, 95% paired-deal bootstrap interval
  −0.95 to +3.20.
- **The primary U comparison is missing.** U's held-out endpoint was never
  evaluated, so the preregistered gain-over-initialization contrast against U
  cannot be computed. ME has no trained-policy score at all.
- **The curricula did change what was played.** E concentrated play on fewer
  pairings (27.9% of starts in its top ten cells, against 15.9% for M) and on
  red-green, as the sampling section below shows. That shift did not show up as
  strength.

Against the protocol's practical threshold (+5 points over U without any deck
worsening by more than 5), neither M nor E comes close on the evidence there is.

## Development curve against heuristic v2

Development uses two deals per ordered cell in both seats: 400 games per point.
The initial weights are shared, so update 0 is the same policy for every arm.

| Update | U | M | E | ME |
|---:|---:|---:|---:|---:|
| 0 | 6.75% | 6.75% | 6.75% | 6.75% |
| 100 | — | 45.75% | 38.50% | — |
| 200 | 55.50% | 53.00% | 49.75% | — |
| 400 | — | 56.25% | 57.75% | — |
| 800 | 67.75% | 64.25% | 65.00% | — |

The update-0, 100 and 400 points were played on the owned CPU host; 200 on
rented A4000s; 800 in one batched CUDA allocation. Placement does not change
the deals or the heuristic, and the three update-0 rows (one per arm's copy of
the initial weights) agree exactly. U has no 100/400 points because its window
was reused from MTG-183, which was monitored on a different deal family.

## Head to head at update 800

All 100 ordered deck cells, two deals each, both native seats: 400 games per pair.

| Pair | First arm's score | Games (W–L) |
|---|---:|---:|
| U vs M | 54.75% | 219–181 |
| U vs E | 53.00% | 212–188 |
| M vs E | 52.00% | 208–192 |

The ordering is consistent (U > M > E) and modest; with one seed it is
exploratory, not an established ranking.

## Held-out endpoints

Held-out uses ten deals per ordered cell in both seats: 2,000 games per point,
on a deal family disjoint from development and from MTG-183. The protocol
selects each arm's last export whose first-seen inclusive cost estimate stays
within U's $7.225 reference, without consulting scores: update 700 for M and
750 for E.

| Policy | Score | Gain over initial |
|---|---:|---:|
| Initial weights | 5.80% | — |
| M, update 700 | 64.65% | +58.85 |
| E, update 750 | 63.50% | +57.70 |
| U endpoint | not evaluated | — |

| Model deck | M700 | E750 |
|---|---:|---:|
| Black-green | 75.0% | 70.5% |
| Red-green | 64.5% | 70.5% |
| Green-white | 49.5% | 49.0% |
| Black-red | 71.0% | 64.5% |
| Blue-black | 62.5% | 62.0% |
| Blue-green | 63.5% | 69.5% |
| Blue-red | 82.5% | 78.0% |
| White-blue | 44.5% | 41.5% |
| White-black | 62.0% | 61.5% |
| Red-white | 71.5% | 68.0% |

By native seat, M scored 64.9% / 64.4% and E 65.0% / 62.0% (seat 0 / seat 1).
The paired interval resamples whole deals (both seats of one deal together),
1,000 shared deals, 5,000 resamples.

## What is missing, and why

- **U held-out endpoint.** Its evaluation was cancelled when the study closed.
- **ME strength.** ME trained to update 870 (last export 850), but its update-800
  GPU evaluation lost its machine before starting, and its CPU evaluations were
  cancelled at close. Only its initial weights were scored.
- **Development points at 1,200 and later**, and U's initial held-out cohort
  (1,600 of 2,000 games, retained as delivered).

Most CPU evaluations were delayed for days: the owned CPU host had no storage,
so it never received checkpoints that training had written to R2. Etude 0.2.13
fixed that delivery, and those runs then played every game. Lost and cancelled
attempts remain in the launcher and in the Task's retained records; none is
counted as a score.

### Update-200 development results by deck and native seat

Each deck row contains **40 valid games per observed arm**: ten opposing decks,
two deals per pairing and both native seats. Cells receive equal weight. Each
seat row contains **200 valid games per arm**, with all 100 ordered cells.
There are no missing or invalid U/M/E games in these strata; Differences are percentage points against U on the same deals.

| Model deck | U | M | E | M−U | E−U |
|---|---:|---:|---:|---:|---:|
| Black-green | 65.00% | 57.50% | 65.00% | −7.50 | 0.00 |
| Red-green | 65.00% | 60.00% | 57.50% | −5.00 | −7.50 |
| Green-white | 35.00% | 32.50% | 37.50% | −2.50 | +2.50 |
| Black-red | 62.50% | 65.00% | 52.50% | +2.50 | −10.00 |
| Blue-black | 60.00% | 55.00% | 50.00% | −5.00 | −10.00 |
| Blue-green | 50.00% | 37.50% | 35.00% | −12.50 | −15.00 |
| Blue-red | 62.50% | 67.50% | 65.00% | +5.00 | +2.50 |
| White-blue | 27.50% | 42.50% | 30.00% | +15.00 | +2.50 |
| White-black | 65.00% | 57.50% | 50.00% | −7.50 | −15.00 |
| Red-white | 62.50% | 55.00% | 55.00% | −7.50 | −7.50 |

| Heuristic opponent deck | U | M | E | M−U | E−U |
|---|---:|---:|---:|---:|---:|
| Black-green | 65.00% | 65.00% | 57.50% | 0.00 | −7.50 |
| Red-green | 45.00% | 37.50% | 37.50% | −7.50 | −7.50 |
| Green-white | 52.50% | 57.50% | 57.50% | +5.00 | +5.00 |
| Black-red | 80.00% | 65.00% | 65.00% | −15.00 | −15.00 |
| Blue-black | 60.00% | 60.00% | 55.00% | 0.00 | −5.00 |
| Blue-green | 62.50% | 57.50% | 60.00% | −5.00 | −2.50 |
| Blue-red | 40.00% | 35.00% | 30.00% | −5.00 | −10.00 |
| White-blue | 65.00% | 62.50% | 52.50% | −2.50 | −12.50 |
| White-black | 40.00% | 47.50% | 45.00% | +7.50 | +5.00 |
| Red-white | 45.00% | 42.50% | 37.50% | −2.50 | −7.50 |

| Model native seat | U | M | E | M−U | E−U |
|---|---:|---:|---:|---:|---:|
| 0 | 53.00% | 52.00% | 49.00% | −1.00 | −4.00 |
| 1 | 58.00% | 54.00% | 50.50% | −4.00 | −7.50 |

E's aggregate deficit appears in both native seats and six of ten model decks;
it is not confined to one deck. M's largest observed gain is white-blue (+15pp),
while blue-green is lower for both adaptive arms. These are small, descriptive
development strata, not significant subgroup effects, final forgetting measures
or evidence that later curriculum concentration caused the differences.
The update-393–400 exposure window below occurs **after** these evaluations.

The derived strata, exact estimator and analysis source, planned schedule and
input artifact digest are retained in
`experiments/data/curriculum-175/development-200/deck-seat-strata.json`.
The shared estimator preserves missing/invalid games as full-target bounds;
it never substitutes the delivered subset for the planned population.

## How the curricula changed actual game selection

This matched training diagnostic uses updates **393–400**, read from retained
Etude metrics at 2026-10-27T12:00:21Z (M) and 12:00:32Z (E). It is a later
training window than the update-200 strength evaluation above. Each game start
is counted once by its ordered anchor pairing; both seats still learn. These
are observed allocation/exposure diagnostics, not additional evaluation games.

After the initial 200-stratum sweep, both adapters reserve every fifth start
for shuffled coverage. Remaining starts randomize one-quarter age refresh and
three-quarters priority: approximately **20% coverage, 20% age, 60% priority**.
M ranks supported pairings by proximity to 33/67 or 67/33 self-play outcomes;
E ranks unnormalized value-target residuals. These are training-policy signals,
not heuristic-evaluation scores. Priority is rank-based, capped at 5% per cell.

| Observed allocation, updates 393–400 | M: 66/33 preference | E: value error |
|---|---:|---:|
| Physical game starts | 2,136 | 2,178 |
| Priority / coverage / age starts | 58.7% / 20.0% / 21.3% | 59.1% / 20.0% / 20.8% |
| Ordered pairings with at least one start | 100 / 100 | 100 / 100 |
| Share of starts in the ten most-used pairings | 15.87% | 27.87% |
| Games involving red-green on either side | 18.73% | 32.92% |
| Learner decisions belonging to red-green | 5.68% | 11.90% |
| Latest priority mass on its ten highest-priority cells | 26.27% | 41.67% |

The ten most-used pairings are selected separately from each observed window;
this is descriptive concentration, not a significance test against uniform.
The final row instead describes the priority distribution frozen at update 400,
not the realized start distribution across the window. Game starts and learner
decisions have different denominators because games/roles yield different
numbers of decisions: total learner decisions were 395,455 (M) and 345,399 (E).

For M, black-red versus red-white had smoothed self-play win probability 33.05%
and its reverse 66.95%; each received the maximum **5% priority weight** at
update 400. They received 48 and 39 starts, respectively (2.25% and 1.83% of
all recent starts). Only 46 of 100 cells had the required 20 completed roles
per seat in the recent teacher window; unsupported cells retained neutral
pre-cap priority rather than being treated as balanced or easy.

For E, red-green versus blue-red and blue-green received 81 and 80 starts
(3.72% and 3.67%). The red-green mirror illustrates signal disagreement: its
smoothed outcome is exactly 50/50, so M gives it only **0.250% priority weight**,
while E gives it **5.000%** because its residual is high. Actual mirror starts
were 1.17% under M and 3.21% under E; safeguards and changing weights explain
why the realized shares differ from the latest priority weights.

Thus E is currently concentrating game starts more strongly, particularly on
red-green pairings. That demonstrates changed sampling, not useful learning:
high residual can reflect prediction difficulty or variance. The strength
comparison, final costs and combined arm are still needed.

Raw selected update rows, per-cell counts/weights, immutable block keys and the
exact analysis source are published in
`experiments/data/curriculum-175/distribution-400/m.json` and `e.json`.
Per-update assignment counts reconcile with cumulative sampler starts after the
window's first row, and role exposure totals reconcile with residual-row counts.

### Sampling differences persisted through update 500

A second matched eight-update window, **493–500**, was read at 13:33 UTC on
2026-10-27 using the same allocation estimator. Both runs have genuine,
downloaded and digest-verified update-500 checkpoints. These are additional
training diagnostics; there are no new strength-evaluation results here.

| Observed allocation | M, 393–400 | M, 493–500 | E, 393–400 | E, 493–500 |
|---|---:|---:|---:|---:|
| Physical game starts | 2,136 | 2,129 | 2,178 | 2,197 |
| Share in the ten most-used pairings | 15.87% | 16.67% | 27.87% | 27.26% |
| Games involving red-green on either side | 18.73% | 18.69% | 32.92% | 34.27% |
| Learner decisions belonging to red-green | 5.68% | 5.89% | 11.90% | 11.30% |
| Ordered pairings with at least one start | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| Latest priority mass on the ten highest-priority cells | 26.27% | 23.15% | 41.67% | 41.67% |

E continued to concentrate starts more strongly than M, particularly on
red-green pairings. The two windows do not establish a monotonic trend or
causal learning benefit. The top-ten sets are selected separately in each
window; priority mass describes the latest frozen weights, not observed starts.
The unchanged E top-ten priority mass follows its rank transformation and cap;
it does not imply unchanged error magnitudes or unchanged favored pairings.

In the new window, priority/coverage/age accounted for **59.93%/19.96%/20.10%**
of M starts and **59.63%/19.98%/20.39%** of E starts. All decision exposures
reconcile with the retained residual counts: **402,914** M decisions and
**343,066** E decisions. M had sufficient recent outcome support for 38 cells,
down from 46 at update 400; the other cells retain neutral pre-cap mass.

Exact analysis source, immutable metric-block keys, selected rows and per-cell
counts/weights are retained in
`experiments/data/curriculum-175/distribution-500/m.json` and `e.json`.
This window was inspected after it existed and is descriptive, not a new
preregistered efficacy endpoint. Final strength, uncertainty, full-history
coverage/forgetting and the combined arm remain outstanding.

## Cost

Training cost, as launcher estimates at each arm's last export (not invoices):
M $7.63 (update 800), E $7.88 (850), ME $7.58 (850). U reuses MTG-183's window
with a $7.225 cleanup estimate. Each arm stayed inside its $10 inclusive
allocation; no allowance moved between arms. Evaluation used owned CPU hosts at
no rental cost, about $0.16–0.18 of A4000 time per update-200 point (detailed in
`evaluation-costs/completed-rentals.json`) and at most $1.50 per batched GPU
allocation.

## Identity and limits

Training and the evaluations above ran from the frozen Manabot runtime
`0fc90e4130aca0e28a5e1b56cd9f2004b86a6845`, world w13, v1 policy input and
heuristic v2. U's initialization digest is
`1004ccdf46f373e833dadcb5278351a758e5328ed7c28049259c8c06be614688`.
Current Manabot reads saved recipes as the current policy input, so the
experiment file on main is a record of the study, not a relaunch path.

One training seed per arm, U's reuse of a window trained under earlier runtime
maintenance, and the curricula's coverage safeguards (20% coverage, 20% age,
60% priority, 5% cap per cell) all limit causal interpretation. Missing points
are reported as missing, never as zero.

## Data

Published under `experiments/data/curriculum-175/` (hashes in
[the artifact manifest](data.manifest.json)):

- `final/held-out-summary.json`: held-out scores, breakdowns and the paired M−E interval.
- `final/development-800-scores.json`, `final/h2h-800-scores.json`: update-800 results.
- `final/tables/`: game tables of the eleven CPU evaluations (development 0/100/400, held-out 0/700/750).
- `development-200/`, `distribution-500/`, `initial-evaluation/`, `evaluation-costs/`: earlier published results.

```sh
uv run --extra artifacts python -m manabot.data fetch experiments/data/curriculum-175
```
