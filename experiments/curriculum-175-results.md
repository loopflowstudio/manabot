# Curriculum adapters: interim results

Development outcomes observed 2026-10-27T11:38:53.947448+00:00; operational
status checked at 12:29 UTC. **The final four-arm effects comparison is not yet
available.** This interim report records complete U/M/E development results,
their deck/seat breakdowns and the retained initial-evaluation failure while
the experiment continues.

The treatments are uniform U, moderate-matchup preference M (peaks near 33/67
and 67/33), value-target error E, and their normalized-rank mixture ME. Jack's
latest allocation is **$10 inclusive per arm, $40 total**, superseding the
original $5-per-arm goal. The [protocol](protocols/curriculum-175.md) defines
selection, fixed evaluation deals, cost-based checkpoint selection and analysis.

## Update-200 development results

On the predeclared 400-game development grid, **U scored 55.50%, M scored
53.00%, and E scored 49.75%**. M was **2.50 percentage points below U** and E
was **5.75 percentage points below U** on these fixed deals. Each cohort contains
all 400 planned games, all valid, with zero missing or invalid games. Scores
weight ordered deck cells and native seats equally.

| Arm | Valid / planned games | Observed full-grid score |
|---|---|---|
| U | 400 / 400 | 55.50% |
| M | 400 / 400 | 53.00% |
| E | 400 / 400 | 49.75% |
| ME | 0 / 400 | Pending |

This is an exploratory development point at equal update counts, not the primary
cost-selected or held-out endpoint comparison. It describes the observed grid;
it does not establish a general curriculum effect or final ranking. The
preregistered four-arm paired confidence intervals and interaction remain
pending ME. The single historical-control seed and maintenance/coverage
confounds still apply. Missing arms are not assigned zero scores.

The frozen identities, admitted rows, coverage and complete-grid bounds are
published at `experiments/data/curriculum-175/development-200/three-arm-outcomes.json`.
The earlier U/E snapshot remains in `early-outcomes.json`. All three update-200
evaluation jobs have now succeeded, with their rentals confirmed ended.
Delivered valid games are retained independently of that operational status.

## Development results by deck and native seat

Each deck row contains **40 valid games per observed arm**: ten opposing decks,
two deals per pairing and both native seats. Cells receive equal weight. Each
seat row contains **200 valid games per arm**, with all 100 ordered cells.
There are no missing or invalid U/M/E games in these strata; ME remains pending
throughout. Differences are percentage points against U on the same deals.

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

## Initial-policy evidence

The common initial policy's held-out evaluation delivered **1,600 valid games
of 2,000 planned**, with **0 delivered invalid games and 400 missing games**.
Eight of ten match units finished before the one-hour execution deadline.
Etude subsequently recorded `lost / machine_lost`, with rental termination
confirmed. No missing game is imputed or automatically replayed.

Giving each ordered deck cell and native seat equal weight, the complete-grid
initial-policy score is bounded by **4.7%–24.7%** when each missing score may
lie anywhere in [0, 1]. These are identification bounds, not confidence
intervals or an adapter-effect estimate. The observed cells are not substituted
for the full target population.

The initial weights are shared by the adaptive arms and the reused MTG-183
control. This initial evaluation therefore establishes starting-policy context;
it cannot establish that a curriculum helped or hurt.

Training status below is the 12:27 UTC snapshot, with evaluation completion
checked at 12:29 UTC; the live dashboard updates independently. The scores above
retain their original frozen-data read.

| Arm | Training at 12:27 UTC | Update-200 evaluation |
|---|---|---|
| U | Reused MTG-183's genuine window through update 1,325 | Succeeded; 400 valid games |
| M | Running, update 444 | Succeeded; 400 valid games |
| E | Running, update 472 | Succeeded; 400 valid games |
| ME | Queued for an L4 | Awaits its genuine export and budget check |

M's first remote update-200 request was refused before machine acquisition or
command start, with a recorded charge of $0. Its unchanged checkpoint and match
keys were retained for a capacity recovery. This is distinct from U's timed-out
evaluation. Both failures and all unstarted-request cancellations remain in the
operational evidence; they are not silently dropped from the experiment history.

## Identity and reproducibility

Scientific execution remains frozen at Manabot
`0fc90e4130aca0e28a5e1b56cd9f2004b86a6845`, world w13 and historical v1 policy
inputs. U's initialization digest is
`1004ccdf46f373e833dadcb5278351a758e5328ed7c28049259c8c06be614688`.
The retained outcome reader verifies checkpoint digest, policy and heuristic
versions, world identity and tournament against the original launch record.
It does not recompute these identities from the newer reporting checkout.

The admitted initial outcomes, frozen policy/anchor/world records and failure
receipts are published as:

- `experiments/data/curriculum-175/initial-evaluation/outcomes.json`
- `experiments/data/curriculum-175/initial-evaluation/operations.json`

Their immutable content hashes are in [the artifact manifest](data.manifest.json).
Fetch them with:

```sh
uv run --extra artifacts python -m manabot.data fetch experiments/data/curriculum-175
```

The original match records and tables also remain in Etude's
`r2://manabot/jobs` store. The [implementation PR](https://github.com/loopflowstudio/manabot/pull/383)
and [report-read support](https://github.com/loopflowstudio/etude/pull/58) provide
integration evidence, not evidence of curriculum gains.

## Cost and interpretation

U retains its historical $7.225 cleanup estimate; it was not trained again.
Its incomplete initial eval had a nominal rental limit of $0.3125. The
update-200 capacity allowance is at most $0.34 per arm. The retained M/E training
rental limits are $8.9975 each; prior startup charges and the full $0.50 reserve
remain included in each arm's $10 limit. ME requires a fresh bound before its
evaluation is submitted. No unused allowance moves between arms.

Live provider billing revised U's elapsed-rate estimate downward when a
closed-hour record arrived. These amounts remain provisional, not invoices;
the lower estimate is not treated as extra spending permission. SDK run-share
cost ends at command completion and can omit the paid evidence-delivery tail.
Final inclusive costs must include each dedicated rental through confirmed
machine deletion, plus setup failures and earlier evaluation charges.

A read-only provider billing check at **2026-10-27T12:47:31Z** found revised
charges for completed rentals that the controller had already stopped refreshing.
The table retains the distinct measurements. Provider hour records are a dated
snapshot, **not settled invoices**; an SDK `estimated=false` flag alone does not
prove that subsequent provider revisions are absent.

| Evaluation rental | Dedicated minutes | Nominal limit | Elapsed-rate estimate | Provider hour records |
|---|---:|---:|---:|---:|
| U initial, lost | 75.05 | $0.3125 | $0.31271 | $0.31457 |
| U update 200, succeeded | 44.11 | $0.3125 | $0.18379 | $0.18494 |
| M update 200, succeeded | 43.30 | $0.3125 | $0.18043 | $0.18169 |
| E update 200, succeeded | 39.06 | $0.3125 | $0.16277 | $0.16384 |

U's initial deletion receipt arrived about three seconds after its deadline;
its later provider record also exceeds the nominal limit. Thus these nominal
limits are retained control settings, not evidence that the final charge is
mathematically bounded by the same amount. The study's $10-per-arm allocation
remains the constraint; no apparent billing reduction authorizes extra work.

For M update 200, the controller retained only $0.00453 of rental billing while
the fresh provider read returned $0.18169. Its run-share field was lower still,
about $0.00113. U/M/E update-200 rentals remained alive for approximately
30.94/32.52/28.93 minutes after their successful commands finished, respectively.
Those intervals include evidence delivery and deletion. The old worker archives
remain unchanged; the bounded-delivery SDK repair applies to future launches.

The lifecycle records, both billing observations and exact read-only audit
sources are retained in
`experiments/data/curriculum-175/evaluation-costs/completed-rentals.json`.
These later observations never rewrite the prospective checkpoint-selection
ledger or select policies based on their evaluation scores.

The final report still requires genuine trained-policy results across all four
arms, paired whole-deal contrasts and uncertainty, deck/seat outcomes, exposure,
coverage and forgetting, and cumulative inclusive cost receipts. It will report
M−U, E−U, ME−U and ME−M−E+U without hiding invalid or missing cells. One historical
control seed, runtime maintenance and adaptive coverage safeguards limit causal
interpretation even when those results arrive.
