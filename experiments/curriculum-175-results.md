# Curriculum adapters: interim results

Observed 2026-10-27T11:38:53.947448+00:00. **The final four-arm effects comparison is not
yet available.** This interim report records the first complete U/M/E development
results and the retained initial-evaluation failure while the experiment continues.

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
The earlier U/E snapshot remains in `early-outcomes.json`. U/E jobs have now
succeeded; M was still delivering its completion receipt at the observation.
Delivered valid games are retained independently of that operational status.

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

Training status below is the 11:36 UTC snapshot; the live dashboard updates
independently. The scores above use the later frozen-data read.

| Arm | Training at 11:36 UTC | Update-200 evaluation |
|---|---|---|
| U | Reused MTG-183's genuine window through update 1,325 | Succeeded; 400 valid games |
| M | Running, update 399 | 400 valid games delivered; final receipt pending |
| E | Running, update 419 | Succeeded; 400 valid games |
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
Its incomplete initial eval had a retained rental ceiling of $0.3125. The
update-200 capacity allowance is at most $0.34 per arm. The retained M/E training
rental ceilings are $8.9975 each; prior startup charges and the full $0.50 reserve
remain included in each arm's $10 limit. ME requires a fresh bound before its
evaluation is submitted. No unused allowance moves between arms.

Live provider billing revised U's elapsed-rate estimate downward when a
closed-hour record arrived. These amounts remain provisional, not invoices;
the lower estimate is not treated as extra spending permission. SDK run-share
cost ends at command completion and can omit the paid evidence-delivery tail.
Final inclusive costs must include each dedicated rental through confirmed
machine deletion, plus setup failures and earlier evaluation charges. For
example, U's completed update-200 rental spanned about 44 minutes, including
final delivery, for a $0.184 elapsed-rate estimate at $0.25/hour; its retained
$0.3125 rental ceiling remains the spending bound, not that provisional estimate.

The final report still requires genuine trained-policy results across all four
arms, paired whole-deal contrasts and uncertainty, deck/seat outcomes, exposure,
coverage and forgetting, and cumulative inclusive cost receipts. It will report
M−U, E−U, ME−U and ME−M−E+U without hiding invalid or missing cells. One historical
control seed, runtime maintenance and adaptive coverage safeguards limit causal
interpretation even when those results arrive.
