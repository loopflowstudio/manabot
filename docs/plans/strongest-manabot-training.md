# Strongest playable manabot: time × money

ETU-80 · Intelligence / Trained Challengers · human design review, 2026-09-24

**Status: approved by the User on 2026-09-24** ("approved"), after reviewing
the eight-cell time × money grid. Implementation follows this design and its
explicit configurable assumptions. Approval covers the planning deliverable;
it does not authorize provisioning, paid experiments or production training.

## Decision

Plan eight scenarios: **10 minutes, 1 hour, 1 day and 1 week**, each with
**approximately $0 (strictly less than $10)** or **at most $1,000** in cash.
This grid supersedes the original four budget tiers and the intermediate
$10/$100 focus. The User requested this simpler framing explicitly.

The goal is the strongest supported playable manabot for the corrected
Allies/Lessons world within each cell's time and money limits. A budget is a
cap, not a target. Start with a small policy distilled from search; compare
self-play where enough time and money remain for a useful experiment.

This task produces the plan and a small reproducible calculator. It does not
provision machines, run training or authorize spending. ETU-14/31/55 remain
retired. ETU-79 owns the corrected training-to-play pipeline and its profiles.

## The grid

These are starting choices and intended deliverables, not measured completion
or strength forecasts. If a cell cannot finish its complete loop, return the
measured limit and retain the incumbent instead of claiming a new challenger.

| End-to-end time | ~$0 (<$10): existing Mac | ≤$1,000: local plus paid compute where useful |
|---|---|---|
| **10 min** | Tiny training → export → load/play check. Use an admitted dataset if one exists; otherwise attempt a tiny fresh corpus. Establish feasibility. | Start with the same local path. Rent only if provisioning and the full useful job demonstrably fit; money may add nothing at this deadline. |
| **1 hr** | Small search-generated corpus → one distilled policy → bounded full-game evaluation. Preserve the checkpoint even if the result is only feasibility evidence. | Accelerate the measured bottleneck with one suitable rented host if setup pays back. Aim for a larger corpus or repeated training attempts, not a broad sweep. |
| **1 day** | Generate data, distill a width-64 policy, and evaluate both deck/seat assignments; repeat training if it fits. | Parallelize independent teacher games on CPU hosts, train independent seeds on one GPU, and compare against the incumbent on held-out games. Aim for three seeds if the measured budget fits. |
| **1 week** | Reuse the fixed corpus, repeat training and evaluation, then add fresh data only if it improves the candidate. Bound active local hours to keep cash below $10. | Compare distillation, self-play from scratch and distillation followed by self-play at matched total cost. Fund the observed winner in bounded increments; retain the best admitted checkpoint and held-out final evaluation. |

Every paid cell also evaluates the corresponding local-only option. Spending
must buy accepted data, a useful additional attempt, better evaluation, or
shorter completion time. Without evidence of that benefit, leave it unspent.
No predicted Elo or guaranteed score belongs in this table.

## Meaning of the axes

- **Time is elapsed wall-clock from starting the run to the playable export
  and its evaluation.** Include setup, provisioning, downloads, generation,
  training, export and evaluation. Parallel host-hours are accounted separately.
  Preparation done before the clock must be declared; it is not silently free.
- **Money is total incremental cash**, including electricity, rented CPU/GPU
  time, idle time, storage, transfer, taxes and failed attempts. Existing
  hardware purchase cost is sunk. Report opportunity cost separately.
- **~$0 means less than $10**, not literally free compute. Use the existing
  M4 Max / 128 GiB where available. A one-week deadline does not imply seven
  days of uninterrupted free Mac time. Clip active hours to both availability
  and the cash cap. Power/tariff assumptions remain explicit until measured.
- **≤$1,000 is an upper bound at every duration.** A ten-minute plan should not
  spend more just to consume that allowance. No credits or spot discounts are
  assumed. The planner accepts an exact cap within either displayed band.

Common starting assumptions: width-64 policy, four local CPU workers, one
Torch thread per worker, CPU inference. The provisional deployment envelope
is p95 ≤100 ms/decision and process RSS ≤1 GiB. Policy-only versus allowing
search during play remains an unresolved product choice, not a User-confirmed
restriction. Keep it an explicit planner input and separate the resulting
comparisons by deployed compute.

## Why these recipes

The [primary-source research](../../experiments/planning/training-evidence.md) remains the
supporting record. The grid is the decision surface; the literature is not
another deliverable to expand indefinitely.

- **KataGo:** balance data diversity, search quality and reuse. Separate
  generation, learning and export so rented accelerators do not wait for data.
  Its ablations motivate pilots, not imported throughput multipliers.
- **AlphaStar:** useful initialization data and opponents matter. Its large
  replay curriculum and league are not available for free here. Retain frozen
  opponents and measure forgetting; do not build a league for this task.
- **AlphaZero/MuZero:** search can supply policy targets. Use the exact engine
  already available. Relabelling costs searches and creates no new independent
  games; learning a replacement rules model is outside this plan.
- **DeepNash, neural CFR and modern offline/model-based RL:** retain their
  evidence and transfer limits in the research notes. They do not provide an
  existing selected-match train/export path here. No new algorithm family is
  hidden inside a budget cell.

Repository evidence favors trying search distillation first, but historical
results have world and shaping confounds. The cheapest falsifier is a bounded
comparison with self-play at equal total cost. More expensive search targets
must earn their cost in complete-game student performance, not training loss
alone. No transferable current-world pretrained policy is assumed.

## Starting configuration and existing consumers

Use the existing consumers and ETU-79's accepted binding; do not create a
second world schema, checkpoint loader, evaluator or receipt store.

| Stage | Existing path | Binding or check still needed |
|---|---|---|
| Corrected world | `Hypers`, `Match`, ETU-79 runner | Exact rules/content/extension and Lesson-pool identities; both assignments; complete legal offers |
| Teacher | `selected_branchdriver_teacher.run_teacher_game`, `sim/mcts.py` | Complete-game shard export with observations, semantic offer order, visits, Commands and source receipts |
| Distillation | `sim/search_supervised.train_search_supervised` | Corrected observation shape, immutable whole-game split and accepted target rows |
| Self-play | `SeatRoutedCollector`, `NetOpponentTrainer` | Corrected match binding and actual device/collection/update smoke |
| Export/play | `save_bc_checkpoint`, `Trainer.save`, `CheckpointVillain` | Current loader acceptance and complete games through the demo adapter |
| Evaluation | Existing arena registrations, replay and statistics | Replace/bind the hardcoded interactive mirror in `arena/match.py:play_cell` |

Defaults and historical scripts still use vanilla or interactive mirrors.
The observation encoder can truncate legal-action tails; admission requires
zero omissions. BC trainers construct default `ObservationSpace()`. The
self-play collector's CPU RNG needs actual CUDA validation before a GPU plan
is accepted. These are integration gaps, not measured failures fixed here.

**Distillation starting point:** `AgentHypers(hidden_dim=64,
num_attention_heads=4, attention_on=True)`, observation-only. Uniform-prior
determinized PUCT with 64 total simulations across four compatible worlds,
random terminal continuations, 2,000-step continuation cap. Record cap hits;
more than 1% rejects that search setting for scaling. Train complete visit
distributions with policy weight 1, value weight 0, Adam LR 1e-3, batch 512
(128 local), at most ten epochs and a fixed 10% whole-game validation split.
Stop sooner when the time envelope requires it. No hidden-truth inputs.

**Self-play alternative:** terminal +1/-1 rewards, no shaping, 16 environments
(four local), 128 learner transitions per stream, LR 2.5e-4, gamma .99,
GAE lambda .95, four minibatches/four epochs, clip .1, entropy .01, value
coefficient .5 and gradient norm .5. Count warm-start data/training in a
continuation's total cost. These are pilot settings, not measured optima.

Keep globally unique game identities and related deals in the same split
across seeds. Repeated optimizer exposures are not new games. Count every
attempt, including failed jobs and incomplete games. Restarted training is a
new attempt: current saves do not contain complete collector/environment/RNG
state for exact resumption.

## Small planner

Inputs: exact elapsed-time limit, exact cash cap, local machine/available
hours/power cost, deployment envelope, dated prices and optional world-bound
throughput evidence. Output the eight grid cells and their local/paid
comparisons in readable Markdown and machine-readable JSON.

Each cell reports only the information needed to act:

1. Recipe, model, data source/reuse, machine class and phase concurrency.
2. Scheduled wall time, billed host-hours, expected cash range, protected
   failure reserve and unspent money. Keep 20% of the cap protected; count
   ancillary charges and taxes explicitly instead of calling them compute.
3. Measured intervals for accepted labels, complete games, training exposures
   and scored paired deals, or null where measurements are missing.
4. Intended playable deliverable, the largest uncertainty and the next cheap
   measurement that could change the choice.

The [calculator](../../scripts/budget_plan.py) and
[reproducer](../../scripts/reproduce_plans.py) implement the eight-cell grid.
[Usage and evidence inputs](../../experiments/planning/README.md) describe
explicit phase schedules, startup assumptions and local/cloud comparisons.
The [generated grid](../../experiments/planning/grid.md) and
[machine-readable output](../../experiments/planning/grid.json) retain null
volumes until accepted profiles exist. Schedules are bounded work allowances,
not promises that a complete training or evaluation cohort will finish.

For a phase at a measured concurrency:

```
cash = billed_host_hours * quoted_host_rate + applicable_ancillary_costs
work_interval = productive_wall_seconds * measured_aggregate_rate_interval
```

Use either an end-to-end rate including overhead or an explicit overhead
allowance with a productive rate; never subtract startup twice. Do not infer
multi-host throughput by multiplying a single-host rate. A fixed per-phase
local/cloud assignment is sufficient; charge its transfer and startup costs.
Round each rental's startup plus work to the provider's billing increment,
separately from occupied host-hours and elapsed time. The selected CPU profile
uses hourly billing and IPv6 access; the GPU profile uses minute billing.
Terminate each rental after its phase; powered-off servers can still be billed.

Use the retained [price snapshot](../../experiments/planning/prices.json) as dated reference,
not guaranteed quotes or reserved capacity. Its CPU/GPU comparison profiles
are CCX33 and A6000; L4/4090 remain alternatives when current quotes and the
same workload profile justify them. Requote before paid execution. Hardware
classes, availability and concurrency must match the evidence used.

Rate evidence binds world, pipeline/config, hardware, concurrency, units and
receipt digests. Preserve all bounded profiling attempts, including failures.
At least three complete attempts provide a sensitivity interval, not a
confidence interval or worst-case guarantee. Short cells may consume existing
profile evidence; they must not pretend to perform all calibration anew.

## Measurement and evaluation

The dated ETU-78 read contained no accepted corrected-world throughput
receipt. ETU-78 was subsequently superseded by ETU-79; the latter owns
repeatable corrected train/export/play and its provenance, timing and cost
receipts. The implementation checkout contains no accepted ETU-79 receipt. Until available, every cell has unknown game volume and strength.
Do not substitute engine SPS or old-world wins. Resolve ETU-79's receipt
before extending the production binding or paying to repeat profiling.

The first discriminator is the smallest complete training → export → play
loop on the intended device. Then compare label cost/quality and actual
local-versus-rented work per dollar. Use these outcomes to choose each cell's
schedule. In very short cells, successful completion is feasibility evidence;
it is not automatically a strength result.

Keep training, development and final evaluation deals separate. Freeze the
incumbent, candidate identities, training seeds, deck/seat matrix, time/cost
caps, inference compute and analysis before scoring. A paired block comprises
four games covering both decks and starting seats. Crashes and timeouts are
failures, not draws or silently replaced samples. Valid draws score 0.5.

Use independent training seeds for method claims; report each seed and
cross-seed uncertainty, with within-seed paired-game uncertainty separately.
Three seeds are an initial target when feasible, not a promise every cell can
fund. Freeze sample counts from measured cost before evaluating. Preserve
competence, legality, information-safety and per-deck failures alongside scores.
Choose checkpoints on development data and score the chosen rule on untouched
final deals. No guaranteed dollar-to-strength curve.

Stop or replan on world/ABI drift, omitted legal choices, hidden-information
leaks, nonfinite training, replay mismatch, quote/throughput/memory failures,
or insufficient time/cash to complete the promised evaluation. Do not consume
failure reserve automatically. A negative result or infeasible cell is useful
output. Human-play claims require a separately agreed human protocol.

## Done when

The durable plan and calculator reproduce **all eight cells**, distinguish
elapsed time from host-hours, include overhead and reserve, honor the strict
<$10 boundary, and leave missing volumes/strength unknown. Checks cover short
windows consumed by startup, local electricity/availability limits, mismatched
world/hardware evidence, and no benefit from extra spending. Every recipe maps
to existing consumers with its remaining binding gaps named.

The accepted plan lives in `docs/plans/`, research/planning records in
`experiments/planning/`, and the calculator in `scripts/`. Obsolete generated
tier outputs and duplicate scratch planners are removed.
Training execution remains separate. This review establishes the grid;
implementation must not claim the old twelve-scenario check proves it works.
