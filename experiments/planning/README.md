# Training planning records

The [accepted plan](../../docs/plans/strongest-manabot-training.md) is the
training decision; [research](training-evidence.md) explains the method ranking.
[grid.md](grid.md) and [grid.json](grid.json) reproduce all eight time × cash
cells. No quantity here is a strength measurement or execution authorization.

Run from the repository root (standard library only):

```sh
uv run --no-project --python 3.12 python scripts/reproduce_plans.py
uv run --no-project --python 3.12 python scripts/reproduce_plans.py --check
uv run --no-project --python 3.12 python -m unittest discover -s tests/planning
uv run --no-project --python 3.12 python scripts/budget_plan.py --format markdown
uv run --no-project --python 3.12 python scripts/budget_plan.py --window 1hr --band near_zero --deadline-seconds 1800 --cash-cap 5
```

`--inputs FILE` accepts a JSON object overriding fields in `Inputs` in the
calculator. All resolved inputs are copied into output. For example:

```json
{
  "local_hardware": "M4-Max-128GiB",
  "local_hours": 12,
  "electricity_low": 0.03,
  "electricity_high": 0.09,
  "inference_mode": "search_assisted",
  "inference_search_simulations": 64,
  "inference_p95_ms": 100,
  "inference_rss_gib": 1,
  "preparation": "Existing environment; no admitted data; count validation in setup"
}
```

Policy-only and search-assisted comparisons are separate deployment profiles.
Neither mode is a demonstrated latency or memory result. The visible width-64
policy remains the initial candidate; search-assisted serving requires the
separately declared inference budget and ETU-79 admission. Startup can be
replaced with `setup_seconds`; a window consumed by overhead is infeasible.
Exact time and cap overrides require one window/band. The chosen window supplies
its schedule template. A zero cash cap works only with zero incremental cost.

## Accounting and decisions

The calculator schedules a serial setup → generation → training → export →
evaluation loop. Allowances (seconds) before cash/availability clipping are:

| Window | Setup | Generation | Training | Export | Evaluation |
|---|---:|---:|---:|---:|---:|
| 10 min | 120 | 180 | 120 | 60 | 120 |
| 1 hr | 300 | 1500 | 900 | 120 | 780 |
| 1 day | 900 | 39600 | 18000 | 300 | 14400 |
| 1 week | 900 | 86400 | 72000 | 300 | 43200 |

These are editable assumptions in the small calculator, not measured durations.
Unused elapsed time stays unused. Local active time is clipped to availability
and worst-case electricity cash. Generation/training/evaluation are reduced
together when constrained; fixed overhead is never reduced. A feasible schedule
means the work envelope fits, not that the loop or a useful evaluation finishes.
Keep the incumbent if the complete loop cannot finish. Capacity intervals count
potential accepted work, not a guarantee of complete data/training cohorts.
Training exposures are clipped to available accepted training rows × at most ten
epochs and remain unknown when corpus size is unknown. Whole-game validation
can hold more or less than 10% of rows, so the accepted training-row count is
measured separately; optimizer repeats do not count as independent games.

Paid alternatives use CCX33 CPU hosts for generation/evaluation and one Lambda
A6000 for training; two CPU hosts in day/week cells, one in short cells. CPU
hosts have four workers each, GPU training one worker, all with one Torch thread
per worker. Local training uses one worker; local generation/evaluation use the
configured worker count. Each rental bills 900 seconds startup and 120 seconds transfer
before work; no GPU is held idle while teacher games run. Export/load checking
uses the local host. Hardware classes and aggregate concurrency must match
profiles; adding hosts does not multiply a measured rate. Host-hours include
startup/transfer. Summed host-hours differ from elapsed time when hosts run
concurrently. Billed host-hours additionally round each complete rental
(startup plus work) up to [Hetzner's hour](https://docs.hetzner.com/cloud/billing/faq/)
or [Lambda's minute](https://docs.lambda.ai/public-cloud/billing/). The work row
carries the rounding charge; it buys no extra work or elapsed time. Cash clipping
includes these increments and taxes before protecting the reserve. Rentals
must be deleted/terminated after each phase; powering a server off does not
end billing. The CPU quote assumes IPv6 access; paid IPv4 needs a new allowance.
Availability is total local active time; it does not model an
hour-by-hour calendar. Supply a deadline that fits actual access windows.

Costs use the retained dated [price snapshot](prices.json), low quoted rates
and high planning ceilings. Local power sensitivity is $0.03–0.09/hour; sunk
purchase cost is excluded and opportunity cost remains separately unpriced.
Rental storage and transfer are explicit $1 allowances each and rental tax is
10%; override them for actual quantities/jurisdiction. Bundled disks are not
billed again. Requote and verify capacity before execution. No spot discounts,
credits, monthly caps or free setup are assumed. The 20% reserve is never spent
automatically; failed jobs require a new plan including their sunk cash/time.

Each paid cell keeps the corresponding local-only alternative at the same
elapsed/cash cap. The lower-cash local cell is also separately reported. A paid
alternative is selected only if all work metrics have matching profiles,
its lower bounds are at least the local upper bounds, at least one is larger,
and no failed profiling attempt remains unresolved. Otherwise leave the money
unspent. This deliberately conservative rule ranks capacity, never strength.
It may miss worthwhile cloud jobs; a complete matched-work pilot can revise it.

Width-64 search distillation is the first attempt in every cell. Day/week cells
seek repeated independent seeds only after observed per-attempt cost fits.
The weekly extension compares distillation, scratch self-play, and distillation
followed by self-play at matched total cost. Those are alternatives for the next
bounded increment, not three full runs silently charged to one schedule.
Use the self-play settings and consumers in the accepted plan. Freeze the next
schedule after profiling; rank candidates by held-out complete-game results.
No league, new planner family, or cloud scheduler is introduced.

## Consuming ETU-79 evidence

ETU-79 owns actual world, pipeline, configuration, store and admission semantics.
No accepted receipt is supplied here. `--evidence FILE` accepts a small summary
of its existing receipts, not an alternative receipt store or authority. Set
`world_identity`, `pipeline_identity`, and `config_identity` in inputs to the
exact admitted values (including rules/content/extensions, Lesson pools,
observation/action ABI, recipe/model/seed/split settings). Pending identities
never accept profiles. Summaries are trusted local inputs; this calculator does
not independently authenticate receipt bytes or perform world admission.

The summary shape is `{"profiles": [{"binding": ..., "attempts": [...]}]}`.
Copy `profile_binding` from the desired generation/training/evaluation phase in
JSON output, then populate it from the actual receipt identities. It binds
hardware, host count, workers, Torch threads, phase, units and deployment budget
as well as world/pipeline/config. Equality is exact; mismatches leave volumes
null. There is no cross-hardware extrapolation. Each attempt has this shape:

```json
{
  "receipt_sha256": "<64 lowercase hex characters from the existing receipt>",
  "status": "complete",
  "productive_seconds": 123.0,
  "counts": {"accepted_labels": 456, "accepted_training_rows": 400, "complete_games": 7}
}
```

Generation counts accepted labels, accepted training rows after the whole-game
split, and complete games; training counts
`training_exposures`; evaluation counts `scored_paired_deals` (one four-game
block covering both decks and starting seats). Failures use `status: "failed"`
and retain their distinct source receipt digest. Keep every bounded attempt,
including failures; the calculator reports their count and does not select a
paid profile with failures. Full failure costs/outcomes remain in source
receipts. At least three complete attempts produce min/max aggregate rate
sensitivity, not a confidence interval or guarantee. Counts must be whole,
nonnegative units. Productive seconds exclude the separately budgeted startup,
transfer and export; they include in-phase I/O and stalls. End-to-end rate
profiles are rejected to prevent subtracting overhead twice.

## Release and measurement gates

Before any execution: obtain separate authorization, ETU-79 corrected-world
train/export/play receipts, a valid quote/capacity envelope and deployment
latency/RSS. Then fix data and checkpoint identities, complete legal-offer
coverage, held-out deals, deck/seat matrix, training seeds, matched inference
compute, sample count and analysis. Stop on ABI/world drift, hidden leaks,
truncation, nonfinite training, replay mismatch, cap hits above 1%, failed
quotes/memory/throughput or insufficient remaining evaluation time/cash.
Retain all failed/incomplete games. Successful short runs establish feasibility;
strength and human-play claims need the protocols in the accepted plan.
