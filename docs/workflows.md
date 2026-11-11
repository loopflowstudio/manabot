# Manabot work boundaries

Jack Heart accepted these four core responsibilities on 2026-11-04 in MTG-245,
with Academy separate. Tasks belong by their intended outcome, not by package.
Workflow installation grants no campaign, resource, spending or shipping authority.

| Responsibility | Owns | Default Graph | Completion evidence |
| --- | --- | --- | --- |
| Experiments | Scientific questions spanning model architecture, training regime and card pools; training and evaluation together | `experiment` | Frozen protocol, reconciled actual runs/failures/costs, analysis and supported conclusion |
| Coverage | Faithful cards/mechanics, shared observations/actions/tokens and rules defects | `coverage` | Working behavior, reusable decomposition, measured optimization and integrated batch handoff |
| Engine performance | Native engine algorithms, allocation, caches, throughput and latency | `engine-performance` | Behavior-preserving gains on representative workloads with same-machine comparison |
| Infrastructure | Shared training/evaluation systems, artifacts, interfaces, CI, developer tools and reliability | builtin `code`; `experiment-performance` for training/evaluation speed | Working maintained systems and their relevant workflow checks |
| Academy | Browser inspection, replay, human interaction and accessibility | builtin `code` | Correct viewer-safe product behavior through its real configured path |

A model comparison belongs to Experiments; checkpoint serialization reliability
belongs to Infrastructure. A new mechanic belongs to Coverage; a general state-copy
speedup belongs to Engine performance. Coverage optimizes its own implementation
without waiting for unrelated global optimization. Scientific results do not gate
native card support. Campaign failures retain their evidence while their fixes
belong to the appropriate implementation owner.

## Training and evaluation performance

For a slow training update, start the performance workflow from the repository
checkout with Loopflow installed:

```sh
lf graph experiment-performance
```

The workflow measures the update, selects a bottleneck, then loops through
implementation, compression, refresh, concept review and performance review.
It retains the publish, demo, delivery and verification boundaries of
`engine-performance`. Its result is a software change with supported timing
evidence, or a documented reason to stop when no useful improvement is supported.

Use `experiment-performance` when the question spans collection, policy inference,
learner work or evaluation matches, including how these costs change with model
size. Use `engine-performance` for a bottleneck within the native engine, such as
state copying or search. Coverage continues to compose `engine-optimization`.
The performance workflow does not change Wave defaults or replace a Task's
captured graph.

The repository skills `experiment-measure` and `experiment-performance-review`
own the measurement plan and its review. Once the standing size-ladder benchmark
and retained baseline ship, measurement uses their documented command as its
reference; the current per-recipe benchmark below remains usable today.

Start with retained receipts and the **Update time** Perf series. Read collection's
native stepping, encoding and policy inference beside learner time; retain
inference calls and rows per call so batching changes remain interpretable.
Distinguish one seed's progress from aggregate throughput or cost: packing can
improve the latter while slowing every seed. Compare overlapping post-warmup
packed samples to matched solo ordinals, keeping both latency and throughput.
`TrainingRun.host` supplies CPU and quota facts; keep GPU, source, world, input,
precision, model size and workload identities with each comparison. Read
evaluation's `performance.json` alongside training to show its share of the
workload's time. Evaluation clocks can overlap and exclude preparation and billing;
do not sum them or label elapsed time as rental cost.

For a bounded training reading on an available, authorized machine with the
[development environment](development.md#environment-and-commands) prepared:

```sh
mkdir -p .runs/update-baseline
uv run python scripts/bench_training_update.py --streams 256 --updates 3 \
  --out .runs/update-baseline/run --json .runs/update-baseline/receipt.json
```

This prints seconds, decisions and games per update, host facts and phase timings
in microseconds per decision. It retains checkpoints and per-update metrics under
`--out`; without that option the run directory is temporary. Keep baseline and
candidate in separate `--out` and `--json` paths, with matching input capacities
and matching, explicit `--recipe`, `--seed` and `--device` settings. `--recipe` accepts a
regime JSON or `module:function`; the device defaults to CPU (`--device cuda`
selects CUDA). The benchmark pins its in-memory recipe to the current world, so
it cannot reproduce frozen experiments. Capacity failure requires a compatible
declared workload, not truncation or edits to a frozen recipe. The default CPU
reading plays three full updates (about 160,000 decisions); on a laptop expect
minutes per update and several GB of memory.

The printout omits batching. In the receipt's stage, the `calls` of
`collection_timings["Policy inference and action selection"]` are inference
calls; inferred rows are its `environment_decisions` (streams × calls −
`idle_stream_steps`), so rows per call is decisions / calls. Idle steps are
complete-game streams waiting for the remaining games; they lower rows per call
without indicating a serving limit.

After an interrupted run, read the retained per-update rows: receipt timers can
cover only the last attempt while counts span attempts. Keep failed attempts and
missing measurements visible.

Use `--cuda-counts PATH` in a separate diagnostic run for kernel launches and
synchronizations. It requires real CUDA and changes timing; its whole-run counts
are not phase-specific counts. `scripts/diagnose_training_transfers.py` provides
bounded retained-input probes for recording and transfers, not whole-update speed
evidence. For evaluation, use the predeclared plan and receipt guidance in
[performance evidence and calibration](evaluation.md#performance-evidence-and-calibration).

Compare baseline and candidate on the same machine with matched workloads and
behavior digests. A GPU label alone does not identify a comparable host; the
[training update investigation](../experiments/training-update-time.md) records
substantial host variation. Never induce artificial load. If required hardware is
unavailable, record the exact command and missing hardware as a deferred check;
do not rent automatically, substitute CPU timing for CUDA evidence or claim a gain.

If behavior digests diverge unexpectedly or the workloads do not match, the
comparison cannot support a speed claim. Revise the comparison plan, repair any
behavior regression and repeat the affected measurements within existing authority.

Fewer seconds per update does not establish faster learning. Changes to precision,
policy lag or batch composition need separate learning evidence. This software
workflow produces no such evidence and grants no campaign, resource or spending
authority. Scientific comparisons remain in `experiment`; frozen worlds,
observation versions and recipes remain unchanged.

## Coverage

`coverage` is a narrow fork of the builtin code lifecycle. Three distinct phases
compose implementation, compression, builtin refresh, concept review, a domain
review and a backward edge to that phase's implementation:

1. `coverage-working`: complete shared rules behavior and directed debug evidence.
2. `coverage-scalable`: remove special paths; demonstrate that real cards compose
   from shared rules types and one owner for each fact.
3. `engine-optimization`: measure bounded representative workloads, improve
   worthwhile bottlenecks, retain debug invariants and behavior digests.

The whole batch then has one publish/demo/gate/delivery/verify boundary. These are
three code loops, not three required PRs. A semantic counterexample invalidates
dependent evidence and requires repairing the working contract. Passing the 10%
regression allowance alone does not establish that optimization is finished.
Artificial load and narrowed choices/observations are forbidden.

Card authors supply interactions, synergies, directed behavior evidence and a
committed integrated snapshot. A separately owned validation outcome reads those
artifacts and authors the functional decks. The graph does not automatically
create that Task, start its campaign or complete it on the card author's behalf.

## Experiments

`experiment` has one lifecycle: design → design review → optional software →
final freeze → execution review → bounded execution → analysis → evidence review.
Architecture, regime and card-pool coverage are lenses used within its design and
analysis, not three obligatory campaigns. Existing checkpoints can go directly
to evaluation without placeholder training or a placeholder PR.

The optional `experiment-software` path composes builtin `pursue` and `deliver`,
retains demo, and uses a narrow integration verification skill. Builtin `verify`
can complete a Task; this variant verifies software without completing the parent
experiment before its empirical obligations. The ready path is empty. Source and
protocol freeze occur after software integration on either path.

Execution review checks actual authorization and accepted settings, not just the
existence of a budget. Missing authority stops headless execution without jobs,
games, rental or spending. A retry reconciles existing jobs and artifacts first;
it cannot silently duplicate a campaign. Evidence review returns to execution only
for a correctable missing portion still covered by the same protocol and authority.
Changed hypotheses/source/bounds require prospective redesign and review.

The first growing-pool instance uses [the MTG-245 protocol](../experiments/protocols/growing-pool-245-v1.md):
A's reference league, authored functional decks, frozen tickets, stable anchors
and explicit splits. Lists, shares and budgets remain proposals until accepted.
Training exposure is separate from evaluation draws and committed action counts;
the latter never substitute for training capture. Task 04 owns actual first-pilot
execution/results and Task 08 Validation B on the expanded mechanism cohort.
The [Standard milestone](../.lf/wave/coverage/research/standard-2026-11/milestone.md) keeps the deeper
strength/generalization track separate, first after full FDN. Tiny runs may
prove execution while leaving learning inconclusive.

## Installation and migration

Repository graphs live in `.lf/graphs/` and domain skills in `.lf/skills/`.
Builtin `code`, `pursue`, `refresh` and `deliver` are composed, never shadowed.
The legacy `.lf/flows/` files are historical Task-specific definitions and are not
part of the installed graph catalog. They are not a second implementation of the
new workflows and are not silently converted into new live Tasks.

Shared defaults use `graph:` in each Wave's `config.yaml`. Personal `lf set`
values override them. Tasks already holding captured graphs retain those graphs;
changing a default does not replay or replace their live work. Check actual
selection with `lf wave graph show <wave>` and `lf wave status <wave> --json`.

Migration uses supported `lf wave rename`, `lf wave add` and, once a membership
mapping is settled, `lf task move --wave`. These operations preserve identities;
never edit Loopflow's database or create replacement Tasks/PRs. Lifecycle commands
write/commit Wave files on the main checkout, so inspect their receipts and carry
those exact moves into the delivery branch without altering other work.

The [dated migration receipt](workflows/2026-11-04-migration.json) retains both
migration phases. Jack Heart selected Training as Experiments' surviving identity
on 2026-11-04. Supported operations renamed it, moved Eval's two open Tasks, then
retired Eval. Task/PR identities and captured Graph process identities are
unchanged. Eval's chapter and completed Task remain readable with
`lf wave show 6b12128f-ecde-4982-ab80-04210e4108f1 --json`; its memory is retained at
`9f42e4e58a88efab347814e6f0c7d83d9bf98a0b:.lf/wave/eval/MEMORY.md`.

At migration, Experiments' memory bytes were unchanged and PR #421 still edited
the old Training path. The receipt binds its then-published head and untouched
local memory bytes. Subsequent upstream commit `fe83a6b06`, integrated here through
`e5c3dd947`, updated Experiments' memory with an exact Eval archive reference and
TLA recovery lessons while retaining the shared-effect evidence and reservations.
This does not certify integration of later contributions from PR #421's checkout.
Provider project/creation sync was pending in the migration receipt; local
membership is verified. MTG-210/211/230/231 retain their
existing memberships pending a separate outcome-based mapping. Academy remains
separate, and no moved Task was executed by this migration.

A concurrent Task created a new, empty Training Wave while this migration ran.
After inspecting its provenance and Jack Heart's reconciliation steer, fc6ffb
(the closure/lambda comparison) moved to Experiments with its live Graph,
comments and separate $25 authority intact. The empty replacement Wave was then
retired through `lf`; its chapter remains historical. The final shared read has
Experiments, Coverage, Engine performance, Infrastructure and Academy active;
the older abandoned Study Wave remains historical.

Validate discovery, all skill references, distinct backward edges, single delivery
and the optional software/experiment boundaries without launching agents or games:

```sh
uv run python scripts/check_loopflow_workflows.py
```

This check uses the installed `lf` compiler and catalog. It reports a missing CLI
or reference as a failure rather than treating missing tooling as an empty catalog.
`lf graph show coverage --json`, `lf graph show experiment-performance --json`
and `lf graph show experiment --json` expose the compiled nodes for review. Gate
owns the affected acceptance suites; setup checks never launch a scientific campaign.
