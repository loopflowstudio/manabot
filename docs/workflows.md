# Manabot work boundaries

Jack Heart accepted these four core responsibilities on 2026-11-04 in MTG-245,
with Academy separate. Tasks belong by their intended outcome, not by package.
Workflow installation grants no campaign, resource, spending or shipping authority.

| Responsibility | Owns | Default Graph | Completion evidence |
| --- | --- | --- | --- |
| Experiments | Scientific questions spanning model architecture, training regime and card pools; training and evaluation together | `experiment` | Frozen protocol, reconciled actual runs/failures/costs, analysis and supported conclusion |
| Coverage | Faithful cards/mechanics, shared observations/actions/tokens and rules defects | `coverage` | Working behavior, reusable decomposition, measured optimization and integrated batch handoff |
| Engine performance | Cross-cutting algorithms, allocation, caches, throughput and latency | `engine-performance` | Behavior-preserving gains on representative workloads with same-hardware comparison |
| Infrastructure | Shared training/evaluation systems, artifacts, interfaces, CI, developer tools and reliability | builtin `code` | Working maintained systems and their relevant workflow checks |
| Academy | Browser inspection, replay, human interaction and accessibility | builtin `code` | Correct viewer-safe product behavior through its real configured path |

A model comparison belongs to Experiments; checkpoint serialization reliability
belongs to Infrastructure. A new mechanic belongs to Coverage; a general state-copy
speedup belongs to Engine performance. Coverage optimizes its own implementation
without waiting for unrelated global optimization. Scientific results do not gate
native card support. Campaign failures retain their evidence while their fixes
belong to the appropriate implementation owner.

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
The [Standard milestone](research/standard-2026-11/milestone.md) keeps the deeper
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

Experiments' memory bytes are unchanged. PR #421 still edits the old Training
memory path in its own checkout: integrate those contributions into the renamed
path, preserving its reserved empirical decisions. The receipt binds both its
published head and untouched local memory bytes. Provider project/creation sync
remains pending; local membership is verified. MTG-210/211/230/231 retain their
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
`lf graph show coverage --json` and `lf graph show experiment --json` expose the
compiled nodes for review. Gate owns the affected acceptance suites; setup checks
never launch a scientific campaign.
