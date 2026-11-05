# MTG-244: rethink the hot path

New direction from Jack Heart in the primary conversation, 2026-11-04:
"keep optimizing", "needs more ambitious rethinking", "make it stupidly fast".
This supersedes a narrow objective of repairing the budget by incremental cache
tuning. Performance architecture within this layers Task is explicitly in scope.
Keep one Task/PR, exact rules semantics, complete legal choices, shared policy
observability, frozen inputs/evidence and the no-merge/no-experiment boundary.

## Outcome

Make ordinary step/search work cheaper by construction, not merely less slow.
Aim to outperform the pinned pre-layer engine on matched ordinary workloads
while adding correct composable layers. The unchanged 10% budget is a minimum
acceptance guard, not the optimization objective. Do not invent a promised
speedup or stop solely because a measurement falls just inside that guard.
Show measured gains and remaining costs; explain any justified diminishing
returns or tradeoff for review.

## Architecture investigation

Use actual profiles and deterministic operation/allocation counts to identify
which work should disappear entirely. Compare a small number of concrete
designs and implement the strongest supported approach. Investigate:

- Prederive immutable source/effect plans and conservative read/write summaries
  from the rules types. Avoid rewalking/cloning definition payloads on each
  revision. Summaries are optimization metadata, never a second rules authority
  or card-name-specific semantics. Discover granted effects and conditional
  reads completely; do not confuse timestamp competition with dependency.
- Partition independent effects and dependency-bearing components before
  expensive semantic checks. Rule out impossible dependency pairs cheaply;
  reserve dynamic before/after instruction comparisons for plausible pairs.
  Avoid whole-context clones and transitive closure over all effects after
  every application. Preserve actual dependency reevaluation and loop fallback.
- Separate state changes that alter the effect plan, affected membership,
  computed values and finalized payment outputs. Recompute only what a relevant
  mutation invalidates. Consider per-component/object work only where dependency
  tracking makes it correct; arbitrary local caches are not an improvement.
- Treat search forks as a primary path: share immutable plans/derived data,
  materialize only necessary changes, and avoid eagerly solving the whole board
  for a query that can be answered cheaply. Preserve branch isolation and real
  decision-boundary clones. No speculative rewrite of unrelated game storage.
- Remove allocation-heavy transient instructions and repeated scans where
  profiles justify compact storage, scratch reuse or specialized ordinary
  execution. A proven fast path belongs behind Game::characteristics and shares
  the same semantics; do not maintain divergent effective-property readers.

These are candidates, not a mandated data structure or implementation recipe.
Prefer removing work and simplifying ownership to piling on fragile caches.
Keep a clear reference computation/debug oracle and differential tests across
event-driven mutations, dynamic grants, source departure and dependency loops.

## Evidence and completion

Measure identical ordinary TLA/search workloads, dependency-bearing positions,
complete payments, and v8 model/encoding paths separately. Retain all full-score,
legal-action, simulation-count and step digests. Use the same-runner comparison
and disclose memory/allocation tradeoffs. No capacity cuts, action suppression,
fewer rollouts, disabled invariants or baseline resets to manufacture speed.

The implemented redesign uses shared immutable source plans/objects,
conservative read/write pruning and lazy SBA snapshots (`614a38fc8`, `68ef4a63d`).
Measurements and remaining acceptance live in the risk note. Incremental repairs and measurement checkpoint publication do
not establish completion. The same Task retains the isolated producer,
investigation and mitigation obligations until the accepted work is addressed.
Evidence uses bounded normal workloads, with no artificial load or contended
local timing as acceptance. Existing risk-validation and CI-repair obligations
remain in force.
