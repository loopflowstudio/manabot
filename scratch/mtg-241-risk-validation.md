# MTG-241 — measure and reduce performance and learning risks

Jack Heart directed this in the Task conversation on 2026-11-04: “Let's make
sure to both measure those risks and attempt to ameliorate them.” This extends
the existing acceptance work. The live code Graph retains implementation
ownership; review remains the boundary. No training experiments or merge are
authorized by this direction.

## Performance evidence and mitigation

Use the existing bounded, same-runner Engine Budget procedure and directed
fixtures, never artificial load. Record exact revisions, configuration, counts
and digests. Separate unchanged-work comparisons from rules-driven trace changes.
Include a main/reference-to-candidate comparison on shared supported workloads;
successive already-modified cuts alone cannot establish net cost. After ordinary
integration, disclose inherited layer/payment costs separately from MTG-241's
incremental costs and remeasure the combined head against the standing budget.

- Measure ordinary step time and search total/tail time; include the common
  no-replacement path and representative nested-choice/replacement positions.
- Attribute demonstrated overhead to fork/clone work, retained frame/batch
  storage, replacement candidate discovery/rechecks, trigger snapshots and
  observation projection. Use bounded profiling or operation/allocation counts
  where supported; state measurement limits rather than inventing precision.
- Report legal branching and decisions per fixed scenario alongside search
  nodes/work at a fixed budget. Separate extra legitimate decisions from slower
  execution of equivalent work. Do not truncate choices to improve a benchmark.
- Attempt targeted mitigations where measurements identify costs: ownership
  transfer instead of redundant clones, eliminate repeated collection/snapshots,
  and keep no-replacement execution cheap. Cache only demonstrated repeated work
  with correct revisions and debug oracles. Preserve complete legal choices,
  simultaneous events and exact clone semantics. Remeasure affected cases after
  mitigation; retain before/after and unresolved costs. The configured 10%
  Engine Budget remains in force; moving a reference does not itself mitigate.

## Learning-facing evidence and mitigation

- Construct paired pending states with identical present offers but different
  consequential iteration progress, selected/result bindings or completion
  events. Check that the successor policy input distinguishes viewer-permitted
  differences, with native/Python parity and actual model-input coverage.
  Complete missing projections and reconcile schema ownership with MTG-244.
- Test the inverse information boundary: changes only to unavailable hidden
  information must not reveal it in observations or offers. Exercise retained
  selected subsets and caller frames through hidden-world materialization;
  preserve known exact memberships and coherently refresh unknown commitments.
- Exercise real Commands, divergent forks, replay and reconnect around nested
  selection/replacement/payment pauses. Assert one cost payment, one logical
  commit/finalization and correct trigger/SBA/priority timing, including the
  actual linked-return entry-replacement payment witness.
- Check capacities/overflow and reject unsupported input versions explicitly
  before inference; never silently erase pending state. Preserve frozen widths,
  saved omissions, evidence and world identities.
- Use bounded deterministic tests and configured model checks to establish
  representation and execution properties. These do not establish convergence,
  win rate or sample efficiency; no new training campaign is implied.

At demo/review, summarize each risk with its witness/metric, original result,
mitigation, final result and remaining limitation. Identify exact tested heads;
earlier focused passes do not validate later edits. Keep residual risks visible
and scoped rather than treating the full suite as proof of all interactions.

Current correction (2026-11-04): `91517292c` repairs completed private captures;
`d77bc8676` integrates the published v8 layer schema. Eleven focused hidden-selection
tests cover both viewers, all six capture lifetimes, aliases, moved captures,
privacy, expired control, departed incarnations and selections from Remembered. These establish execution/
representation properties of binding ownership, not successor policy completeness.
The final alias correction over `d77bc8676` additionally updates selections from
Remembered and their legal offers; the restored Command witness passes in both
samplers. Five focused Python checks pass after atomic extension rebuild.
The plan's CR 614.16/event-creation correction is a new prerequisite for complete
retained-payload separation. No new engine timing, model-cost or learning-efficacy
claim is made, and the inherited MTG-244 regression remains unresolved evidence.
