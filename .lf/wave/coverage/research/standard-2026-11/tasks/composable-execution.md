Composable rules execution, selection and replacement decisions

Jack Heart authorized implementation and launch on 2026-11-04 as one of three
engine improvement Tasks ahead of broad Standard authoring. Produce a reviewable
PR using the code Graph; stop at its interactive review boundary. This is not
authorization to merge or launch training experiments. A separate layers Task
was subsequently authorized; it owns characteristic-layer algorithms.

Outcome: effect composition can pause inside nested operations, preserve exact
objects/results, and continue once without losing choices or changing when
priority, state-based actions and trigger placement occur. Shared selection and
replacement machinery replaces overlapping card-shaped execution paths.

Acceptance philosophy: a generally working, trainable engine is valuable; bugs
are expected. Prioritize systematic distortions, training failures and issues
that could invalidate learning conclusions. Do not require exhaustive Magic
conformance or revoke defined cards' playability. Preserve retained evidence and
scope any changed semantics to the versions and conclusions affected.

Source findings at 5dd11cec4 (recheck current main before edits):
- flow/resolution.rs ForEachTarget builds local subframes and debug-asserts
  children return no Decision; release discards a returned decision.
- flow/decision.rs EffectFrame stores one latest moved/created/this_way/
  excess_damage result and an owned effect queue. Nested composition lacks
  general result scope and parent/child completion ownership.
- MillAndSelect, ReturnMilledThisWay, OptionalDiscardDraw, MayDiscard,
  SearchBasicLand and other search/move variants duplicate selection and
  continuation work. SearchBasicLand mixes destination and follow-up flags.
- flow/proposed_event.rs collects replacements once and uses a fixed order,
  including an order preferred for a player. General rules require affected
  player choices and applicability re-evaluation.
- SourcePower reads a live source or zero despite existing ObjectLki power;
  TriggeringPower already has an LKI fallback. Entry replacement conditions
  read owner instead of ZoneEntry's actual entering controller.

Implement one explicit runtime ownership/completion protocol for nested
sequence/iteration/branch operations and scoped results. Keep native Effect
values as the sole authored representation: no DSL, compiler or second IR.
Use exact object references and appropriate current/LKI/prospective-entry reads.
Repair the two contextual-read defects as part of this work. Coordinate with
the layers Task: this Task owns resolution/proposed-entry contextual callers;
layers owns characteristic derivation and its cache dependencies.

Use real current cards to migrate the overlapping selection composites. Share
candidate selection, cardinality, visibility, order, movement and result binding
only where semantics match. Preserve fail-to-find, reveal, shuffle, ordered
remainders and simultaneous movement. Surveil 2+ is a useful witness: current
code lacks kept-card ordering. Retain keyword-specific events where required.
Delete each migrated handler and dedicated continuation rather than leaving
parallel old/new executors.

Replacement proposals retain transformed events, affected chooser, applied
replacement identities and logical batch progress across decisions. Recheck
applicability after each replacement; commit once. Only committed events feed
trigger matching. Keep pre-batch sources and LKI where rules require them. Do
not run ordinary SBAs, place triggers or grant priority during unfinished
resolution merely because a child asks a question. Keep the no-replacement
path cheap; indexes are candidate discovery, not semantic authority.

Directed acceptance witnesses:
- ForEachTarget with a child choice: clone at the decision, choose differently,
  resume the correct iteration and perform parent finalization once.
- Leaves from the Vine counter placements under Doubling Season and Caradora,
  Heart of Alacria: affected-player order changes 1 counter into 3 or 4.
  Verify current official rulings and preserve the logical event batch.
- A replacement makes a further replacement newly applicable; applied effects
  cannot reapply indefinitely to the same event.
- Existing source-power attack trigger resolves after source removal; use its
  captured power, never a returned incarnation's power.
- Nonowner entry with differing attack histories uses entering controller for
  raid/entry conditions. Source departure and nested choices retain exact refs.
- Selection with decline, private viewing, kept/remainder order and subsequent
  use of selected/moved objects; normal Command/replay/fork path, not helper-only.

Carry changed state and choices into learning tokens, current native/Python
policy observations, LLM text, Academy, replay and reconnect. Preserve frozen
input meanings; declare version boundaries where necessary. Coordinate shared
Decision/Effect/observation changes with the mana and naming Tasks instead of
inventing competing frame owners. Let their independent implementations proceed;
integrate via ordinary Task sync, not manual worktree sharing.

Profile shared work before caching; use proper revisions and debug oracles.
Measure ordinary steps plus search total/tail with the existing same-runner
Engine Budget and equivalent digests. Explain rules-driven trace changes. No
artificial load or claimed speedup from fewer legal choices. Follow AGENTS.md
and docs/development.md validation, including debug cargo tests and extension
rebuild for changed Rust. No global storage rewrite or full card-set rollout.

The root session also authored managym/PRINCIPLES.md and its README link. Include
that approved documentation in this Task's PR; its exact draft is appended below
when the Task is created. Treat the retrospective as evidence, not permission
to replay historical Task/merge instructions.
