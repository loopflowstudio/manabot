# Decisions and open questions — 5da5f4

- Dependency direction, Jack Heart, 2026-11-04: use latest Etude source instead
  of waiting for released SDK versions ("i think that might be better long term
  for now"). Adopt Etude Git main for Manabot's uv source with an exact commit
  in uv.lock; refresh explicitly, never float dependencies inside a run. Cover
  launcher, remote locked worker, extras and clean-environment integration.
  Historical protocols/locks/receipts remain immutable. Local billing fixes
  are not on main merely because source tracking is enabled; deliver them in
  Etude and record their actual revision. This does not increase the budget.
  Implemented: all four dependency declarations use valid `>=0.2.14` metadata;
  public Git main resolves in `uv.lock` to `64e9eaabf016f567cb9567280bcf8730f6f5d524`
  (merged Etude PR82). Launcher and clean locked worker dependencies match;
  anonymous access and offline locked re-sync passed. Linux resolution is a
  dry-run only, not paid remote admission.

- Jack Heart's latest 2026-11-04 direction authorizes a viable GPU restart toward
  a sustained learning plateau within the ORIGINAL cumulative $25, including
  supported batching and necessary Etude SDK release/Vast use. Earlier CPU-only
  stop and return-to-review restrictions are superseded. Results stop at
  demo/review; no merge, model/default promotion or MTG-213 funds are approved.
- Current outcome: cancelled eight-learner science and three unsuccessful restart
  admissions; no replacement scientific cohort or plateau. All five Vast rentals
  remain absent at the repaired 20:10 UTC read. Full-day Vast charge records report
  $0.398, or $0.4705825939 including the settled L4 bill; these supersede the
  $0.4487194142 estimate as current reported charges, not final settlement.
  Etude source repair now retrieves typed daily totals and removes the invalid
  timestamp-to-settlement conversion. The corrected read still supplies no final
  coverage through teardown; no SDK release or repeated query can invent it.
  Reservations remain $24.9725825939 including $6 ancillary, leaving $0.0274;
  no new $0.20 admission fits until authoritative settlement permits release.
  Exact attempt denominator, evidence and remaining work are in
  [the plan](representation-learning-plan.md).
- Operator batching assumption: 256 streams changes optimization, so it needs
  fresh paired seeds and a prospective identity, preserving the 65,536-game
  horizon and schedule by exposure. No admission score selects a treatment.
  The last machine loss establishes neither OOM nor batching affordability.
- Reconciliation (2026-11-04): the current executable supplies admission and
  analysis only. Replacement scientific orchestration remains implementation
  work after accounting and measured affordability; no architecture decision
  follows from the CPU endpoint or partial GPU attempts. T7's merged rules and
  chooser changes require prospective source/native admission, not relabeling
  earlier receipts. No new product decision is required for that authorized work.
- Broader history/memory, relation splitting and semantic coalescing remain
  deferred. MTG-240 owns the separate keyword timestamp defect; it does not
  justify deleting effect records. The plan retains the required counterexamples
  and viewer/reset/restoration constraints.

Complete preceding decisions, attribution, contrary evidence and historical
validation remain at `1b2ddcdfadcb09df76b2dda450152cbdfa60c8c2:scratch/`.
The live plan consolidates those sources; no uncommitted evidence was removed.

## Inherited T7 context from main

The sync retains T7's chooser API and stable viewer perspective. Control does
not imply a perspective switch or new identity field. Preserve frozen input
widths and saved omissions, and keep synthetic pilot fixtures separate from
strict runtime/schema rejection.

T7's evidence does not establish search WebSocket recovery, crash durability
or retained-checkpoint compatibility with the new world. Earlier green T6
suites missed targets-before-payment and automatic-four-card search; current
checks must not inherit stronger claims.

T7's publication and no-experiment boundaries belong to that Task, not this
representation study. Its complete decisions, demo references and historical
sources remain at
`4463e57f5a92ebd3b60e2fb50c046b84ac2c0372:scratch/questions.md`.

- Reversible implementation assumption (2026-11-04): repair the authorized SDK
  dependency in `.runs/etude-billing`, based on released Etude 0.2.14, without
  changing another Task's checkout or the study's locked worker. Preserve the
  SDK patch and source identity with reconciliation evidence; the repair is
  merged as https://github.com/loopflowstudio/etude/pull/82. No new SDK
  release can supply the missing provider coverage. Loopflow's nested
  checkout sync inherits this parent Task's branch binding; SDK delivery uses
  its own GitHub branch without altering Manabot Task association.
