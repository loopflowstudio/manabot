# MTG-245 — exploration, 2026-11-04

## Accepted execution scope — 2026-11-04

Jack Heart said “ok, lets do that” to the four-Wave/workflow proposal, then
corrected the proposed separate Task with “nah i think jsut leave that in this”.
Implement this work inside MTG-245 alongside its original protocol deliverable.
No separate Task was created. The existing live Graph owns implementation;
do not start a competing Graph or replay completed steps solely for this steer.
This supersedes earlier statements below that workflow installation and Wave
reorganization were only proposals.

Accepted outcome:

- Four core Waves: Experiments, Coverage, Engine performance, Infrastructure.
  Keep Academy separate. Infrastructure owns shared systems, tooling and
  reliability. Route Tasks by outcome, not the files they change.
- Compose builtin code graphs where suitable. Use narrowly named forks only
  where domain-specific placement requires them. Preserve review, delivery and
  verification boundaries; do not shadow generic builtin code.
- Coverage has three independent code loops: working behavior → scalable
  decomposition for growing the card file → measured performance optimization.
  Each owns implementation, checks, review and revision until its criterion is
  satisfied. Three PRs/merges were not requested. Preserve shared semantics,
  debug invariants and the prohibition on artificial load.
- One experiment graph encompasses training and evaluation. Architecture, regime
  and new-card coverage are design/analysis lenses, not three mandatory flows
  or automatically launched campaigns. Specialist branches can reconcile into
  one protocol when useful. Existing checkpoints need no placeholder fitting.
- Validation owns authored decks and a bounded experiment on the integrated
  snapshot. Card authors provide interactions and behavior evidence. Preserve
  A's reference league and the versioned protocol/splits. Tiny runs may prove
  execution while leaving learning inconclusive.

Use supported lf operations for Wave identities and membership. Preserve Task
IDs, PRs, histories, frozen evidence, captured graphs and live runs. Do not claim
or execute other backlog Tasks to reorganize them. Refresh state before effects;
bring consequential ambiguous mappings here. Prefer identity-preserving rename/
move operations to duplicates. Preserve Wave memory/provenance through the
selected migration procedure.

Refresh these observed integration facts: training selects missing `iterate`;
eval's goal is empty; legacy `.lf/flows/` are not discovered by the installed
graph catalog. PR #421, Task 5da5f4, modifies `.lf/wave/training/MEMORY.md`;
preserve that work and its reserved empirical decisions. MTG-241/244 and this
Task have live work; MTG-243 is at demo. MTG-213 has separate paid-run authority.

Acceptance adds discoverable/validated graphs and skills, three Coverage loops,
one experiment lifecycle, documented Wave boundaries/defaults, supported migration
receipts, unresolved mappings where necessary, and proof live work/frozen identities
were preserved. Validate compilation/discovery and safe fixtures; do not launch a
scientific campaign as a setup test. No new games/training campaigns, rental,
spending or automatic shipping authority is supplied. Task 04 still owns actual
training/results and Task 08 the later campaign. Use MTG-245's existing review.

## Implementation state — 2026-11-04

Seven repository graphs and domain skills are installed in this checkout and
compile through the real `lf` catalog. Coverage has three separate implementation/
review loops and one delivery boundary; experiment has one design/execution/
analysis lifecycle with an optional software branch and explicit reviews.
`docs/workflows.md` records ownership, defaults and the dated migration receipt.
Engine → Coverage uses the same identity and unchanged memory; Engine performance
and Infrastructure were added through supported lifecycle operations. Shared
defaults are authored here for delivery, replacing Training's missing `iterate`.

Consolidating Training/Eval into Experiments still requires a surviving-identity
judgment, especially while PR #421 modifies Training memory. No duplicate
Experiments Wave is created. Provider synchronization is separately pending under
Linear's rate limit. Existing Task graphs, PR identities and frozen evidence are
preserved; no campaign or spending was launched. Training-exposure acceptance and
proposed numeric settings remain for authored review as recorded in questions.

## Selected protocol and alternatives

Jack Heart selected A on 2026-11-04: “i agree A is the place to start”.
Use the reference league: authored functional decks, frozen independent per-seat
sampling tickets and stable evaluation anchors. The protocol, construction
checks, held-out lifecycle, two-batch authoring example and bounded Task 04
handoff now live in `experiments/protocols/growing-pool-245-v1.md` and its JSON.
`scratch/explore.html` retains the interactive A/B/C comparison for review.

A reuses native Deck/UniformDecks, TrainingRegime/TrainingJob and the shared
replay evaluator. It offers interpretable failures with little new plumbing,
at the cost of author bias, limited breadth and recurring deck maintenance.
The six-seed historical curriculum study found no useful benefit from its
particular simple-to-complex intervention. It does not compare A/B/C or current
worlds; no strategy is established as the strongest learner.

B (constrained generation) could broaden coverage and reduce list writing, but
requires audited construction rules, frozen functional screens and split-safe
selection. C (adaptive challenge search) could expose unanticipated weaknesses,
but adds selection cost, nonstationarity and incentives to exploit deck imbalance
or engine bugs. Both remain future alternatives, not first-handoff requirements.
Their exact illustrative envelopes and authoring discussion remain at
`190b0c603d70e790f2217c61c970c38f82544d08:scratch/explore-design.md`.

The exploration's 50/30/20 example and executable proxy's 50/25/25 proposal are
both unaccepted parameters. Neither sampling mass nor deck membership establishes
mechanic use. Exact lists, relative emphasis on learning versus coverage, seeds
and numeric bounds remain for authored review. Selection of A does not waive
training exposure or require B's generator to obtain it.

## Implementation acceptance

- Preserve authored deck identities, ordered tickets, both-seat split isolation
  (including sideboards), and construction provenance. Native support, deck
  function and competitive strength are different claims.
- Retain stable panels, anchors and deal families. Held-out cards, complete decks,
  within-deck combinations and unseen mechanics are distinct; cross-seat exposure
  is possible for the combination split. Definitions/tokens remain visible.
  Promotion retires the held-out claim prospectively. Opponents and any future
  screens/search obey the same exclusions.
- Distinguish world, deck and policy changes. New worlds need admitted controls;
  fresh training is the first handoff. Compatible continuation needs explicit
  optimizer/input/world admission and matched controls, never relabeled evidence.
- Reconcile every declared seed/export/callback/game, with paired whole-deal gains
  per seed and declared uncertainty. Separate execution completeness, numerical
  health, learning and exposure. Policy-only and search-assisted evidence have
  distinct compute envelopes. Self-play win rate is not a learning metric.
- Keep native trivial skipping, complete legal actions, strict overflow, private
  failure evidence and frozen configurations. No artificial load, random fallback
  or support gate for known rules bugs. MTG-213's first-64 and paid-run exceptions
  supply no authority to this Task.
- Task 04 authors integrated Adventure/prepare decks and exact policy-probe IDs,
  accepts settings and actually runs/measures the first pilot. The planeswalker
  revision retains anchors and old-family tests with a prospective binding.
  These future authoring specifications are not delivered executable manifests.
  Task 08 owns the deeper campaign after full FDN.

## Workflow design and migration

Jack Heart requested independent working → scalable → performant code loops,
then unified training/evaluation under one experiment lifecycle. Architecture,
regime and card-pool questions are design/analysis lenses; they are not three
mandatory flows or automatically launched campaigns. Specialist review can
reconcile into one protocol if useful. No parallel agents are implied.

`docs/workflows.md` owns the implemented boundaries and graph composition;
`docs/workflows/2026-11-04-migration.json` retains before/after identities,
supported operations, protected graph hashes, memory provenance and pending sync.
Infrastructure uses builtin code. Coverage optimizes its own implementation;
Engine performance owns cross-cutting bottlenecks. Validation owns authored
decks and bounded scientific execution on the integrated snapshot. Card authors
supply interaction/behavior evidence; a validation Task may consume several
integrated card Tasks. Passing the engine's 10% regression allowance is not
proof that optimization is finished.

The migration deliberately retains Training/Eval identities and paths pending
consequential consolidation review, especially PR #421's Training memory and
reserved empirical decisions. MTG-210/211/230/231 ownership remains unresolved.
No backlog Task was claimed/executed to reorganize it. Existing captured graphs
and live work remain their owners. Current unresolved work and validation are
kept in `scratch/questions.md`, rather than another implementation ledger.

## Delete — do not maintain

Completed: replace Training's missing `workflow: iterate` with `graph: experiment`;
remove the Engine directory through the supported identity-preserving rename
while retaining memory bytes. Remove superseded installation sketches from
ambient scratch; durable workflows now own the procedure. The historical
exploration is retained at the git reference above. No exclusive tests/fixtures
were removed; the protocol and workflow checks exercise the surviving paths.

Keep legacy `.lf/flows/` as historical Task-specific definitions; they are not
predecessors of these new graphs. Keep builtin code/pursue/deliver and captured
Task graphs unchanged. The experiment software verification fork must not invoke
builtin verify's parent-Task completion. No remaining code deletion target was
identified; consolidating Training/Eval awaits authored judgment.
