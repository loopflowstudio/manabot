# MTG-245 — accepted design, 2026-11-04

Jack Heart selected A (“i agree A is the place to start”), accepted the workflow
work inside MTG-245 (“nah i think jsut leave that in this”), and resolved the Wave
identity (“training is the main one and the evals tasks get moved into it”).
Training's original identity survives as Experiments. Training-exposure capture
remains required. No scientific execution, spending or shipping was authorized.

The accepted outcome is four core responsibilities: Experiments, Coverage,
Engine performance and Infrastructure, with Academy separate. Coverage has
independent working, scalable and performance loops before one delivery boundary.
One experiment lifecycle spans training/evaluation, with optional software work
that cannot complete its parent scientific outcome. Architecture, regime and
card-pool questions are design lenses, not mandatory campaigns or parallel agents.
Validation authors decks and a bounded experiment on the integrated snapshot;
card authors supply interactions and behavior evidence.

`docs/workflows.md` owns graph composition and responsibility boundaries;
`docs/workflows/2026-11-04-migration.json` preserves the supported operations,
Task/PR/graph identities, memory provenance and pending provider synchronization.
The migration includes retiring the concurrent replacement Training Wave after
moving its live Task intact. No moved backlog was executed. PR #421's original
memory contributions and empirical reservations must survive integration into
the Experiments path. MTG-210/211/230/231 mappings remain unselected.

`scratch/questions.md` owns current implementation, assumptions, remaining
ownership and validation. The historical review is retained in
`scratch/concept-review.html`; its two questions are resolved by Jack Heart's
2026-11-04 continuation, not open scope-waiver decisions. Its counterexample still
matters: favorable evaluation and a complete exam cannot establish what the
learner encountered. Training exposure must remain independent evidence.

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
  Task 08 owns Validation B after the expanded mechanism cohort. The separate
  deeper track starts after full FDN; its owning Task is not yet assigned.

## Delete — do not maintain

Completed: missing `workflow: iterate` replaced by the shared experiment default;
Engine renamed Coverage; Training renamed Experiments; redundant Eval and the
concurrently recreated Training retired through lf after preserving work/history.
Eval's leftover authored config and the exposure-unavailable report placeholder
are removed. Native card models moved from sim/card_evidence to env/card_capture
with consumers, leaving one shared model owner. Frozen JSON, historical capture
bytes, existing evaluation evidence and captured Task Graphs are preserved.

No new generator, mechanics taxonomy, campaign manager, rules state or checkpoint
namespace was introduced. Legacy .lf/flows are historical definitions, not a
parallel implementation to repair. Keep builtin graphs unchanged.


The exposure aggregation owns only admitted terminal-game counts; the reader
owns run/source/deck admission and frozen budgets. This keeps interrupted/missing
records separate from observed zero without rebuilding counters inside each game.
Receipt hashing and durable writes share one JSONL serialization, and native
training/evaluation exports share the card projection without a JSON text roundtrip.
