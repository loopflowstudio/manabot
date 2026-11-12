Design training and evaluation for incrementally growing card pools

Jack Heart requested this Task on 2026-11-04 as part of the post-rotation
Standard plan. Card implementation proceeds in substantial shared-mechanic
batches, often across sets, so the available pool will not resemble a complete
set or a mature tournament metagame. We need to understand how to train and
evaluate useful agents at these intermediate stages.

This is a separate research/design Task, not a replacement for Task 04.
Jack Heart explicitly corrected that distinction: Task 04 must actually run
training and report measured results. Task 08 remains the later training and
evaluation campaign. The near-term plan now has nine new Tasks. This Task can
start immediately and delivers the methodology and executable protocol consumed
by those execution Tasks. Jack Heart authorized launching this Task in Rex;
this does not authorize launching training experiments or allocating compute.

Outcome: a concrete, executable and versioned training/evaluation protocol for
irregular card-pool growth, with a justified sampling strategy, meaningful
benchmarks, generalization splits, crash evidence and a concrete specification for the bounded first pilot.
Separate design/protocol readiness from results actually obtained. Research
alone must not be reported as evidence that a training strategy works.

Inspect the repository's actual deck construction, training, observation/token,
search, evaluation, replay and Etude experiment interfaces first. Reuse their
owners. Keep experiment data out of checkpoint namespaces and preserve frozen
worlds, configurations and results. Do not introduce a second card definition,
training-only rules behavior or an incomplete-card admission gate.

Questions to resolve through explicit alternatives and small informative probes:

1. Turning an uneven card file into games. Compare curated coherent decks,
   constrained/generated decks and mixtures. Define mana bases, curve and color
   constraints, interaction density, synergy requirements and playable-game
   checks without narrowing all learning to a few hand-authored archetypes.
   Uniform random cards are a comparison candidate, not the assumed curriculum.
   Distinguish game/deck legality, functional deck construction and competitive
   viability. Define who authors or generates new decks as a batch lands and
   what cost that process imposes. Do not claim generated decks are a metagame.

2. Training distribution. Decide the unit of sampling: card, mechanic, deck,
   matchup, seat, or game state. Compare retaining old anchors, emphasizing new
   cards and broad mixed pools. Measure actual exposure and meaningful decisions,
   not mere presence in a deck. Address rare mechanics, ineffective/dead draws,
   repeated dominant cards, both seats and unequal deck strength. Consider
   continuation from compatible checkpoints versus fresh training; distinguish
   transfer and forgetting from engine/rules or vocabulary changes. Specify
   when a curriculum changes and how its version is retained.

3. Evaluation and attribution. Maintain stable anchor matchups/opponents,
   expanding-pool evaluations and directed mechanic probes as distinct evidence.
   Separate better rules, stronger decks, changed sampling and increased compute
   from improved policy skill. Evaluate policy-only play and search-assisted
   play under declared budgets when both are relevant. Report uncertainty and
   multiple seeds where the conclusion warrants them; self-play win rate alone
   does not measure improvement. Do not compare incompatible historical worlds
   as if they were the same experiment.

4. Generalization. Define held-out cards, held-out combinations/decks, unseen
   combinations of known mechanics, and genuinely new mechanics separately.
   Distinguish visible card definitions/learning tokens from training experience
   and evaluation leakage through opponent pools or curriculum design. Explain
   what a changing catalog means for each split; never silently move the test
   set into training and keep calling it held out. Include cross-pool transfer
   and retention on old anchors. Stronger held-out decks are not by themselves
   proof of worse generalization; design controls that separate these effects.

5. Operational validation. Define bounded learnability checks, actual collection
   and update throughput, search/step performance and capacity/failure reporting.
   Capture failures from ordinary games with seed/RNG, Commands, pending state,
   deck/config/world/input/checkpoint identities, original error and capacity
   counts. Retain private debugging state appropriately. Reproduce in debug and
   route minimal engine regressions to their owner. Never truncate choices,
   invent random-policy fallbacks or generate artificial stress/load. Preserve
   native trivial-decision skipping in real training and evaluation.

Deliver a concrete example for two successive uneven batches (for example,
Adventure/prepare then planeswalkers), showing which training decks/distributions,
anchor evaluations and held-out splits change and which remain comparable.
Select a strategy with evidence and explicit uncertainty rather than delivering
an unranked menu. Specify numeric game/update/compute budgets, metrics and
failure/stop criteria before any pilot launch, using the actual experiment
workflow. Proposed budgets are not permission to provision or spend resources.

Keep two tracks: frequent modest checks of learning, performance and crashes;
sparser deeper strength/generalization studies after major pool expansions.
Task 04’s first campaign validates the selected protocol and ordinary learnability;
it does not attempt the frontier of model intelligence. The deeper track's
first milestone follows full FDN. A useful trainable engine can have known bugs;
scope defects to affected conclusions rather than invalidating all checkpoints.

Acceptance: an executable versioned protocol and deck/sampling manifest, an
explicit recommendation with alternatives considered, frozen evaluation/split
rules, a two-batch example, and a runnable first-pilot specification handed to Task 04. Actual training
results belong to Task 04, not this research Task; do not claim protocol
validation from design alone. Keep the minimal implementation of selected protocol
plumbing in this Task; broad unrelated model/search redesign is outside it.

Dependencies: design has no hard dependency on the four running engine Tasks.
Coordinate observation/action versions with MTG-243 and receipts/history with
planned Tasks 01/03. Task 02's Adventure/prepare and Task 06's planeswalker
cohorts are representative future inputs, not prerequisites for all research.
Validation A waits for its declared integrated engine snapshot. Tasks 04 and 08 consume
this protocol instead of inventing separate training methodologies. Task 04
remains an actual training run with measured learning/performance and crash
evidence; it is not satisfied by writing an experiment configuration.
