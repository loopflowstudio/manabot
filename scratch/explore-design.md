# MTG-245 — exploration, 2026-11-04

Jack Heart invoked `explore` on 2026-11-04, then selected A as the starting
approach: “i agree A is the place to start”. The reference league is the accepted
initial direction: authored functional decks, a frozen mixture and stable
evaluation anchors. The live implementation Graph owns protocol plumbing.
This decision does not authorize games, training or spending, or accept the
illustrative numeric parameters. B and C remain future alternatives.

## Evidence and constraints

- Task brief: executable versioned protocol, uneven batches, fixed evaluations,
  leakage controls, failure evidence, and a pilot handoff to Task 04. Task 04
  owns measured training; Task 08 owns the later campaign.
- `scratch/questions.md`: current implementation proposes integer population
  tickets compiled into existing independent `UniformDecks` draws; fresh
  initialization, strict capacity errors, frozen runtime/input identity.
- `manabot/training/decks.py`: ordered deck populations, explicit pair semantics,
  stable entry IDs, deterministic reset-seed sampling and native admission.
- `manabot/training/regime.py`: existing recipes, cadence and bounded execution.
- `manabot/training/state/game_errors.py`: durable attempts, private failure
  inputs and a declared rolling error allowance. An interrupted run is not
  counted as an ordinary failed game.
- `manabot/eval/evaluation.py` and `experiments/tla_pilot_v9.py`: shared evaluation
  and Etude integration. Reuse owners, not MTG-213's frozen protocol or its
  first-64 exception.
- `experiments/curriculum.md`: historical six-seed simple-to-complex intervention
  demonstrated no useful benefit; heuristic primary contrast was inconclusive.
  This is limited evidence against assuming staged simplicity helps, not a test
  of the three visions here or evidence about current worlds.

## Explored visions — A selected as the starting approach

### A. The reference league

Each batch gets a small authored population of functional decks and a frozen
mixture. Illustrative per-seat tickets: 50% retained anchors, 30% new-batch
decks, 20% cross-batch decks. Draw seats independently with existing
`UniformDecks`; ticket shares do not guarantee mechanic exposure or balanced
matchups. If the new cohort has 30% mass, at least one seat has a new-cohort
deck in 1 - 0.7² = 51% of games in expectation. Actual plays remain measured.

An experiment author owns mana, curve, interaction and synergy review; a rules
owner resolves admission failures. Future batch A has Adventure/prepare decks;
batch B retains selected A decks, adds planeswalker decks and creates a few
mixed lists. Existing test lists and deal families stay fixed. Within a changed
world, rerun a newly bound control; never claim raw cross-world score deltas
measure policy progress. Lowest new plumbing; strongest author bias and ongoing
deck-maintenance cost. Selected recommendation for the first runnable handoff,
because it isolates failures and uses the current deck source. Not a claim of
superior learning.

### B. The coverage workshop

A constrained generator assembles functional decks around mechanic families,
then freezes a diverse population before training. Illustrative authoring limits:
one or two colors, 24–27 lands in 60 cards, at least 12 plays costing at most
three mana, and at least six interaction cards. These are editable manifest
constraints, not universal definitions of good Magic decks; native legality,
functional construction and competitive strength stay separate.

An experiment author defines constraints and synergy roles; generation automates
list production. An illustrative screen is at most 16 candidates × four ordinary
game legs = 64 games per revision, with a separately fixed screen opponent/deal
set. This is proposed future work, not run here or a power calculation. Screens
reject only declared functional failures for this study, never remove native
card support. Freezing before training prevents the learner from changing its
own test. Coverage uses ordinary-game evidence (drawn, offered, chosen, resolved),
not just card membership. Some precise mechanic exposure metrics need additional
instrumentation; do not pretend existing deck receipts provide them.

Batch A generates Adventure/prepare families; B adds planeswalker families while
retaining old-family quotas. Better breadth and less manual list writing; more
generator/audit code, game cost and a risk of functional but strategically hollow
decks. Held-out combinations must be excluded from candidates and screens.

### C. The discovering league

Radical success: training finds its own blind spots. A bounded opponent/deck
search proposes challenging ordinary games from training-only pools, and a
versioned curriculum changes only at declared checkpoints. A fixed 50% replay
share protects old anchors; the remaining share uses discovered challenges.
These shares are illustrative design choices. Learning progress and exposure
drive proposals; catastrophic bugs and capacity failures route to their owner,
never become a reason to reward an opponent or silently skip a mechanic.

An illustrative proposal round screens at most 64 candidates × four legs =
256 games; total rounds, wall time and spending require separate fixed bounds.
Batch A might expose poor resource use around Adventure; B might expose failure
to pressure planeswalkers. These are invented examples, not observations.
Freeze challenger generations, selection rule, rejected candidates and parent
checkpoint identity. Keep final held-outs inaccessible to the search.

Highest potential to discover interactions people did not author. Highest
engineering and evaluation cost; risk of exploiting deck imbalance or engine
bugs, curriculum nonstationarity and hard-to-attribute gains. This exceeds the
minimal first-pilot plumbing and would need separately scoped work if selected.

## Common scientific boundary

Every option keeps ordinary native rules and trivial skipping, complete legal
choices, strict capacity failure, frozen inputs and private replay evidence.
No new implementation of cards or training engine. Evaluation has independent
anchor, expanding-pool and mechanic-probe panels; policy-only and search-assisted
results use separate fixed compute envelopes. The modest track checks learning,
throughput and failures; the deeper track first follows full FDN.

Held-out card identities, combinations, complete decks and genuinely unseen
mechanics are separate labels. Visible definitions/tokens are not gameplay
experience. Training opponents, generator screens and adaptive challenge search
obey the same split exclusions. Promoting a test item into training retires its
held-out claim prospectively. Fixed deck-role/seat swaps and common opponents
help separate deck strength from policy skill; raw self-play win rate cannot.
Fresh training is the clean first handoff. Continuation needs admitted compatible
world/input/optimizer semantics and a fresh control at matched compute; incompatible
worlds never become comparable by changing metadata.

## Implementation slices and acceptance

Continue the existing implementation rather than starting another Graph:

1. Freeze authored deck identities, roles, split exclusions and deterministic
   population tickets through the existing deck-source owner. Document who
   authors/reviews additions and the mana, curve, interaction and synergy checks.
2. Retain fixed evaluation anchors and show two successive batch revisions;
   separate changed decks, changed world and changed policy. Verify sampling
   determinism, split isolation and rejection of mismatched world/input identities.
3. Hand Task 04 a runnable bounded pilot with explicit proposed numeric budgets,
   metrics, failure criteria and evidence requirements. Do not report measured
   learning or coverage from manifest construction alone.

These specialize the existing Task acceptance to A. Generator development and
adaptive challenge search are not required for the first handoff. Exposure
reporting remains a Task-brief requirement; selecting A does not silently add
B's generator or accept a particular instrumentation design.

## Decision and remaining parameters

Page: `scratch/explore.html`.

1. Settled by Jack Heart, 2026-11-04: A, the reference league, is the place to start.
2. The exact mixture, authored lists, numeric pilot bounds and relative emphasis
   on learnability versus coverage remain proposed engineering choices. No
   separate answer to the page's second question was supplied.

Numeric examples are exploration parameters, not a replacement for the live
Graph's complete pilot budget or acceptance criteria. Latest review evidence:
`scratch/explore.html`.

## Follow-on discussion — workflow ownership (2026-11-04)

Jack Heart considered adding tiny training runs with authored decks to card
authoring, then suggested deck creation could instead belong to the validation
Task because that is simpler. Jack Heart also asked whether dedicated Loopflow
Graphs and skills should target growing the card file. These are proposals
under discussion, not approval to install a Graph, create Tasks or launch runs.

Recommended shape: a coherent mechanic batch is the unit of iteration. Keep
card delivery and empirical validation as separately owned outcomes, coordinated
by a thin workflow. A validation Task can consume several integrated card Tasks.
It authors the actual decks and runs the bounded check on a declared integrated
snapshot; the card author supplies intended synergies and behavior-test evidence.
This avoids assigning a training campaign to every individual card change.

Proposed domain skills:

- `card-batch`: select the cohort and implement reusable rules, definitions,
  observations/tokens and directed debug tests; hand off supported interactions
  and known defects. Reuse ordinary delivery operations.
- `validation-decks`: on the integrated snapshot, author a small set of coherent
  new-mechanic and mixed decks; retain anchors; document mana, curve, interaction,
  synergy and actual split membership; emit the MTG-245 manifest.
- `pool-validation`: consume that manifest and the versioned MTG-245 protocol;
  freeze source and budget, run only within explicit execution authorization,
  retain updates/throughput/failure evidence and paired initialization evaluation.
  Very short runs can demonstrate collection/updates and diagnose failures;
  improvement may remain inconclusive. Report that rather than making learning
  gain a per-card admission gate.

Current sequence: batch plan → working code loop → scalable code loop → performant
code loop → ordinary card delivery → integrated validation/deck authoring → bounded
experiment → evidence review. Keep delivery, usable game coverage and demonstrated learning
as separate reported facts. A weak learning result does not revoke native card
support. Generic gate/deliver operations and Etude remain their existing owners.
The deeper campaign follows major expansions rather than every batch.

Repository inspection: `.lf/flows/rules-audit.yaml` and related legacy flow files
exist, but `lf graph list` currently lists only builtin Graphs. Do not treat the
legacy syntax/locations as a verified installation recipe; an implementation
Task must use installed Graph/skill authoring guidance and validate discovery.

Open choice: capture this as a small dedicated workflow Task once its boundary
is agreed, without expanding or interrupting MTG-245's live protocol work.

## Training/eval workflow audit — 2026-11-04

Jack Heart expanded the exploration to the training and eval Waves and stated:
“we should likely use the existing lf builtin code graphs as subgraphs or forks
or somethign”. Preserve reuse of builtins as the design preference. The specific
Graph names, skills and rollout below remain proposals; no installation or Wave
default changes have been selected.

Evidence: fresh `lf wave status training --json` and `lf wave status eval --json`,
both Wave memories, `docs/training.md`, `docs/evaluation.md`, installed
`lf help code`, `lf help pursue`, `lf help deliver`, `lf graph show code`, and
Loopflow's local `docs/orchestration.md` / graph authoring model. Reads only;
this session has not taken over either Wave or its Tasks.

Findings:

- Training's objective explicitly requires measured improvement at declared
  cost. Its live work mixes experiment MTG-213 with representation implementation
  5da5f4; MTG-210/211 are unstarted contracts/verifier work. MTG-213 has a merged
  enabling PR while its Task still works from verify: code delivery and scientific
  completion are plainly separate. Its memory repeatedly distinguishes preparation,
  GPU execution, trained exports, evaluation, recovery and cost settlement.
- Eval's current Tasks are MTG-175 (curriculum comparison, idle with a failed
  integration check) and 513158 (anchored checkpoint ratings, unstarted). Memory
  also captures hosted-model comparisons and inference measurements. Recurrent
  distinctions are protocol/model admission, paired schedules, budget accounting,
  missing outcomes, replay, uncertainty and claims. Training calls the same
  evaluator; a training-specific second evaluator would duplicate ownership.
- Training's configured default is `iterate`, absent from the installed Graph
  list; the checkout still has `workflow: iterate`. Eval defaults to `code` and
  its authoritative Wave goal is empty. These are setup findings, not permission
  to repair defaults, alter existing captured Task graphs or author a Wave goal.
- Current Loopflow composes graphs using `graph:` and preserves nested interactive
  steps. `code` already composes `pursue` and `deliver`. Local graph definitions
  are captured for Tasks; later definition changes do not rewrite live execution.

Revised recommendation after the discussion below: one `experiment` graph,
with question-specific design and analysis, shared execution and evidence
handling. Card growth uses it after its three code loops and deck authoring.
Training and evaluation are stages in the experiment, not separate graph families.
Preserve separate card and integrated-validation Task ownership. The final
implementation must respect Loopflow's Task capture/completion boundaries;
a diagram is not a license to silently create or complete child Tasks.

Pure trainer, encoder or evaluator implementation continues to use builtin
`code`. No custom graph should force a paid campaign onto an infrastructure fix.
An already admitted executable protocol skips the software branch without a
placeholder PR. Software changes must precede final source/protocol freeze.

Prefer named subgraph composition to copied builtin bodies, so the repository
owns only domain behavior. Use a separately named fork of `code` only if a
domain review/check must occur inside its implementation/delivery cycle and
composition cannot express the required placement; retain provenance and accept
maintenance cost. Do not shadow global `code` or remove its review boundaries.
“Fork” here means a customized definition, not parallel agent execution.

Share narrow protocol, execution-reconciliation and evidence-analysis skills.
Etude owns jobs, allocations and continuation; Loopflow sequences the work.
Reviews need not ask again for spending already explicitly authorized, but a
graph name or headless launch never creates an experimental spending allowance.

First rollout proposal: implement the smallest card-batch validation instance
of the experiment graph around MTG-245's protocol. Generalize from actual use.
Leave live Graphs and Wave memory untouched during this audit.

## Card-growth code loops — direction from Jack Heart, 2026-11-04

Jack Heart requested first getting the batch working, then finding the most
scalable decomposition, then making it as performant as possible, and clarified
that **each has its own code loop**. This supersedes treating card implementation
as a single loop with three checklist items.

1. **Working loop:** implement the rules behavior, definitions, complete shared
   observation/action paths and tokens; directed debug tests and review establish
   intended behavior. Follow repository architecture rules from the outset;
   this stage does not permit card-specific engine hacks or hidden semantics.
2. **Scalable loop:** use the working examples to decompose reusable rules types,
   selectors, ownership and data flow. Exercise distinct consumers/representative
   interactions; remove redundant special paths. Review whether further cards
   compose from the same mechanisms. Scalability here is growth in supported
   mechanics/cards and maintainability, distinct from runtime optimization.
3. **Performance loop:** profile representative bounded ordinary workloads,
   improve measured bottlenecks, compare on the same hardware, and recheck debug
   invariants and behavior digests. Iterate over worthwhile supported gains;
   passing the 10% engine regression budget alone does not satisfy this ambition.
   Report remaining bottlenecks and justified tradeoffs; do not claim an absolute
   optimum. Never induce artificial load, discard choices or narrow observations
   to manufacture throughput.

Each stage owns implementation → checks → review → revise until its criterion
is met. Stage-specific review can return to that stage; a discovered semantic
defect returns to the working contract and invalidates dependent evidence.
Preserve unique phase identities when composing repeated builtin loops.

Implementation proposal: reuse the builtin `pursue`-style code iteration inside
each phase and preserve normal demo/delivery/verification boundaries around the
whole batch. Three distinct iteration loops are selected; three separate PRs
or merges are not. Exact graph composition and delivery granularity remain open.
No Graph installation or execution is authorized by this design discussion.

## One experiment, several scientific lenses — 2026-11-04

Jack Heart challenged separate training/evaluation experiment flows because
experiments involve both. Jack Heart distinguished NN architecture experiments,
training-regime experiments and trying new cards, then questioned making those
three separate flows, suggesting a fork within experiment instead. Preserve
these distinctions without treating the precise routing structure as settled.

Current recommendation: **one experiment graph**:

question → design and controls → review → builtin code loop if software is
needed → freeze final source/protocol → bounded execution → analysis → evidence
review. Training, evaluation and intermediate checkpoints follow the frozen
protocol. Some experiments use existing checkpoints; no placeholder fitting
stage or PR is necessary. Changes after freeze get a prospective revision.

Design/review lenses:

- **Architecture:** vary model structure/representation; hold regime, world,
  deck distribution and evaluation constant where the hypothesis permits. Declare
  parameter counts, capacities and the matching clock (transitions, time or cost);
  do not imply equal updates mean equal compute. Changed inputs require an explicit
  representation contrast, not a hidden confound. Use replicated learning evidence
  for strength claims; correctness or compression alone does not prove them.
- **Regime:** hold architecture and environment fixed while varying declared
  training choices (sampling, objective, schedule, opponents). Preserve controls,
  independent seeds, total exposure and compute accounting. Distinguish continued
  optimizer state from fresh training and measure forgetting as applicable.
- **Card pool:** bind an integrated expanded world and authored decks; measure
  actual mechanic exposure, functional games, learning, performance and failures.
  Re-admit controls within each world. Separate card delivery and usable gameplay
  from demonstrated learning; finite runs cannot certify every reachable state.

A design step can consider all three lenses and choose the useful intervention;
they are not three automatically launched campaigns or exclusive classifications.
An explicitly factorial experiment can combine axes with appropriate controls
and budget. If specialist branching/forking earns its cost, put it in design or
review and reconcile to one protocol before execution. Parallel design opinions
and experimental arms are different things. Actual parallel agents have not been
requested/launched here, and no installed routing syntax is claimed by this sketch.

Shared owners: protocol identities, seeds/splits, bounded Etude execution,
recovery/cost reconciliation, replay evidence, uncertainty and report publication.
Scientific emphasis changes by question; lifecycle machinery is shared. Ordinary
infrastructure work still uses builtin `code` without an obligatory experiment.

## Proposed four-Wave organization — 2026-11-04

Jack Heart proposed “Run experiment”, “Expand coverage”, “Optimize engine”, and
“Catchall infra”, then asked whether these are the four Waves. Recommended yes
for the core engine-and-learning program, with responsibility names Experiments,
Coverage, Engine performance, Infrastructure. This is the current organization
proposal, not a completed migration or authorization to rewrite Wave memory.

| Wave | Owns | Completion evidence / default procedure |
| --- | --- | --- |
| Experiments | Scientific questions about model architecture, training regime, card pools and their interactions; training and evaluation together | Frozen protocol, actual runs, reconciled failures/costs, analysis and supported conclusion; shared experiment graph |
| Coverage | Faithful cards and mechanics, shared observations/actions/tokens, rules defects and integrated playable batches | Behavior tests and working → scalable → performant code loops; handoff of the integrated snapshot to experimental validation |
| Engine performance | Cross-cutting engine algorithms, allocations, caching, throughput and latency | Behavior-preserving gains on representative workloads and the same hardware; measure/optimize/review code loop |
| Infrastructure | Shared training/evaluation plumbing, artifacts, experiment interfaces, CI, developer tooling and operational reliability | Working maintained systems through builtin code; no experiment automatically required |

Route by intended outcome, not changed package. A model comparison is Experiments;
reliable checkpoint serialization is Infrastructure. A new mechanic is Coverage;
a general state-copy speedup is Engine performance. Coverage still optimizes its
own new implementation; reusable cross-cutting bottlenecks are separate performance
Tasks. A campaign's rules failure stays linked to its evidence while the fix belongs
to Coverage. Do not duplicate ownership or require every batch to wait for unrelated
global optimization. Infrastructure should have a positive reliability/developer-
experience objective rather than become an unbounded miscellaneous queue.

Academy is an existing browser-product responsibility outside this core four-Wave
proposal; no decision has been made to retire it or absorb it into Infrastructure.
Existing training/eval Tasks, graph captures, ongoing runs and historical provenance
remain unchanged. A later explicit reorganization should preserve that ownership
and history rather than infer a migration from this discussion.
