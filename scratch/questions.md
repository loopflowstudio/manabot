# MTG-245 — growing-pool protocol

Protocol baseline on 2026-11-04:
`190b0c603d70e790f2217c61c970c38f82544d08`. Current working changes also contain
the accepted workflow implementation and simplification.
The Task has no Wave binding. Wave memory content is not curated; supported
lifecycle operations preserve it during the accepted migration.

## Accepted outcome and constraints

Jack Heart selected exploration A on 2026-11-04: “i agree A is the place to
start”. The initial approach is a reference league of authored functional decks,
frozen sampling mixtures and stable evaluation anchors. The executable protocol
must retain split isolation, exposure reporting, failure evidence, a two-batch
example and a bounded first-pilot specification. Selecting A does not require
B's generator or C's adaptive search, or waive the original exposure requirement.

Exact lists, 50/25/25 shares, two seeds and numeric budgets are proposed engineering
parameters, not accepted experimental settings. Task 04 owns actual training and
measured results; Task 08 owns the later campaign after full FDN. No training
launch, games, rental or spending is authorized here.

The implementation reuses Deck, UniformDecks, TrainingRegime, TrainingJob and
EvaluationRequest. Independent per-seat integer tickets express the mixture.
Both training seats exclude held-out lists/cards; the combination split is
within-deck only, not absence of cross-seat co-occurrence. Strict capacity errors,
native trivial skipping, fresh initialization and the full world vocabulary
remain required. World/input changes need prospective binding; frozen evidence
and checkpoints are never reinterpreted.

The unrelated inherited MTG-213 notes remain at
`8e8d702f807d61675fc1bca46417eee69316daa0:scratch/questions.md`.
Its first-64 exception grants no truncation authority here.

## Implemented

`experiments/protocols/growing-pool-245-v1.json` freezes nine current-world proxy
lists, eight ordered tickets, five evaluation panels, input/model/learning
configuration and proposed bounds: 512 training games, 800 evaluation games,
$5 inclusive. Its adjacent Markdown owns the methodology, alternatives,
two-batch authoring specifications and Task 04 handoff. The 40-card Elemental
proxy is not Standard or Validation A's eventual integrated snapshot.

`growing_pool_protocol.py` admits native lists, checks the declared construction
template and splits, and compiles existing training owners. Source admission
rejects revisions missing selected protocol bytes. Both Run declarations admitted
the committed analysis revision above. Changed source must be committed and
re-admitted before execution; declaration admission is separate from campaign
authorization and experimental results.

`growing_pool.py` supplies canonical Etude discovery and four policy/deck/seat
legs per deal. Its shared comparison schedule accepts an absent TrainingRun;
the offline reader starts from both declared seeds, retaining missing runs,
exports, callbacks and individual game slots. Checkpoints require
the exact regime and raw/initial role as well as ordinary world/input admission.
Fixed heuristic source identity excludes manifest edits.

Existing sealed card blobs provide actual evaluation opening/drawn copies;
`growing_pool_exposure.py` adds committed choices by participant, decision kind
and action type with replay links and explicit zero-choice participant totals.
These are evaluation observations, not training exposure or mechanic resolution
counts.

`growing_pool_analysis.py` reconciles the full 800-game, 200-callback schedule
against sealed Etude declarations, metrics, EvaluationResults and card/replay
blobs. It validates source/export/registration/world/input/deck/trace identities
and computes per-seed, per-panel whole-deal paired gains. The JSON binds bootstrap
seed 2459001, 2,000 resamples and health metric coordinates, including sampled
gradient/parameter changes at updates 1/25. The evaluator owns the 95% interval.
The callback seals its existing EvaluationResult alongside replay/card blobs;
analysis checks retained replay receipts without executing native replays.

`complete` describes schedule and health-evidence completeness, not scientific
acceptance or healthy values. `modest_check` separately applies the primary
anchor criterion: insufficient primary/health evidence is unknown; complete
evidence yields true/false. Missing secondary panels make reporting incomplete
without adding a score gate. Every report retains the unresolved training-exposure
boundary. Controlled sealed fixtures cover complete results, wholly absent runs,
missing exports/callbacks/games/replays, identity contradictions and health failures.
Their synthetic gains are software evidence only. No games, training or empirical
learning result are established.

## Workflow state and retained evidence

Training's shared-effect and TLA-pilot memory, Eval memory and
`experiments/tla_pilot_analysis.py` informed the analysis boundary: descriptive
curves are not paired gains, missing work cannot shrink the exam, and allocated
input capacity does not guarantee ordinary-game coverage. This adopts none of
MTG-213's cutoff, frozen criteria or spending authority. Detailed reconciliation
is retained at `190b0c603d70e790f2217c61c970c38f82544d08:scratch/questions.md`.

Jack Heart's 2026-11-04 steer accepts workflow installation and Wave migration
inside MTG-245. Seven discoverable graphs and their domain skills now implement
three Coverage revision loops, a reusable engine optimization loop, and one
experiment lifecycle with optional software. Builtin code remains unchanged;
experiment software verification cannot complete the empirical parent outcome.
`docs/workflows.md` owns boundaries/defaults and links supported migration receipts.
Engine → Coverage preserves its ID and memory; new Engine performance and
Infrastructure Waves have supported creation receipts. Training/Eval consolidation
still needs a surviving-identity decision; their memory paths, chapters and
PR #421 contributions remain intact. Shared defaults are authored in this branch,
not yet installed on main. Linear provider synchronization is rate-limited.

The exploration's 50/30/20 example and executable proposal's 50/25/25 remain
illustrative; neither ratio has been accepted by Jack Heart.

## Remaining work and decisions

1. **Exposure acceptance, unresolved:** per-card legal opportunities, effect
   resolutions and training per-card exposure are unavailable. The current
   evaluation draws/action counts support a narrower check; they do not close
   the original training-exposure requirement. Review must decide whether that
   narrower evidence suffices for the first handoff or shared capture must be
   extended first. Any broader implementation coordinates with Tasks 01/03;
   neither their existence nor selection of A approves deferral. No coverage
   claim follows from deck membership, pass counts or software checks.
2. **Integrated campaign, Task 04:** freeze its actual engine snapshot, author
   complete Adventure/prepare lists and exact shared policy-probe scenario IDs,
   review mana/curve/interaction/synergy, and accept numeric bounds before an
   authorized run. The later planeswalker revision retains anchors and old-family
   tests but requires its own authored manifest. The table is a specification,
   not delivered batch support or executable probes. Runtime admission, actual
   learning, exposure, performance and failure evidence remain unmeasured.
3. **Validation/review:** gate owns the remaining regime/discovery and affected
   CI suites required by `docs/development.md`; demo reviews proposed lists,
   shares, budgets and exposure acceptance. Prior CI collection found 370 tests
   without collisions at the baseline revision. Existing native sampler tests
   own seeded recovery/determinism; protocol tests cover ticket order. The
   checkpoint schedule now has one owner, `Budget.export_updates`, shared by
   scheduling and missing-export reporting; frozen JSON/identities are unchanged.
4. **Workflow migration review:** choose the surviving Training/Eval identity
   for Experiments and consequential cross-boundary memberships. Existing
   Training/Eval IDs and paths remain unchanged; do not create a duplicate Wave.
   The supported Engine → Coverage move and two new Waves are applied locally;
   MTG-241/243/244 now belong to Coverage without replacing their graphs/PRs;
   retain pending Linear sync/title updates in the receipt. Shared defaults and
   graph files await this Task's normal delivery. No graph was launched to test
   setup, no campaign ran, and existing Task graphs remain captured independently.

## Retained implementation assumption

Existing sealed Etude training metrics supply numerical health and parameter
changes, keyed by declaration seed rather than display names. Missing metric
coordinates remain unknown. This reversible engineering choice supplies no
training per-card exposure and does not settle its acceptance boundary.

## Workflow implementation assumptions — 2026-11-04

The unambiguous migration is Engine → Coverage (same Wave identity), plus new
Engine performance and Infrastructure responsibilities. Training/Eval consolidation
has no selected surviving identity. It is consequential because PR #421 still
edits Training memory; retain both identities, chapters and memory paths for
review rather than create a duplicate Experiments Wave. The authored experiment
graph can serve both legacy memberships meanwhile. No backlog execution follows
from membership changes. MTG-230's authoring tooling and MTG-231's learned-belief
work cross boundaries; leave their membership for review.

The active Delete — do not maintain list is in `scratch/explore-design.md`.
Obsolete installation proposals were removed from ambient notes; the durable
workflow documentation and dated receipt retain implementation and migration
ownership. Workflow compilation and domain checks now have separate entry points,
so controlled boundary fixtures need no CLI launch or shared-state mutation.

Check (2026-11-04): `uv run --extra dev --extra artifacts --extra evaluation pytest tests/training/test_growing_pool_protocol.py tests/training/test_growing_pool_analysis.py -q` → 48 passed; `uv run python scripts/check_loopflow_workflows.py` → seven graphs compiled with all references and boundaries valid; four controlled invalid delivery fixtures rejected; scoped Ruff and `git diff --check` passed. Gate owns remaining affected suites; demo owns exposure/settings and consequential migration judgments. `lf context` remains within both limits; no campaign ran.
