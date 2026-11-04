# MTG-245 — growing-pool protocol

Updated 2026-11-04 from implementation at `6fcc65646503501ba698efc31207f3c3ca79fb72`,
Jack Heart's accepted exploration choice, and related training/eval evidence.
The Task has no Wave binding; related Wave memories remain read-only.

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
rejects revisions missing selected protocol bytes. Both Run declarations were
verified at the prior HEAD; this analysis change must be committed before its
source-bound revision can be selected for a campaign.

`growing_pool.py` supplies canonical Etude discovery and four policy/deck/seat
legs per deal. It preserves missing export slots for each supplied TrainingRun
and missing game slots for each returned evaluation result. Checkpoints require
the exact regime and raw/initial role as well as ordinary world/input admission.
Fixed heuristic source identity excludes manifest edits.

Existing sealed card blobs provide actual evaluation opening/drawn copies;
`growing_pool_exposure.py` adds committed choices by participant, decision kind
and action type with replay links and explicit zero-choice participant totals.
These are evaluation observations, not training exposure or mechanic resolution
counts. No games, training or empirical learning result are established.

## What reconciliation changed

The earlier readiness wording conflated a runnable declaration with complete
reporting, and the MTG-245 handoff with Task 04's eventual experimental outcome.
The protocol now separates these boundaries and retains the gaps below.

Related evidence: Training memory's shared-effect and TLA-pilot sections at
`8e8d702f8`, Eval memory, and `experiments/tla_pilot_analysis.py` distinguish
paired seat/deck execution from paired checkpoint analysis. Descriptive curves
cannot establish improvement; missing runs/matches must not shrink the exam.
V7 capacity allocations likewise do not guarantee ordinary-game coverage.
These lessons apply without adopting MTG-213's cutoff, frozen criterion or
spending authority. The available `origin/main` reference matches this checkout's
base; no newer upstream integration was established in this reconciliation.

Jack Heart's subsequent 2026-11-04 steer accepts the workflow/Graph and four-Wave
work inside MTG-245; `scratch/explore-design.md` records the complete accepted
scope. This supersedes the earlier separate-Task proposal. Installation and
migration remain unimplemented; the existing Graph owns that work. The exploration's
50/30/20 example is illustrative; the executable proposal remains 50/25/25, with
neither ratio accepted by Jack Heart.

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
3. **Validation/review:** gate retains the regime/discovery suites required by
   `docs/development.md`; demo reviews proposed lists, shares, budgets and the
   exposure choice. Prior 26-test evidence is retained in this file at
   `6fcc65646503501ba698efc31207f3c3ca79fb72`. Existing native sampler recovery/determinism tests
   own seeded draws; the new protocol tests cover ticket serialization/order.
4. **Accepted workflow implementation, MTG-245:** implement discoverable skills
   and graphs with three Coverage code loops and one experiment lifecycle;
   document the four core Wave boundaries/defaults, keeping Academy separate.
   Refresh authoritative state before supported identity-preserving migration;
   retain migration receipts, unresolved mappings and proof that live work,
   Task/PR identities, captured graphs and frozen evidence survive. Preserve
   PR #421's training-memory work and reserved empirical decisions. This is
   outstanding implementation for the existing Graph, not a completed setup
   or authority to launch scientific campaigns. Full scope/attribution is in
   `scratch/explore-design.md#accepted-execution-scope--2026-11-04`.

## Analysis implementation cut — 2026-11-04

Delete — do not maintain (completed): the inline evaluation bootstrap seed in
`growing_pool.play`, obsolete “analysis unimplemented” notes, the fabricated
missing-seed `TrainingRun`, and independently stored `Comparison.expected_games`.
No exclusive fixtures need removal: absent-run tests exercise the surviving
schedule directly. Required JSON parameters own bootstrap/health sampling;
comparison slots carry their actual panel and derive the game count. Callback
and analysis share policy-source identity, preserving its hash format. Replay
validation consumes the outcome row directly. Keep descriptive Etude curves,
the comparison schedule, frozen worlds and all prior evidence. No remaining
protocol deletion targets; workflow migration targets require fresh inspection.

Reversible assumption: consume existing sealed Etude training metrics directly
for numerical health and parameter changes, using the declaration's seed (never
parsing display names). Missing metric coordinates remain unknown. Seal the
existing `EvaluationResult` alongside each callback's card blobs so offline
analysis can validate registrations and replay outcomes through their owners.
This adds no games or training and does not resolve training exposure acceptance.

`growing_pool_analysis.py` now reconciles both declared seeds (including a wholly
absent run), exports, callbacks and every game against sealed Etude evidence.
The callback seals its existing EvaluationResult beside replay/card blobs;
analysis validates source/export/registration/world/input/deck/trace identities,
then computes per-seed, per-panel whole-deal paired gains. Training declarations
and sealed numerical metrics supply health evidence, with sampled gradient and
parameter deltas at updates 1/25. Complete reporting and the primary modest
criterion are distinct; missing secondary panels are reported without becoming
a new score gate. Every output retains the unresolved training-exposure boundary.
No games, native replays, training, jobs or spending were launched.

Check (2026-11-04): `uv run --extra dev --extra artifacts --extra evaluation pytest tests/training/test_growing_pool_analysis.py tests/training/test_growing_pool_protocol.py -q` → 48 passed; scoped Ruff and `git diff --check` passed. The earlier affected CI invocation collected 370 tests without collisions. `lf context` reports scratch within both limits. Gate owns the remaining affected suites; the existing Graph must return to implementation for the accepted workflow/migration scope before claiming that outcome ready.
