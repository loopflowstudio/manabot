# MTG-122 — Is a simpler full game a useful learning diagnostic?

**2026-10-14 · Active execution goal; scientific allocation pending.**
Jack Heart requested running and analyzing this experiment through the public
Etude SDK, replacing corresponding local experiment infrastructure. The previous
design-only restriction is superseded for implementation and bounded software
proof. Scientific machine/budget remains awaiting confirmation; proposed protocol
choices below are not measured results. Gameplay complexity and catalog size stay
separate contrasts; curriculum transfer remains separately selectable.

Main 9576dc40 is integrated at fcbea2f8. Academy namespace separation,
cross-balanced collection and selected-roster mirror admission now exist; the
historical source inventory below describes 43c6e479 and must not be treated as
current missing-work authority. Simple-pool native semantic authoring and common
catalog admission remain required. Public etude-sdk 0.1.0 is pinned; Python 3.12
is the minimum. SDK tracking/checkpoint/offline-report admission passed, not yet
an end-to-end training migration. The native w4 extension now builds/imports.

## Outcome and demo

The useful result is a plot answering whether the *same learner* makes reliable
full-game progress in a simple environment, even if Allies/Lessons progress is
weak. Each pool gets its own initialization, frozen control and fixed opponents.
The dashboard shows improvement, not a misleading league table across worlds.
A negative or budget-infeasible result is an acceptable scientific disposition,
not completion of the chapter's improvement outcome.

Proposed entry point (not implemented):

```bash
uv run python -m experiments.runners.environment_complexity plan --out .runs/mtg122-plan
uv run --extra notebook python -m experiments.runners.environment_complexity report --input .runs/mtg122
```

`plan` resolves declarations and prints identities, cell counts, resource bounds
and unresolved admission requirements without collecting games. `report` reads
retained evidence into one create-once editable `comparison.ipynb` and read-only
HTML. It never trains, evaluates, rents hardware or overwrites notebook edits.
A completed report shows three pool panels, four deck-role cells per anchor,
paired training-seed changes, cost/decision curves and retained failed attempts.
Without results it says **not run**, never zero strength.

## What the existing system actually supports

- `Experiment.resolve()` and its `Environment`, `Model`, `Pipeline` and
  `RunControl` components already own declarative recipes and provenance.
  `TrainingRegime`, `ExperimentSchedule`, `run_experiment`, `TrainingRun` and
  VerifyStore own execution and persistence. Reuse them; no new experiment DB,
  worker scheduler, checkpoint format or tracker authority.
- `managym/src/semantic/content.rs::AuthoredRuntime::matches` requires the exact
  multiset of authored deck **and sideboard** pairs. A subset or same-deck mirror
  does not select the authored compiled pack. `flow/setup.rs` then uses general
  content. This is a real semantic/identity change, not just a smaller deck.
- `SemanticCardEncoder` (`manabot/model/semantic_cards.py`) accepts only
  `ur-lessons-vs-gw-allies` and constructs its transport-ID lookup from that
  exact authored matchup. `BoundSemanticPack.bind` can join another manifest by
  checked registry names, but the ordinary encoder does not use the runtime
  subset manifest. Reusing its integer IDs on a general-pack subset is unsafe.
  Its token vocabulary is also catalog-dependent: independently shrinking
  catalogs would vary both representation and parameter count.
- `checkpoint_world`/`validate_checkpoint_world` bind exact setup pairs, actual
  content, rules and input schema. `validate_agent_setup` rejects another pair,
  including mirrors of a trained cross pair. This is deliberate admission, not
  an obstacle to remove with a bypass.
- `monitor_evaluation._manifest` explicitly requires `selected_match()`.
  `arena.match.play_cell` supports the selected cross matchup or historical
  Interactive mirror, not arbitrary pools or the four deck-role matrix.
  `MonitorProtocol` has one anchor, four-leg summaries; the study protocol has a
  closed discriminator. New science must not masquerade as an existing study.
  ETU-125's requested matrix is the behavioral contract to reuse; its integrated
  implementation is **not present in this base**. Adopt it when available rather
  than creating a competing evaluation path.
- `SeatRoutedCollector` initializes every stream with the same ordered match
  and alternates only the learner seat. That balances learner seats, not each
  deck across play order. Its `collect(num_steps)` takes that many transitions
  **per stream**. Balanced deck assignments need explicit collector support;
  neither balance nor batch size can be inferred from four streams alone.
- `teacher1_evidence.runtime_fingerprints` takes explicit match and observation
  arguments, but defaults to Interactive. Every pool/cell must pass actual
  arguments. Never rewrite frozen fingerprints to make them fit.
- `GameSession`, `Trace`, `_persist_attempt`, and the table's
  `experience_protocol.py` already own player execution and durable attempt
  records, with private canonical replay stored in `AttemptStore`. Canonical
  historical decisions are not every automatic pass or engine microstep.
  The experiment uses arena command journals/replay, not table decision counts
  as a learning clock and not a new session/identity store. This design proves
  neither policy-free recovery nor crash durability of arbitrary execution.
- The selected-world checkpoint loader is `sim.flat_mc.load_checkpoint_agent`;
  it validates the world before loading weights and checks compiled semantic
  input. Synthetic fixtures can test that path, but cannot admit a real trained
  candidate or establish completed human games.

Native inspection was unavailable: this checkout has no `managym._managym`
extension. Static source findings above are not a successful subset runtime
probe. Building/running admission fixtures belongs to later software work.

## Chosen pools and fixed training composition

Use implemented cards, ordinary 20-life full games, no reward shaping, no
scenario injection and no hand editing. Two explicitly named deck roles per
pool train **cross-role current-policy self-play only**, with equal seat/deck
assignment. Four streams freeze `(seat-0 deck, seat-1 deck; learner seat)` as
`(A,B;0), (A,B;1), (B,A;0), (B,A;1)`. Each stream retains that assignment on
reset and recovery, with sideboards traveling with decks. This gives equal
learner-transition quotas per deck/play-order cell, not equal completed-game
counts or optimizer exposures. Record those separately. Extend the existing
collector's setup/reset/receipt contract (or reuse ETU-125 support) rather than
building another collector. Mirrors are evaluation only; no matchup mixing.

Draft simple-deck shape (counts; empty sideboards). Jack Heart requested on
2026-10-13 that L1 omit haste and first strike. Replacement vanilla definitions
remain to be selected; this is not a frozen or runnable decklist:

| Card | L1 swift-rw | L1 heavy-rw | L2 swift-rw | L2 heavy-rw |
|---|---:|---:|---:|---:|
| Mountain | 9 | 9 | 9 | 9 |
| Plains | 8 | 8 | 8 | 8 |
| Vanilla low-cost creature (TBD) | 8 | 4 | 6 | 2 |
| Vanilla mid-cost creature (TBD) | 8 | 8 | 5 | 5 |
| Gray Ogre | 7 | 11 | 4 | 8 |
| Lightning Bolt | 0 | 0 | 4 | 4 |
| Yip Yip! | 0 | 0 | 4 | 4 |
| **Total** | **40** | **40** | **40** | **40** |

L1 means basic lands plus creatures with no abilities: no haste, first strike,
or other keyword, triggered, activated, or static abilities. L2 shares that
vanilla base and replaces eight creatures with removal and a combat trick.
Do not strip abilities from existing card definitions to satisfy this constraint.

The earlier Raging Goblin / Youthful Knight selection is withdrawn. Static
inspection of this base finds Gray Ogre and Craw Wurm as explicitly registered
vanilla creatures; it does not establish suitable low-cost red/white replacements.
Select implemented vanilla definitions and reconsider colors/curve together
before freezing the lists. The counts above retain the draft's 40-card shape,
not an approved card choice. Any need for new native cards is a scope decision,
not authorized by this revision. Semantic authoring/admission remains required.

Yip Yip! retains its real Ally-dependent flying clause in L2; the final vanilla
roster must be checked for Ally subtypes rather than silently assuming that
clause cannot apply. High copy counts are a diagnostic custom format, not
sanctioned constructed decks.

L3 is **unchanged** `MatchHypers.authored("ur-lessons-vs-gw-allies",
"ur_lessons", "gw_allies")`, with each sideboard traveling with its deck.
Freeze resolved counts from `content/semantic/v1/two_deck.source.json`:

- UR Lessons (41): Island 9, Mountain 8, Tiger-Seal 2, Otter-Penguin 2,
  Fire Nation Cadets 2, First-Time Flyer 2, Forecasting Fortune Teller 1,
  Dragonfly Swarm 1, Firebending Lesson 4, Igneous Inspiration 2, Pop Quiz 2,
  Divide by Zero 2, It'll Quench Ya! 2, Accumulate Wisdom 2.
  Sideboard: Firebending Lesson 1, It'll Quench Ya! 1, Accumulate Wisdom 1.
- GW Allies (40): Plains 9, Forest 8, Water Tribe Rallier 2,
  Invasion Reinforcements 2, Compassionate Healer 2, Earth Kingdom Jailer 2,
  White Lotus Reinforcements 2, Earth King's Lieutenant 2;
  Kyoshi Warriors 2, Badgermole Cub 2, Suki, Kyoshi Warrior 1,
  South Pole Voyager 1, Allies at Last 2, Yip Yip! 1, Fancy Footwork 2.
  Sideboard: Yip Yip! 1, Fancy Footwork 1.

Exported definitions use only canonical registry names, without aliases. Preserve the authored 41/40 asymmetry. Simple roles
are never called Allies/Lessons. Complexity is a bundle, not a randomized single
mechanic: curve, creature density, mana colors, deck size, opponent strength and
horizon differ. No raw-score comparison identifies complexity causally.

Freeze a vector of complexity descriptors, not an invented scalar index:
unique main/sideboard definitions, nonland programs/opcodes/target roles,
keyword and triggered/activated/choice families, mana curve, creature/spell
fraction and token closure. Measure game turns, commands, surfaced decisions by
kind, legal-support quantiles, forced-choice fraction, decision-to-terminal
lengths, draws and truncations with anchor games. Keep those empirical measures
separate from authored complexity descriptors.

## Identity-safe architecture before any score

Choose a **fixed research semantic vocabulary/catalog** for all three arms:
the unchanged full authored definitions plus the implemented simple cards above.
Use the existing GRU semantic encoder, visible-state attention and flat offer
head; no belief, history, compound decoder or architecture treatment. The same
ordered token vocabulary, definition programs and learnable tensors are used in
every arm. Unused definitions remain in the catalog; this avoids smaller-pool
models quietly becoming smaller models.

Native rules remain authoritative. Register an explicit research content/setup
binding for the simple decks with reviewed compiled semantics, and admit full
Allies/Lessons mirrors against the **original full compiled content**, not the
general fallback. This may reuse delivered ETU-125 support; otherwise it is a
Rules-boundary prerequisite. Product decklists, default selection and original
compiled content bytes remain unchanged. No globally relaxed name-subset match.

The policy catalog is not the native pack. Its binding joins the *actual native
manifest's* CardDefIds to canonical research catalog rows. The catalog and each
mapping are immutable and hash-bound. Manifest meanings, exact programs and
referenced-definition closure must agree; equal names/dimensions alone do not
admit a binding. Rebuild derived mapping buffers per admitted runtime, never
copy transport-ID buffers from another world. Reject missing/unrecognized
visible definitions, changed rules/programs, sideboards and observation caps.

Persist a finite admitted evaluation-setup roster with new research checkpoints:
cross pair and both role mirrors for that pool, with exact content/input
bindings. Training setup and evaluation roster are distinct fields. The ordinary
loader and arena validate every requested pair against it; no evaluator monkey
patch or `validate_agent_setup` suppression. Historical checkpoints retain
strict original admission and are never retroactively granted mirror/transfer
rights. Full-world checkpoint serving must still validate the authored pack.

This is additional configuration/binding support, not proof of model portability.
All three new experimental arms use it. They are not byte-compatible replacements
for the existing ETU-118 baseline; any full-world reference stays separately
labeled. Freeze source/native/lockfile/runtime hashes, full content and catalog,
input/action ABIs, setup roster, regime/protocol digests, seed plan, exact raw
artifact bytes and inference profile. No scientific reuse across source freezes.

## Proposed learner and resolved plans

One `Experiment` declaration, three environment cases, three independent seeds
`12251, 12252, 12253`. Rotate pool execution order by seed: L1/L2/L3,
L2/L3/L1, L3/L1/L2. Shared seed pairs initial learnable tensors; assert their
hashes before collection. Matching seed numbers do not mean matching physical
deals or trajectories across different pools.

Start from the frozen small recipe in
`experiments/regimes/early-learning-rate-mini-source.json`, **not** its LR-screen
arms. Pin width64, two attention layers, four heads, scalar masked-mean value,
no history/belief, CPU float32, one thread, four streams, 64 transitions per
stream/update (256 total learner transitions/update),
terminal rewards and Adam. Replace only semantic binding with the common
research catalog and environment with each pool. Export all resolved defaults,
including encoder/padding/expansion, and assert equal learnable parameter counts.
Freeze common observation caps at 60 cards/player, 40 permanents/player,
64 actions, two focus objects and 32 event slots (history remains off), as in
the source recipe. Capacity overflow is an admission failure, not truncation.
The four-stream deck-assignment roster above is identical in structure across
pools; it is an explicit change from the source recipe, not an existing feature.

Retain the recipe's Ataraxos-move settings: policy/value lambda .5/.8, inclusive
.75 quantile with .01 floor, actor-and-critic filtering, clip .2, collection KL
.1, action-type-uniform reference, max gradient norm .267 and EMA .999.
Learning rate is clamp(.5 / iteration^1.1, 5e-6, 1e-4); tau is
.05 / iteration^.3. Absolute collect/update iterations include empty-filter skips;
never compress that clock to optimizer calls or rebase it by pool. Export EMA
for audit, but score raw only. Evaluation uses stochastic legal policy sampling,
CPU/one thread/batch1/one forward pass, no search, against source-pinned greedy
and uniform-random anchors. Freeze their actual implementations and tie-breaking.

A single self-play stage targets common **N** updates. Scientific checkpoints
are 0, N/2 and N (N divisible by 100; mid/end exact absolute milestones).
Use existing step cadence; do not turn elapsed scheduling into another treatment.
TrainingRun retains environment decisions, learner transitions, optimizer calls,
selected exposures, filtered/empty updates and numerical-health metrics. Equal
updates or collected transitions are not equal optimizer exposures. Completed
endpoints collect 256N learner transitions (256,000–2,560,000 over the proposed
N range), before filtering; native microsteps and opponent actions are separate.

The no-update control is each seed/pool's exact initialization export, not a
zero-LR learner that still changes buffers or EMA. Re-evaluate those saved bytes
through the ordinary loader/arena at the mid/end cohorts. Store zero optimizer
calls and unchanged weights without inventing a completed training stage.
Initialization evaluation can serve both arms once; later control evaluations
are real separate attempts. Shared policy and deal/action seeds should reproduce
initial predictions exactly; drift is an instrument failure, not learning.

`experiments/runners/environment_complexity.py` will own the declaration and
thin CLI; reusable matrix/admission/report changes stay in their existing owners.
The resolved plan embeds exact definitions, baseline bytes, ordinary regimes,
a new explicitly admitted complexity protocol, schedule, hardware, calibration
receipts and expected cell manifest. No copy of the historical depth-screen
protocol with a changed label.

## Evaluation, cost and admission protocol

### Matrix and pairing

For each pool/checkpoint/anchor, score four cells independently:
model A vs anchor A, A vs B, B vs A, B vs B. A/B mean swift/heavy in L1/L2,
and GW Allies/UR Lessons in L3. Each cell has **50 held-out deal blocks × two
play orders = 100 games**. Both anchors receive this cohort: 800 games per
checkpoint. Candidate and frozen control share exact within-pool assignments,
deal seeds and policy/anchor random streams. Deal blocks keep both play orders
and all four cells together in uncertainty calculations. Training gets disjoint
seed namespaces; exported actual seeds are checked for collision, not assumed
disjoint merely because namespace labels differ.

Reserve separate disjoint 50-deal families for initialization, midpoint and
endpoint. Freeze all before scoring; endpoint is not inspected to select a
checkpoint. Controls evaluated on each new family supply that family's exact
initialization baseline. Consequently each milestone's learning contrast is
trained minus its **same-deal frozen initialization**, not subtraction of two
unpaired cohort means. Original initialization scores remain visible separately.

Planned scientific game count: nine initialization cohorts + nine trained
midpoint + nine trained endpoint + nine frozen midpoint + nine frozen endpoint,
each 800 games = **36,000**, plus bounded calibration/anchor diagnostics.
No hidden reduction from 100 total games to 25 per matrix cell. If that floor
does not fit, the scientific plan is unadmitted; do not launch a smaller cohort
and call it accepted evidence.

### Hardware and budget — proposed, not reserved

Choose a **local CPU run after approved laptop cohorts release their resource**,
using configured hardware and host/evaluator leases. No Mini slot (ETU-125 owns
its separate selection), no GPU rental and no displaced existing work. Proposed
ceiling: **48 occupied wall hours and 64 additive process hours**, at most one
single-thread learner and one single-thread evaluator. No dollar allocation;
report dollars unavailable unless an explicit host price is frozen. This is a
budget proposal only, not permission to wait/queue/start work automatically.

Reserve process time before admission: calibration 1 h, scientific learning 18 h
(2 h maximum per pool/seed), evaluation 36 h, setup/export/report/recovery 9 h.
All attempts, failures, clean-play/replay and artifact transfer count. Wall and
process ceilings both bind; overlap is not free. No carryover of another Task's
budget. Future explicit authorization and resource availability are required.

Calibration would require separate authorization; its proposed bounded phase is: all three
pools, a fourth seed 12250, same model/streams/rule, up to 100 updates per pool,
reload/complete games in each matrix cell against both anchors, plus greedy vs
random and same-anchor checks. Use at most one hour total; failures stay retained.
No checkpoint from calibration enters scored science. Record setup/export/replay,
per-update block timing and per-cell game time, not bare forward-pass rates.

Resolve N = largest multiple of 100 no greater than 10,000 for which a 1.5×
margin on the slowest measured per-update block cost fits every 2 h learner cap.
Also require the full 36,000-game projection (1.5× slowest per-cell measured
mean, with measured setup/replay costs) to fit the 36 h evaluation reservation,
and full projected critical path to fit 48 h. N < 1,000 or incomplete calibration
means **budget-infeasible**, not a smaller silent launch. Timings extrapolate;
they do not promise sustained throughput. Commit resolved counts, absolute
milestones, cutoffs and receipts before any scientific score is revealed.

Game watchdog: 120 seconds/10,000 Commands, unchanged across pools. Timeouts,
nonfinite losses, illegal Commands, hidden-truth dependence, identity drift or
replay mismatch retain the attempt and mark the affected cohort incomplete.
An incomplete cell gets no rate; missing cells/seeds suppress pooled method
claims. No best-seed replacement, early strength stopping or automatic restart.
Recover only an explicitly admitted exact state within the original budget;
otherwise stop with missing/censored evidence. Report engineering failures apart
from negative learning. Endpoint is last planned raw checkpoint, not best score.

## Analysis and reporting

Before running, proposed prediction: L1 gains at least 10 percentage points in
mean greedy score over frozen initialization by N, positive in all three seeds;
L2 gains at least 5 points; L3 gains less than L1. These are design predictions,
not evidence or historical diagnoses. The endpoint is the primary contrast;
midpoint and time-to-progress are descriptive, not sequential significance tests.

Per pool/anchor/seed/cell report wins/draws/losses and score = win + half draw,
paired change versus frozen control, and uncertainty. Pool summaries equally
weight four cells then seeds, never pool games as independent training seeds.
Show all three paired seed effects, their mean and a 95% t interval (df=2)
explicitly labeled small-sample/assumption-sensitive. Separately show paired
whole-deal bootstrap intervals conditional on each checkpoint, using 10,000
replicates and fixed analysis seed 12290. Cross-pool contrasts pair training
seeds but never pretend different deck worlds share identical deals.

A **useful positive diagnostic** requires complete legal/replayed cohorts,
positive greedy improvement in all three seeds, mean improvement ≥10 points
and a seed-level lower interval above zero, plus no demonstrated random-anchor
regression (report its interval rather than treating a nonsignificant difference
as equivalence). This is a conservative descriptive gate, not a multiplicity-
corrected method-selection test. Do not search many card lists for a winner.
Report each L2/L3 contrast even if the simple result fails this criterion.

Time to declared progress is the first *observed* midpoint/endpoint meeting the
10-point greedy improvement threshold for that seed, confirmed at endpoint.
Report interval censoring between observed checkpoints; not reached is right-
censored, never zero time or an extrapolated failure probability. No adaptive
extra milestone. Freeze cost cutoffs from calibration; at each cutoff use the
last complete checkpoint available, only over overlapping observed ranges.
No interpolation or invented initialization/history.

Show three axes: actual learner decisions/transitions and optimizer exposures;
active collection/update seconds; inclusive elapsed cost (setup, exports,
evaluation/replay, transfers and failures). Charge shared initialization/control
and calibration once in campaign accounting and explicitly allocate shared
measurement overhead when showing per-arm costs. Sum parallel process cost
separately from occupied wall time. Fixed controls' training exposure stays zero.
Do not plot them as if they consumed the learner's samples.

The notebook leads with status/freshness, this within-pool improvement and all
four role cells, then learning diagnostics and confounds. Trace every point to
run/checkpoint/protocol/arena hashes. Explain greedy baseline asymmetry, action
mix, land/curve differences, horizons, filtered exposure and inference latency.
Anchor-vs-anchor calibration is context, not a correction factor that makes raw
scores comparable. Policy entropy is not competence. Neither a simple success
nor a full-world point estimate promotes the product baseline (ETU-118 owns that).

Extend `docs/experiment-metrics.md` for formulas, clocks, denominators, censoring
and unavailable states. Use existing create-once notebook/report regeneration,
optional W&B projections and private content-hashed S3 publication/readback.
TrainingRun/VerifyStore and immutable arena evidence remain authority. Missing
credentials mean publication unavailable, not fabricated archival; keep local
originals. Never publish hidden worlds/private replay targets into viewer reports.

## Separately selectable curriculum follow-up

Do not combine transfer with the initial contrast or choose a transfer source
from whichever seed happens to win. Proposed later question: at equal total
training compute, does L1 pretraining improve full-world learning versus scratch?
Use three fresh paired seeds, L1 pretraining for 25% of a separately calibrated
training allocation then L3 for 75%, versus L3 scratch for 100%. Retain the same
capacity/rule/evaluation matrix; include pretraining, import, setup and evaluation
in total-cost accounting and report exact exposures. Optional L2 transfer is a
new arm requiring its own budget, not an automatic expansion.

Reuse the fixed research vocabulary only after validating every canonical
program meaning and actual native transport binding. Transfer learnable weights
only; reset Adam/EMA/schedules/collector streams at the declared target boundary,
so this is a specific warm-start intervention, not process continuation.
The exact reset convention is frozen before scoring and listed as a confound
versus uninterrupted scratch. Preserve source artifact/world identity and create
an explicitly target-bound imported artifact with lineage, never rewrite the
source `world_binding`. Existing strict import/loader support is not evidence
that cross-world transfer is admitted; implement and test that distinct contract
only if this follow-up is selected. Use one identical held-out **full-world**
four-cell matrix across transfer and scratch arms. Simple-world scores do not
enter the transfer outcome. No pretrained bytes or transfer claims exist here.

## Alternatives, failure modes and implementation sequence

Rejected: legacy numeric-only simple models versus semantic full models (changes
representation); stripping abilities from authored cards (new rules pretending
to be old); independently shrunken policy catalogs (different parameterization);
reusing default fingerprints/IDs; mixed mirror training (ETU-125's experiment);
or a second private runner/store. The chosen fixed-catalog route costs explicit
admission work but isolates the learner and preserves the real full world.

Wild success: a reproducible simple-game learning curve becomes an inexpensive
positive learning diagnostic, and highlights where the full game warrants a
focused next experiment. Wild failure: catalog admission silently changes rules,
weak anchors reward artifacts, or evaluation consumes the entire budget. Exact
bindings, per-cell controls and all-in calibration address those failures; three
seeds and two simple anchors still cannot establish general playing strength.

1. **This slice — design/capture only.** Resolve scientific intent, cards,
   identity gaps and proposed admission/budget. No implementation or calibration.
2. **Identity and matrix prerequisite.** Reuse/adapt ETU-125's matrix if delivered;
   otherwise extend the existing arena/protocol. Bind finite setup rosters,
   native simple content and one common policy catalog through ordinary
   checkpoint admission. Add the four-stream cross-role assignment contract to
   ordinary collection/reset/recovery and verify per-cell transition quotas.
   Preserve authored full-world behavior and old readers.
3. **Reproducible declaration/report software.** Add the typed complexity
   protocol, side-effect-free plan export, frozen-control evaluation, calibrated
   count resolver and evidence-only notebook/HTML. Reuse execution/publication.
4. **Only under a later bounded authorization:** native/software admission,
   calibration, pre-registration commit and unique launch. Finish all declared
   cohorts or report incompleteness; publish originals and analysis. Scientific
   and chapter acceptance remain separate from software delivery.

**Delete — do not maintain:** the live W&B implementation in
manabot/infra/experiment.py and Trainer-specific W&B branches are removed in
favor of the SDK; historical readers and serialized fields remain evidence.
The experiment's old scheduler/tracking/report ownership must be replaced by
Etude rather than duplicated; domain recipe, checkpoint admission and game
replay semantics remain Manabot's responsibility. In the later
cutover, replace the single-catalog construction assumption in
`SemanticCardEncoder` and selected-match-only routing in the shared monitoring
path with explicit validated bindings. Preserve the historical authored recipe
and four-leg evidence reader as supported contracts, not competing execution
implementations. Do not add a private subset encoder, ID alias table or a second
matrix evaluator. No frozen plans, reports, catalog bytes or receipts are removed.

## Done when / headless gate design

Future tests (names are proposed, not commands claimed to pass now):

```bash
cargo test --manifest-path managym/Cargo.toml --test complexity_content_tests
uv run pytest tests/training/test_environment_complexity.py tests/model/test_world.py -q
uv run --extra notebook python -m experiments.runners.environment_complexity verify-fixture --out /tmp/mtg122-fixture
```

The end-to-end fixture uses deterministic untrained exports through the ordinary
loader, all three pools × four role cells × two anchors × two play orders
(48 complete games), native semantic Commands and exact arena replay. Assert
canonical-ID joins, unchanged product content, same learnable initialization,
normalization/legality, hidden-world invariance and setup-roster rejection.
The collector fixture checks all four deck/learner-seat assignments, sideboards
and reset/recovery, and exactly 256 learner transitions per complete update.
Invalid pack, sideboard, missing token closure, capacity and altered program
fixtures fail closed. Frozen control is reevaluated with unchanged weights.
A separate tiny optimizer fixture checks numerical/coordinate wiring only.
Headless notebook execution must regenerate retained metrics without playing a
game, preserve an edited notebook and show failed/missing cells as unavailable.
No human-play or strength conclusion follows from these fixtures.

Scientific completion additionally requires a calibrated, committed resolved
plan, all attempts/controls, the frozen independent-seed matrix and uncertainty,
verified artifact publication, and an honest usefulness conclusion. Missing
artifacts, unavailable native checks or unapproved resources remain explicit;
none are permission to bypass identity or launch science.

Check (2026-10-10): `uv run python` static JSON/source assertions verified deck totals, 36,000-game arithmetic and 256 transitions/update; `uv run python -c 'import managym'` failed with `ModuleNotFoundError: managym._managym`; native subset/mirror/replay checks remain for the later software gate.

Check (2026-10-14): release native build/import and 19 focused SDK, checkpoint-world and matchup tests passed. First test collection lacked nbclient; rerun with the documented notebook extra passed. No scientific training ran.
