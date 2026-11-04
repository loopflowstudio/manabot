# Growing card pools: the reference league

**MTG-245, 2026-11-04. Prospective; no games, training or paid jobs executed.**
Jack Heart selected exploration A: authored functional decks, frozen sampling
mixtures and stable evaluation anchors. The exact lists, shares and budgets
below are proposed engineering parameters. Task 04 owns measured validation;
Task 08 owns the later campaign. This document grants no allocation.

The executable current-world example is
[growing-pool-245-v1.json](growing-pool-245-v1.json), consumed by
[growing_pool.py](../growing_pool.py). It is a **40-card Elemental proxy league**,
not a claim that the emerging post-rotation Standard pool is ready, that these
are Standard decks, or that a generated population is a metagame. Validation A
must freeze its declared integrated engine snapshot and actual batch decks in a
new protocol before execution. The proxy is runnable from its pinned checkout;
it is not a substitute for Task 04's real training campaign.

## Recommendation and alternatives

Start with a small reference league and fresh training. Keep half the per-seat
mass on old anchors; assign one quarter to new-cohort lists and one quarter to
authored cross-batch lists. This is a comprehensible baseline, **not an empirically
optimal mixture**. The native population sampler already supports it. An author
can explain each list, audit admission failures, and hold the exam constant while
changing learning. The price is author bias and recurring deck maintenance.

Evidence for this starting point is limited but relevant:

- The repository's [six-seed curriculum study](../curriculum.md) found no useful
  benefit from its particular simple-to-complex intervention. It does not test
  this league or today's world. It argues against assuming that staged simplicity
  is necessary before trying ordinary mixed-pool self-play.
- [Deck-quality controls](../exp-08-two-deck-matchup.md) and
  [control competencies](../exp-09-control-competency.md) warn that aggregate
  win rates can conceal deck imbalance and a specific inability to act. Hence
  policy/deck swaps and separate mechanic probes.
- [CoinRun](https://proceedings.mlr.press/v97/cobbe19a.html) demonstrates why
  training performance and generalization need separate environments. Our
  analogue is an immutable experience split; definitions remain visible.
- [Prioritized Level Replay](https://proceedings.mlr.press/v139/jiang21b.html)
  adaptively samples procedural levels using learning-potential estimates.
  That is a plausible later alternative, not evidence that loss-based deck
  sampling works for Magic. It would add selection costs and nonstationarity.

| Alternative | Potential value | Why it is not the first handoff |
|---|---|---|
| Only a few curated decks | Cheapest interpretation and stable anchors | Too little breadth; keep them as half the population |
| Uniform random cards | Cheap negative construction control | Mana, curve and synergy failures confound learning; no assumption that random lists teach useful play |
| Constrained generation (B) | Broader coverage with less list authoring | Needs audited constraints, functional screens and leakage isolation; no generator is implemented here |
| Adaptive challenge search (C) | Find interactions authors missed | Can exploit deck imbalance/bugs; needs a fixed-mixture control at equal total compute |
| Authored reference league (A) | Existing execution path, explicit coverage intentions | Selected; uncertain coverage and author maintenance remain measurable costs |

The v1 choice is falsifiable: if neither seed improves on the fixed anchor panel,
or intended new-cohort choices are absent in ordinary evaluation, investigate
construction, exposure and update evidence before increasing compute. A failure
does not establish that generation or a bigger model would fix it.

## Owners and the executable contract

`managym.Deck` owns main and sideboards, and native `check_decks`/game creation
owns support. All defined non-token cards remain playable regardless of this
study's exclusions or known bugs. `Manifest` owns experimental membership;
`UniformDecks` owns seeded independent per-seat draws; `TrainingRegime` and
`TrainingJob` own training and bounds. `EvaluationRequest` owns execution,
Commands, replay and failure denominators. `GameEvidence`/`card_metrics` own
card captures; Etude owns jobs, blocks and blobs. No engine, model or card
representation is replaced.

The JSON pins native world/card identity, engine source digest, all input
capacities, full model/learning configuration, decks, construction provenance,
tickets, split, deal seeds, update coordinates and budget. Code checks these
before execution. Training job declaration rejects a revision that lacks the
selected source/protocol bytes. Each execution also retains its source bundle,
compiled extension, configuration, checkpoint and runtime receipts. Checkpoints
bind the world, not permission to infer a new training population.

V1 uses explicit input v7, 128 local facts, 128 card rows/player, 64 permanent
rows/player, 512 legal-action rows, 16 focus objects and 32 event rows; every
other axis is serialized in `observation_shapes`. These are bounded allocations,
not coverage guarantees. Overflow is **error**, never a prefix, page, random
fallback or silently dropped game. Native trivial-decision skipping stays on.
The TLA first-64 exception is unrelated. Coordinate prospective input changes
with MTG-243; no frozen input or checkpoint is relabeled compatible.

## Deck authorship and functional construction

An experiment author proposes lists after a mechanic batch lands. A second
review of the list's mana, plan and interaction is part of Task 04's freeze;
native rejection goes to the rules owner. Target authoring cost is at most two
hours per modest batch (record actual minutes and unresolved lists). Exceeding
that target is evidence to consider B, not a reason to generate unreviewed decks.

The initial template has 40 cards, 17 lands, one or two colors, at least eight
creatures and eight spells with printed mana value at most two, at least four
copies of author-identified interaction cards, and at least four copies across
two named synergy/core cards. Main plus sideboard has at most four nonbasic
copies. All nine exact lists and their role annotations are in the JSON.

The four base lists retain the authored Elemental mana bases: seven/eight basics
plus two Fabled Passage. These are approximately nine/ten sources of each color
when Passage is available, not an unconditional untapped-source guarantee.
An author checks color intensity and timing, sideboard/Learn access, and whether
interaction can answer the expected threats. The cross-batch and test variants
preserve lands, complete printed mana costs and card types for each substitution;
the checker derives these from the native definitions. It does not infer that
identically costed cards are strategically interchangeable.

| Entry | Training tickets | Construction / retained purpose |
|---|---:|---|
| anchor-ur | 2 | Exact Elemental UR spells deck and Lesson sideboard |
| anchor-gw | 2 | Exact Elemental GW Allies deck and Lesson sideboard |
| new-ub | 1 | Exact Elemental UB value/interaction deck, proxy new cohort |
| new-gr | 1 | Exact Elemental GR power/pump deck, proxy new cohort |
| mixed-ur | 1 | UR: one Accumulate Wisdom → It'll Quench Ya! from UB |
| mixed-gw | 1 | GW: Haru → Badgermole Cub from GR |
| test-ur-composition | 0 | UR: Pop Quiz → Divide by Zero; known cards, new counts |
| test-ur-card | 0 | UR: one Firebending Lesson → Lightning Bolt; held-out experience |
| test-gr-combination | 0 | GR: Swiftspear → Cadets; Cadets with Tiger-Dillo within one deck |

List legality, functional construction and competitive viability are distinct.
The static audit establishes only the first two's listed prerequisites. Task 04
uses its initialization evaluation as the bounded functional screen: record
opening/drawn cards, committed choices, early lack of action, durations and
failures. No extra screening games are hidden in the budget. If a deck produces
no nonland choices in any completed leg, or a desired family has no observed
choice evidence, report an exposure failure/inconclusive family. Inspect the
retained games; do not replace a weak deck after seeing endpoint scores. Any
changed list starts a new prospective identity. Competitive viability is outside
this pilot; heuristic strength and deck balance are uncalibrated here.

## Sampling, exposure and curriculum changes

The sampling unit is **one deck entry per seat per new game**, independently and
with replacement. Eight ordered tickets give 50/25/25 group mass; repeated
anchor tickets are intentional. Mirrors are allowed. Both seats contribute
complete-game experience; a fixed games-per-stream budget avoids confusing
transition quotas with equal games. In expectation 43.75% of games have at least
one new-cohort deck (1 − 0.75²), and 75% have at least one non-anchor deck.
Neither figure is a mechanic-use rate. Exact lists, entry/ticket IDs, seed and
native reset derivation are reproducible through the existing deck-source owner.

Report counts separately, with missing quantities explicitly unavailable:

| Quantity | Producer and interpretation |
|---|---|
| Planned deck/seat exposure | Manifest tickets; an expectation, not observed gameplay |
| Training games, environment/learner decisions, update clocks | Existing TrainingRun/Etude coordinates; report actual completed counts, including early endpoints |
| Opening and actual drawn copies in evaluation | `card_counts`, via replay-backed native capture; returns/tutors are not draws |
| Committed choices by acting participant, decision kind and action type | New `decisions` table from retained traces, with evidence blob digest and explicit zero-choice participant totals |
| Cast/activation decisions | Subsets of committed action-type counts, not proof of resolution, strategic benefit or opportunity frequency |
| Per-card legal offers, mechanic resolutions, dead-draw reasons | Not available in the retained v1 trace; never inferred from presence or written as zero |
| Training per-card draws/offers/resolutions | Not supplied by this minimal protocol; evaluation exposure is labelled evaluation, not extrapolated into training totals |

`games` retains all four leg slots, validity and attempted flags. Card blobs are
sealed alongside match tables, not in checkpoint storage. Use
`manabot.eval.card_query.card_db` / `CARD_ROLLUPS` for card queries; the pinned
SDK's generic evaluation job does not declare rollups, so `card_db` derives them
once from the retained tables. Use `etude.eval.query.read_blobs` for evidence.
The `decisions` table counts committed choices only; forced native choices are
skipped. A high count of priority passes is not evidence of meaningful mechanic
practice. Separate those action types and inspect examples before a family claim.

Tasks 01/03 own broader history/receipt work. The useful next instrument is
per-family/per-participant **legal opportunity → selected Command → resolved
effect** counts, keyed by native object incarnation and world. Its decision value
is to distinguish rare opportunity from refusal/misplay. The cheapest credible
producer is an extension of existing native capture/replay, shared by training
and evaluation; no label parsing or parallel mechanics taxonomy. That extension
is not required to run the first ordinary learnability check. Without it,
mechanic-coverage claims remain narrower than aggregate learning claims.

Rare families get authored decks and separate directed probes, not inverse-
frequency loss weights. A repeatedly dominant card is capped by list counts and
reported by drawn/choice evidence, then tested in a prospective ablation rather
than banned mid-run. A dead draw may reflect mana, board state or policy failure;
one cannot distinguish those with drawn counts alone. Freeze any adjustment to
tickets, lists, seeds or learning before its run; archive old values and reasons.
No within-run adaptive curriculum is selected.

## Evaluation and held-out lifecycle

Five panels remain distinct: old anchors, expanding pool, held-out composition,
held-out card, and held-out within-deck combination. Each has four frozen deal
seeds and four policy/deck/seat legs. Every policy plays both decks on play and
draw. Deals are common across checkpoint comparisons within a panel; inference
seeds use the evaluator's existing paired derivation. Initialization and updates
8, 16 and 32 play the same full-support heuristic. Update 32 also plays its own
initialization. Missing exports leave expected comparison slots unplayed; an
early endpoint never replaces update 32 or earns its acceptance claim.

- **Held-out cards:** Lightning Bolt never enters a training main deck or
  sideboard, for either participant. Its definition and learning tokens are
  visible in the full world vocabulary. This is no-experience transfer, not
  unseen-description or unseen-rule inference. New mechanics introduced by
  known cards' effects would require additional origin exclusions.
- **Held-out decks/compositions:** all cards may have training experience but
  the exact complete main/sideboard pair has none. Renaming a training deck does
  not create a test deck. One-card variants are deliberately near-distribution;
  they cannot establish broad unseen-archetype generalization.
- **Known-mechanic combinations:** Cadets + Tiger-Dillo is excluded within a
  training deck. Independent training opponents can expose them on opposite
  sides. This split explicitly does **not** claim no joint game-state exposure.
- **Genuinely new mechanics:** require a new prospective world/catalog binding
  and a rule-aware control; absent from this executable proxy. Separate their
  zero-experience probes from combinations of already exercised mechanics.
- **Promotion:** once a test card/list/combination enters training or opponent
  selection, retire that held-out label in the new manifest. Preserve old test
  results; select a new test prospectively. No test-driven deck selection,
  generator screen or opponent search is allowed to leak the final exam.

Stronger held-out decks can raise or lower score without a generalization
failure. Compare the trained-minus-initialization gain **within each fixed
panel**, not raw trained scores across panels. Then compare those paired gains,
retaining the differing difficulty and opportunity caveat. Deck swaps control
seat/deck allocation, not all difficulty. Self-play's symmetric win rate is
never the learning outcome. Heuristic drift requires a new anchor identity;
freeze its source together with the candidate inference code.

Directed mechanic probes stay a separate evidence axis: ordinary shared actions
in authored positions, scored by required behavior (e.g. choose the Adventure
mode when it preserves a needed creature, or attack a walker when it prevents
its imminent payoff). The rule owner supplies positions and counterexamples;
Task 04 names exact scenario/test IDs before claiming competence. Existing
`manabot.verify` checkpoint-scenario machinery is the owner. The proxy makes no
claim to have these future scenarios. A rules-unit test is not a policy score.

Use `bootstrap_mean_interval` over four-leg deal blocks, pairing identical
blocks across checkpoints; never treat 16 legs as 16 independent draws.
Report per-seed delta and interval plus every raw block. Four deals are very
coarse; two seeds establish a bounded engineering check, not a method-ranking
study. A method claim waits for the deeper track, independent seeds and an
explicit uncertainty analysis. Do not average missing/failed games as losses,
zero exposure or successful draws; publish attempted/expected/valid denominators.

## First pilot for Task 04

The exact JSON configuration is the authority; budget fields are required rather
than filled from Python defaults. Proposed values:

| Bound | Value |
|---|---:|
| Fresh training seeds | 2450401, 2450402 |
| Model | Full world vocabulary, width 64, two attention layers, four heads, FiLM, masked-mean value |
| Learning | Existing damped-policy recipe; 32-step schedule mapped to reference 2048 |
| Collection | 8 streams × 1 complete game/update, both seats; one producer/thread |
| Updates and exports | 32; initialization, 8, 16, endpoint |
| Training games | 256/seed, 512 total if complete |
| Training watchdog | 1800 seconds/seed; 120-second stop reserve, plus finalization |
| GPU allocations | At most 1 hour and $1 per seed, including setup; no automatic retry |
| Evaluation | 5 panels × 4 deals × 4 legs × 5 comparisons × 2 seeds = 800 games |
| Evaluation bounds | One worker; 60 seconds/game, 10,000 Commands/game; 2 hours/$2 total proposal |
| Cumulative proposal | $5: $2 training + $2 evaluation + $1 reserve; reserve is not automatic retry permission |

A fixed complete-game target bounds samples; wall time bounds cost. The wall
caps may prevent completion, especially on slow hardware. Retain incompleteness
rather than changing samples, seeds, max Commands or game deadlines after a
failure. Two hours is not a guarantee that all 800 evaluations finish. Setup,
CUDA numerics, failed attempts, storage/transfer and evaluation all count toward
the authorization Task 04 obtains. Repeated `etude eval` invocations are not a
fresh allowance: the operator must use one cumulative ledger and remaining cap.

**Pre-registered modest-check decision:** proceed to another modest batch only
when both seeds complete 32 updates, actual optimizer changes are nonzero,
finite health checks pass, every primary anchor leg replays, and both seeds'
anchor score at 32 exceeds their own initialization. A positive paired lower
95% bound is stronger evidence, not required for this two-seed engineering
check. Otherwise label learning inconclusive/failed, inspect update and exposure
evidence, and propose a new protocol. No positive result establishes superiority
to another sampling strategy. All five panels must be reported, including
missing cells; held-out results are secondary descriptions, not a hidden gate
chosen after inspection. Unknown coverage stays unknown.

Stop on the first strict training engine/capacity error, rejected numerical
update, nonfinite metric, missing policy export, exhausted wall/memory/cost bound,
or evidence corruption. No random fallback or catch-and-continue training.
Evaluation retains failed/unattempted slots and can finish unrelated cells under
its cap; any primary invalid leg prevents the pilot pass. Known rules bugs are
scoped to affected cards/conclusions; a replayed workaround does not invalidate
unrelated checkpoints, and a bug does not remove native card support.

Record actual environment decisions/s, learner decisions/s, games/s,
collection/update seconds, phase timers, peak RSS, parameters/optimizer changes,
value/policy loss, entropy/support, rejected/skipped updates, and wall/GPU costs.
These come from existing Etude metrics and performance receipts. Report cold
startup separately. Compare native step/search budgets only on the same host
and workload; do not reuse a different runner's wall time as a speed baseline.
No artificial pressure or stress workload is permitted.

Strict failure capture retains deck/config/world/input/checkpoint identities,
seed/RNG and reset derivation, prior Commands, pending decision/Command,
original error class/payload and native capacity diagnostics when available.
Unflushed data after hard termination is missing, not zero. Keep private debug
state in restricted job outputs. Reproduce the retained seed/Command prefix in
a debug build; minimize to a directed engine regression and route it to the
rules owner. Never rewrite a historical world to make a replay pass. Tasks
01/03 may enrich receipts without inventing past fields.

## Two successive uneven batches

The following are concrete **future authoring specifications**, not current
card-support assertions or changes to the proxy JSON. Use new manifest/source
identities; retain each old list verbatim. Card functions/sets and complete lists
are chosen from the integrated native catalog at freeze, never invented here.

| Item | Batch A: Adventure / prepare | Batch B: planeswalkers |
|---|---|---|
| Anchor tickets | Existing two anchor lists, two tickets each (4/8) | Same two lists and tickets (4/8) |
| New tickets | A-tempo: cheap Adventure core; A-value: prepare plus value core, one each (2/8) | B-control: walker protection; B-pressure: efficient attackers/removal for walkers, one each (2/8) |
| Mixed tickets | Old UR-like core + Adventure package; old GW-like core + prepare package, one each (2/8) | Retain A-tempo as one old-family ticket; authored A-value + walker-protection package as one ticket (2/8) |
| Construction | 40 cards/17 lands in this method example; 8 cheap plays, 4 interaction and 4 family/core copies; relevant mana sources explicitly reviewed | Same template; protect expensive walkers with cheap creatures and interaction; reviewed walkers occupy capped top-end slots |
| Stable exam | Original anchor lists, heuristic source and anchor deal family | Keep exact lists/deals; rerun newly admitted controls if world changes |
| Expanding exam | A-tempo vs A-value | B-control vs B-pressure; keep A pair as a separate retention panel |
| Held-out composition | Alternate counts/package of known A cards, absent from all training lists | Keep A test excluded; add B known-card composition |
| Held-out card | One supported Adventure/prepare card never in training or opponent lists; full tokens visible | Keep it excluded or explicitly retire its claim before promotion; add one withheld walker |
| Held-out combination | Two known A mechanics within one deck, absent from training lists | Reserve Adventure + walker package within a deck; name cross-seat exposure separately |
| New-mechanic probe | Resolve the modal Adventure/prepare choice through shared authority | Loyalty activation and walker pressure; report separately from old-mechanic retention |

This changes the *new* and *mixed* distributions, not the historical anchors.
If B's world/catalog/input differs, A's checkpoint is not relabeled for B.
Fresh B training and initialization are the default. Retention means performance
on the old deck exam in B's admitted world; it is not raw A-minus-B win rate.
Within a truly compatible world, a follow-up may use existing
`LearnerContinuation`/`LearningStateImport` with parent raw/EMA/Adam and absolute
schedule clocks, a fresh equal-compute control and a same-boundary continuation
control. Resetting streams/RNG is a new segment, not exact recovery. Weights-only
warm starts and new vocabularies are distinct transfer interventions. No new
continuation route is implemented here.

## Deeper track after full FDN

First deep milestone: freeze the full-FDN integrated catalog and reviewed league.
Propose six training seeds, 128 updates × 16 streams × one game = 12,288 training
games, plus initialization and endpoint on 3 principal panels × 25 deals × 4 legs
× 2 checkpoints × 6 seeds = 3,600 policy-only evaluation games. Freeze a further
1,800 endpoint games for the same policies with existing `agent_puct`, 16
simulations/decision and one CPU thread; separately record policy calls, native
steps and decision latency. Search is not combined with policy-only scores and
is not present in the first pilot. The exact supported search spec and game
deadline must pass ordinary admission before freezing that later protocol.

Proposed deeper ceiling: six one-hour GPU jobs, eight hours CPU evaluation and
$25 cumulative including failures; no launch or promise of completeness. If
calibration cannot fit the fixed design, revise prospectively. Report cross-seed
uncertainty on paired improvements, old-family retention, held-out panel
contrasts and per-family probe scores. Compare a different curriculum only with
matched total training **and selection/search** compute and its own frozen test.
This is a milestone specification for Task 08, not executable permission today.

## Runnable handoff and acceptance boundary

Read-only admission and regime export (no training):

```sh
uv run --extra artifacts --extra evaluation python -m experiments.growing_pool_protocol check
uv run --extra artifacts --extra evaluation python -m experiments.growing_pool_protocol regime > /tmp/growing-pool-regime.json
```

From the reviewed committed revision, `runs(revision=...)` or Etude's normal
job-file reader produces two `Run` declarations with `TrainingJob` payloads.
Task 04 obtains explicit cumulative authorization, freezes source through Etude,
quotes the actual hardware, and launches those declarations through its existing
cohort workflow. Do not invent a launcher here. Build/sync uses the locked Python
3.12/native environment. No local training is an implicit substitute for a failed
GPU placement.

After authorized training, from the **same pinned checkout**, local evaluation
uses the ordinary Etude discovery/callback path and an explicit output directory:

```sh
ETUDE_OUTPUT_DIR="$PWD/.runs/growing-pool-evaluation" uv run --extra artifacts --extra evaluation etude eval experiments/growing_pool.py --source r2://manabot/jobs
uv run --extra artifacts --extra evaluation etude report experiments/growing_pool.py --source r2://manabot/jobs
```

The evaluation command plays games and needs Task 04's authorization and budget;
it is not part of `check`. Its report curves are descriptive. Join the expected
`comparisons()` schedule with obtained match rows before assessing the decision
rule; absent exports and failed callbacks remain missing. A new protocol after
execution must use a new experiment identity/source; never overwrite v1's JSON
or recycle its experiment name to blend populations.

MTG-245 supplies code, manifest, interpretation and the two-batch plan. Task 04's
handoff is complete only after its integrated revision, authored lists and
budget are accepted and its actual run reports update/throughput/exposure/crash
receipts. No design audit or successful discovery test is evidence of learning.
