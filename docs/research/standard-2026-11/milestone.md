# Standard milestone — 2026-11-04

Status: proposed sequence plus separate research Task [MTG-245](loopflow://manabot/task/MTG-245), authorized to launch. Tasks 04 and 08 remain actual training/evaluation runs; no training experiment has been launched.
Accepted direction: full post-rotation Standard; faithful, trusted rules, performance and composable code first; coherent changes; large shared changes early; complete FDN first, then generally newer sets. Bugs do not automatically revoke playability or invalidate every run. Preserve evidence so conclusions can be scoped.

The [HTML walkthrough](milestone.html) is the detailed plan and API explanation. API snippets are proposals, not implemented signatures. Existing excerpts are pinned to `5dd11cec4100aa0e1f4a23c063dfd5778e2ba28f`.

Four already launched Tasks (MTG-241 execution/replacements, MTG-242 mana, MTG-243 naming, MTG-244 layers) are dependencies, excluded from the eight below. No new executor, card DSL, parallel characteristic authority or per-card payment bypass is proposed.

## Milestone 1: shared casting and history

01. **Costs, contributions & casting receipts.** One casting/payment transaction owns announced choices, composite costs, nonmana contributions and actual payment facts. Reuse MTG-242’s resource solver; deliver substantial FDN cost batches plus cross-set cost/provenance cases. FDN cost cohort; cross-set bargain, forage, crew/saddle and paid-mana witnesses. Keep station’s distinct rules explicit. Acceptance: Offer, payment and execution agree; a nested mana choice resumes without double-paying. Preserve sacrificed/tapped object provenance and actual mana spent.

02. **Adventure, Omen, prepare & casting permissions.** Explore and implement a shared casting model across Adventure, Omen, modal faces and prepare, including linked castable spell copies. Deliver whole mechanism cohorts alongside FDN permissions, using existing grants, exact identity and the shared casting path. FDN permission cohort, HOB’s 17 Adventures, and broad SOS/FRA prepare and TDM Omen cohorts, assigned after dependency reconciliation. Acceptance: Preserve profile characteristics, exact exile identity and type quotas. Prepared-copy permission follows current control; ability loss does not unprepare, and copying the permanent does not copy preparation. Distinguish resolving from countered Adventures.

03. **History & contextual queries.** One typed query vocabulary over committed events, scoped counters, payment receipts and instruction results; implement broad FDN history/query batches and cross-set consumers without per-keyword history stores. FDN history and contextual-query cohorts; cross-set expend, void, disappear, life-gained and actual-spent-mana cases. Acceptance: Requested life loss/destruction differs from actual results. First/nth events reset at the right scope; forks and observations preserve decision-relevant history.

04. **Validation training A.** Execute the first bounded training/evaluation campaign using MTG-245’s protocol on the integrated engine snapshot. Report measured learning, performance and crash evidence; a design or configuration alone does not complete this Task. Learning, collection/search cost, crashes and capacity failures. Acceptance: Demonstrate learning on declared canaries; compare fixed-opponent evaluation and throughput; retain exact failing states and scoped conclusions. No frontier claim.

## Milestone 2: FDN object and game procedures

05. **Battlefield forms, copies & persistent abilities.** Unify exact object links, battlefield copiable snapshots and persistent ability sources. Explore transform, meld and face-down identity before fixing the representation; deliver the FDN lifetime/copy cohort and a coherent broader form/copy cohort. FDN lifetime/copy cohort; FIN transform/meld and face-down boundary cases. Task 02 owns linked castable spell copies, not this Task. Acceptance: A blinked card is a new object; a copied payload keeps its required exceptions; linked effects survive only for their stated lifetime. Nested results remain MTG-241’s responsibility.

06. **Planeswalkers, loyalty, grants & emblems.** Implement planeswalkers as a broad capability across FDN and FRA: loyalty costs, timing/use limits, scoped granted abilities, tokens and persistent emblem sources. Compare Classes and Rooms to expose which activation/state pieces truly compose. FDN planeswalkers plus FRA’s 40-row loyalty/token/emblem cohort; include timing overrides, empowerment and zero-loyalty exceptions. Acceptance: Doubling Season changes entry counters, not loyalty costs. Timing/usage constraints hold; an emblem remains effective and observable after its planeswalker leaves.

07. **Game procedure & terminal outcomes.** One procedure owner for opening-hand decisions, turn termination/cleanup and win/loss arbitration. Explore player-control changes and special-action boundaries against this model; deliver FDN procedure cards and a coherent cross-set procedure cohort. FDN procedure cohort; investigate Secret of Bloodbending, Dominion Bracelet and Cosmic Cube as shared player-control cases. Acceptance: Opening-hand entry follows mulligans; ending a turn follows the correct cleanup path; poison and loss-prevention rules use the same outcome authority. Between-game sideboarding stays a separate scope decision.

08. **Validation training B.** Repeat the bounded operational campaign with the expanded FDN mechanisms and mixed TLA/FDN decks. Regression against A, old anchors and new capacity pressure in ordinary play. Acceptance: Distinguish rules-driven trajectory changes from speed changes. Compare learning and collection costs; reproduce crash receipts in debug and retain targeted regression fixtures.

## Broader delivery and ambitious exploration — 2026-11-04

Jack Heart requested roughly 1.5–2× more cards per implementation Task and more
ambitious exploration of Adventure/prepare, planeswalkers and other mechanisms
that substantially affect the rules engine. The prior examples were witnesses,
not measured card budgets; do not claim a numeric doubling without assigning
actual ledger rows. The revised Tasks own substantial mechanism cohorts, their
tokens and cross-mechanic acceptance cases. Integrate newly expressible ordinary
cards in the same cohort rather than stopping at two witness definitions.

Exploration is the opening part of each implementation Task, not extra planning
Tasks. Compare concrete candidate representations against contrasting real
cards; implement decisive probes before choosing an API; then complete the
cohort and delete replaced paths. Full FDN remains the first set closure, while
mechanism-driven cross-set batches are deliberately larger and earlier.

Candidate cohorts are scheduling inputs, not promised supported-card counts:
FDN has 100 cost-primary rows, 13 permission rows, 47 history + 34 query rows,
23 lifetime + 3 stack-copy rows, and 11 procedure rows. HOB has 17 Adventures;
SOS/FRA have 36/21 prepare-layout cards; FRA has a 40-row loyalty/token/emblem
cohort. These groups have dependencies and do not imply automatic implementation
of every row by one primitive. Assign each definition once, follow dependencies,
and include adjacent composition cards where their semantics already work.

Task 02 must contrast Adventure/Omen/modal profiles with prepare's distinct
linked copy and control-following permission. It owns castable spell-copy links;
Task 05 owns battlefield forms/copiable characteristics and ability lifetimes.
Do not leave preparation as a boolean or defer its identity requirement to an
unrelated future Task. Task 05 compares one-card transform, two-card meld and
hidden face-down characteristics; preserve exact physical-card identity and
viewer-safe observations. Task 06 explores ordinary and granted loyalty,
variable costs, timing versus use quotas, command-zone emblems, empowerment and
zero-loyalty exceptions. Compare Classes/Rooms without pretending a Class level,
unlocked Room and loyalty counter are the same state or action. Task 07 explores
player control separately from ownership and ordinary priority decisions.

Dependencies are completion contracts, not a serial launch order: 01 needs
241/242; 03 needs 241 and 01 receipts; 02 needs 01/03 and 243's naming integration
for prepared spells. 05 needs 241/244 and coordinates copy identity with 02.
06 needs 01/03/05 and 244. 07 needs 241/03. Validation A needs the integrated four
foundations and 01–03; validation B needs A evidence and integrated 05–07.
05 can start before A; training uses pinned snapshots while development proceeds.

Training Tasks expand their mechanic/deck coverage to match delivered cohorts.
Do not mechanically double run budgets: ordinary validation stays bounded;
deeper frontier/generalization work remains a separate, sparse track.

## Demonstration and integration contract

A real card asks a nested choice; fork at that decision, choose different legal actions, resume the correct child and finalize once. Cast through a non-hand permission, pay actual composite costs, and observe correct history and later resolution. Exercise Commands, replay, native/Python observations, learning tokens, LLM text and Academy where affected. One-choice decisions keep normal native skipping.

Use native CardDefinition/Effect as the authored representation. Extend existing CastChoices/Payment and cast-grant owners; migrate their consumers and remove overlapping handlers/flags. Reuse MTG-242 resource reasoning and MTG-241 execution/results. Task 03 extends existing event/counter/query authorities; Task 05 consolidates exact links, copiable snapshots and persistent grants without creating a generic duplicate store. Layers own characteristic computation and cache dependencies. Reassess concrete deletion lists against the four foundation PRs before implementation.

Gate follows docs/development.md and AGENTS.md: debug Rust suite (not release alone), formatting/clippy, extension rebuild, focused cross-consumer observations/replay/choice checks and current-world arena checks when applicable. Existing same-runner Engine Budget covers step/search equivalence and changed digests. Each implementation design names the actual new directed test commands; this planning artifact does not invent test names or passes.

## Separate training-methodology Task: MTG-245

[MTG-245](loopflow://manabot/task/MTG-245) investigates training/evaluation as
irregular card cohorts arrive. Jack Heart explicitly retained Task 04 as an
actual training run; MTG-245 supplements it. The near-term plan therefore has
nine new Tasks: six engine/card Tasks, one methodology Task, and two actual
training/evaluation Tasks. Its [brief](tasks/growing-pool-training.md)
requires deck/sampling alternatives, stable anchors, held-out card/combination/
mechanic splits, attribution controls, a two-batch example and an executable
pilot protocol. Research begins independently of engine completion. Tasks 04
and 08 consume it and must produce actual run evidence. This research launch
does not itself launch training or provision compute.

## Training and evidence

Tasks 04/08 are bounded operational campaigns, not frontier runs: fixed configurations/seeds, declared game/update budgets, learnability canaries, anchor opponents and mixed-mechanic decks. Compare learning and end-to-end collection/update cost separately from native engine speed. Capture crash seed/RNG, Commands, world/input/checkpoint identities, exact pending state, original error and capacity counts; reproduce in debug and retain directed regressions. No stress load or silent truncation.

The first deeper assessment follows full FDN; repeat sparsely after larger pool/mechanic expansions. Compare strength under matched compute and experience, multiple seeds/uncertainty, held-out cards/decks/combinations and cross-pool transfer. Distinguish definitions being visible from held-out training experience. Keep frozen worlds/checkpoints/evidence intact; use compatible runs or paired fresh training. A defect scopes affected conclusions rather than invalidating everything. Set concrete budgets/hardware/criteria at experiment design, before execution.

## Rough arc

Complete coherent remaining FDN batches, tokens, membership/admission and integration; then operational validation and deeper generalization. Prefer FRA, HOB, MSH, SOS, TMT, ECL, TLA closure, SPM, EOE, FIN, TDM, DFT, moving shared prerequisites earlier without finishing their whole sets. Refresh for Star Trek, Nauctis and subsequent releases: 13 currently released survivor sets are not the whole future format. Keep tournament sideboarding separate from card-file support; FDN opening-hand mechanics require mulligans.

Fewer than 50 total Tasks: conservatively count four foundations plus nine new Tasks, leaving at most 36 for later authoring, both training tracks, repairs and new releases. This is a ceiling to plan against, not a guarantee. Replan after each milestone; do not hide unrelated work in a catch-all or sacrifice quality to satisfy the count. Full FDN is the first completed set, not a claim that all 517 researched identities fit into the near-term Tasks.

Validation: documentation only; no engine tests, benchmarks or training executed for this plan. HTML verified in headless Chromium at 1280px and 390px, with desktop/mobile code inspection, eight Task rows, valid local/anchor links, working disclosures, no page errors or document overflow, and successful print rendering.

## Publication handoff

Synced to main `8e8d702f8` before publication. Research findings remain pinned
to their inspected source; main already implements Secret of Bloodbending/player
control (`4463e57f5`). Reuse that authority for remaining player-control cards.
MTG-245 is running; Jack Heart selected authored functional decks, frozen
sampling mixtures and stable evaluation anchors as its initial approach in
that Task's conversation. Exact decks, mixtures and budgets remain proposed.
Tasks 04 and 08 still execute actual training and report measured results.
The engine principles guide is published with this research; MTG-241 can reuse
it rather than independently publish the older embedded draft.
