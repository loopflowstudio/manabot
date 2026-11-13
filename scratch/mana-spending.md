# MTG-347 — higher-level player control of mana spending

Inherited MTG-324 scratch (precision plan, QA page) is removed; it is in git at
`f4e1822fa`.

## Problem

PR #511 (`105e7e2aa`, MTG-297) made "which pooled mana to spend" a native
decision, one unit at a time. Which lands to tap stays automatic. So the
low-level half is exposed and the half a player reasons about ("what do I
leave open") is hidden. The goal is a way for bots and Academy users to
control spending that keeps strategic choices without flooding the action
space with payment steps.

## What people said

- Jack Heart, 2026-11-13 (Task): "then start a task where we run experiments
  and explore a higher level approach to letting users decide how to spend
  mana"; earlier: "choosing which lands to leave untapped also is a difference".
- Jack Heart, 2026-11-13 (comment): "lets abandon 346 and start 347 then. but
  also note i think there is an existing task for this" ... "it mentions
  combompund decisions". MTG-346 (revert #511) is abandoned; #511's decisions
  stay on main. The Task body's "now being reverted to automatic" is stale.
- Same comment: first deliverable is "spending decisions per game and as a
  share of all decisions in self-play on the Elemental recipe and TLA Limited
  decks, before any design work. Exploratory, well under USD3."
- Coordination set there: MTG-265 owns the frozen-protocol learning study of
  compound vs sequential decisions; "hand learning-effect claims to MTG-265's
  protocol rather than running a competing study". MTG-342 (autopay over-taps,
  e.g. Mountain, Mountain, Forest for {1}{G}, floating {R}): "Measure with and
  without that bug's effect". Both are unstarted (lf, 2026-11-13).
- Task deliverable: `experiments/<name>.md` with measured frequencies, "at
  least one small learning comparison", and a recommended design for review.
  Implementing the design is a separate Task. USD25 launcher cap; more needs
  approval.

## Current architecture (read 2026-11-13 at `f4e1822fa`)

- Land tapping is automatic. `Game::produce_mana` (`managym/src/flow/mana.rs`)
  asks `PaymentModel::plan(.., SourcePolicy::FixedOnly)` for a witness and taps
  every `PaymentStep::Fixed` source in it. Callers: `flow/play.rs:1931`,
  `flow/cast_permission.rs:181`, `flow/decision.rs:2057`. `witness` is a
  depth-first search in source order that stops at the first payable pool, not
  a minimal one; that is MTG-342's over-tap. No decision offers which fixed
  sources to tap.
- Spending is `Decision::ManaSpend { player, cost, spent, completion }`
  (`flow/decision.rs`), decision code 31, stage `"mana_spend"`
  (`decision_context.rs`), published as `ActionSpaceKind::Modal` with one
  `Action::ChooseMode` per `ManaSpendChoice`. `Mana::spending_choices`
  (`state/src/mana.rs`) offers every pooled color/class that pays the next pip
  (colored, then hybrid, then generic). `Game::mana_spend_choices` filters to
  choices that keep all saved obligations payable and offers one when the
  usable pool equals the cost; `advance_mana_spend` auto-advances single
  choices. A decision therefore appears only with surplus usable mana and at
  least two distinct units for a pip, and repeats per pip.
- The spell's receipt (`ManaSpent`) feeds Toph, converge and Animal Attendant;
  pooled classes carry restriction, spend effect and combat expiry.
- #511 changed no policy-input shape (`observation_layout.rs` edit is a
  description string); pooled classes are fact rows (`FACT_EFFECT_PROGRAM`,
  id `PooledMana`).
- Counting hook that exists today: `managym.Env.decision_context_json(viewer)`
  returns `stage`; `manabot/semantic/text_observation.py::_STAGES` lists them.
  The opponent policies in-engine are `none`/`random`; `manabot.sim.players`
  builds Torch-free `heuristic` and `random` players.

## Prior art

- Arena: official mobile FAQ documents auto-tap/full-control settings; official
  Brothers’ War note describes planned-mana preview for color-sensitive spells.
  Primary source links are now in the report (verified 2026-11-13).
- Forge: a "Pay Mana Cost" prompt where the player clicks sources, plus an
  Auto button (1.5.7 changelog). Its developer's stated reason: how you tap
  and empty the pool "is rarely of consequence strategically". Users report
  Auto tapping a mana creature they wanted to attack with.
- phase-rs (static reading, v0.123.0, rechecked 2026-11-13): the flat action
  list omits mana actions, while its native per-object grouping retains them.
- All three: automatic by default, manual source choice as the override, no
  per-unit pool prompt in ordinary play. None offers "payment outcome" as one
  choice.

## Chosen approach

Measure first, then design from the numbers. One experiment,
`experiments/mana_spending.py` with report `experiments/mana_spending.md`,
exploratory, in three parts.

1. **Frequency (first deliverable).** Mirror games on the ten Elemental decks
   (`experiments.elemental_grid.DECKS`) and on TLA Limited decks
   (`experiments/tla_pools.py` corpus), driven through `managym.Env` with
   `skip_trivial` on. Per game: decisions by stage, `mana_spend` count and
   share, offers per spend decision, spend decisions per cast, and games with
   none. Policies: `heuristic` mirror and `random` mirror on every deck; a
   checkpoint mirror where an existing checkpoint loads on the current world
   (MTG-183 final for Elemental). Add one deck with Heartbeat of Spring as the
   upper end. Local CPU, USD0.
2. **Over-tap attribution (MTG-342).** Classify each spend decision at the
   payment that produced it: over-tap when some source tapped by that
   `produce_mana` call can be removed and the cost is still payable from the
   pool, while saved obligations remain feasible. Report both the attributed
   and non-over-tapped subsets against the original decision denominator,
   without calling the latter a post-fix rate. Mechanism: a native counter
   read through `Env.info()`, computed inside `produce_mana` where the witness
   and sources are known; it changes no behavior.
3. **Shape comparison on the same games.** For each cast in part 1's games,
   compute what each candidate shape would have offered, without implementing
   the shapes:
   - A. today: per-unit `ManaSpend` after automatic tapping;
   - B. automatic everything (first feasible unit), zero decisions;
   - C. automatic with override: one extra offer per cast only when outcomes
     differ;
   - D. payment outcome as one decision: one offer per distinct outcome, where
     an outcome is (what stays available: untapped sources by what they can
     produce, plus floating pool) × (receipt facts a rule reads: colors spent,
     spend effects applied). Equivalent payments collapse;
   - E. source and spend folded into the cast offer (cast X leaving Y).
   For each: decisions per game, offers per decision (median, p95, max), and
   how often the choice is real (≥2 distinct outcomes). D and E need distinct
   outcomes enumerated from the payment model; bound the enumeration and count
   the casts that hit the bound.

Then the **small learning comparison** (Elemental, as recorded below): Elemental recipe,
arm A (main) against arm B, 2 seeds each, matched updates, scored against the
heuristic, under `--max-usd 25`. It answers only "does removing unit spending
decisions visibly change early learning or steps per game on this recipe"; it
is exploratory and the report says so. Claims about compound vs sequential
learning go to MTG-265. Arm B needs a native switch that takes the first
feasible unit in `advance_mana_spend`; it is experiment scaffolding on this
branch and the report names whether the recommended design keeps or drops it.

The report ends with one recommended design for review, covering both players
of the shared contract: policy input/action space and Academy.

### Leading hypothesis, as tested

D with B's automation for single-outcome casts. Part 3 answered its risk: flat
D exceeds the 64 policy slots (422 casts certified above 64), so the report
recommends D grouped natively.

## Alternatives considered

- Revert #511 to automatic (MTG-346): abandoned by Jack's comment.
- Fix MTG-342 first and stop there: makes spend decisions rarer but leaves
  land choice hidden and Heartbeat-style surplus per-unit. Part 2 says how far
  it goes.
- Manual tapping as priority actions (Arena/Forge override): adds a mana
  action for every untapped source at every priority; the largest action-space
  cost of any shape, and pooled mana empties between steps. It remains a
  qualitative alternative; the bounded A–E slice does not count priority taps.
- A full learning study of the shapes here: belongs to MTG-265.

## What the frequency run showed (2026-11-13, `58e5923c53`)

Report `experiments/mana_spending.md` holds the tables; these facts change
the remaining work.

- Spend prompts are rare on ordinary decks: 0.11–0.60% of decisions, 0.5–1.3
  per game, 81–89% of games have none. Heartbeat: 9.7–13.8%, every game.
- Over-tap dominates them: 51–95% of ordinary-deck spend decisions follow a
  payment with a removable source; 8–15% of all payments over-tap (e.g.
  Elemental heuristic 400 of 2717). Strong evidence for MTG-342 by itself.
- Offers per spend decision are small: median 2–3, max 5. The 64-slot risk
  does not arise in shape A; it remains open for D/E, which add land choice.
- So the per-unit prompt is not the cost; the hidden choice of which sources
  to tap is the unmeasured part. Part 3 must report, per cast, how often
  sources offer ≥2 distinct "what stays available" outcomes, separately from
  pool-receipt outcomes.
- MTG-183 final (`4f2ad656…`, input v1) is refused with
  `UnsupportedPolicyInput`; no checkpoint on this world exists to count with.
  Frequency rests on heuristic and random mirrors; receipt retained.

## Learning comparison decision

Keep Elemental as named in the Task: seeds 34701/34702 per arm, 32 matched
updates. Its ~0.2% pool-prompt share makes this a narrow exploratory control;
it does not test the recommended grouped source chooser or support equivalence.
The source is `00ee188998`, with the same regime apart from arm label and the
job’s explicit automatic-spending environment switch. Score final policies on
the existing 50-pair Elemental heuristic panel under their own spending modes.

## Risks

- Share depends on the policy; report each policy separately, never pooled.
  Without an admissible checkpoint there is no trained-policy row.
- Ten TLA builds in file order and eight Heartbeat games per policy are
  bounded convenience samples; say so wherever their rates appear.
- Enumerating outcomes can blow up; cap it and report hits rather than
  truncate silently.
- Engine Budget CI: the counter in `produce_mana` runs on every payment; it
  probes removability only for surplus payments. Budget CI has not run on it.

## Scope

In: the experiment file and report, the native over-tap counter, the arm-B
switch, published data under `experiments/data/mana_spending/`, the
recommended design.
Out: implementing the design; fixing MTG-342; any MTG-265 protocol; changing
policy-input layout or frozen evidence; Academy UI changes.

Forbidden: a second payment implementation beside `PaymentModel`; counting by
re-deriving rules in Python; a learning claim stated as a finding.

## Status

Implementation and experiment work are complete; gate, review and PR delivery
remain (no PR exists yet, so Engine Budget CI has not run on the native
counters). The shape comparison (separate source and receipt outcomes, cap
hits) and the learning pilot with its cost, which the last iteration direction
asked for, are both done; tables, receipts, costs and limits are in the report.

- Frequency (`58e5923c53`, `9f2f1b69b8`, `b63d486f81`): native counters in
  `Env.info()` computed in `produce_mana`, the decision count in
  `advance_mana_spend`'s offer arm; behavior unchanged. Over-tap exclusion is
  attribution against the original denominator, not a post-MTG-342 rate.
- Shapes: opt-in `PaymentModel` enumeration (`flow/mana/outcomes.rs`). The
  merged engine changed final digests, so `current/` is a second baseline on
  the same decks and deals; all 336 action sequences, stages, offers and
  counters match the original, whose bytes are preserved. 14,775 payments,
  9,089 cast payments, 420 cast caps; D reaches 712 exact outcomes, E 1,552
  priority offers on its completely measured subset.
- Pilot: four runs at `00ee188998cb`, 32 updates each. Evaluations
  `mana-spending-A-303f5f0a6a59` and `mana-spending-B-fe362b5c294d`: 800 valid
  replay-checked games. Seed 34701: A 26.0%, B 27.0%; seed 34702: A 28.5%,
  B 32.5%. Neither an improvement nor equivalence is established, and D is
  untested. Both evaluation logs show missing-focus ChooseMode warnings; three
  recovered training resource-limit failures discarded 915 samples. No encoder
  change was made mid-pilot.
- Recommendation: grouped native D with exact source choices and a bound
  automatic plan, shared by policy and Academy. Defaults stay unchanged.
- All 24 pilot artifacts and five baseline/shape artifacts are published with
  manifest entries.

## Done when

`experiments/mana_spending.md` states, for Elemental and TLA Limited decks,
spending decisions per game and their share of all decisions, with and without
over-tap (done); compares the shapes on decisions and offers per decision;
shows the small learning comparison with its cost; and recommends one design
with what it changes for the policy and for Academy. Data is published and the
manifest committed.

## Checks run

- `cargo test --manifest-path managym/Cargo.toml --lib over_tap` (debug):
  passed; the second Heartbeat position was corrected to {2}{G} after the run
  showed one doubled Forest pays {1}{G} exactly.
- `count --games 8 --seed 0`: exit 0, 336 games, 336 verified replays.
- Compress (2026-11-13) refactored `experiments/mana_spending.py` only (one
  owner for arm mode, provenance and the evidence directory; shape rows written
  through their read types). `summary` on a copy of `current/shapes/shapes.jsonl`
  wrote a byte-identical `summary.json`; `shapes` on the first six games wrote
  byte-identical rows. Native code changed by one doc comment; no rebuild.
- `tests/eval/test_experiment_files.py`: the discovery test passes.
  `test_a_tracked_self_play_run_draws_every_named_metric` failed once in the
  four-test run and passed alone in 121 s; it does not read this experiment.
  Gate should rerun it.
- `uv run --no-sync --extra dev pytest tests/infra/test_experiment_data_untracked.py -q`:
  1 passed after publishing all 24 pilot artifacts.

## Delete — do not maintain

The report’s planned-only shapes instructions are replaced by runnable replay
measurement and results. Preserve original frequency evidence, the native
payment authority, existing source/action behavior, and frozen A/B revision.
The experiment switch is a reproduction control; the recommended production
chooser replaces forced automation with an explicit bound native plan.

## Launch facts (2026-11-13)

Launch `518a4948ea80`, runs `f2f5adec2478`, `cfff739d2db5`, `5e0ed566acd9`,
`2f6bd7e98142`: no job command started. The second L4 was unavailable.
`mtg347-l4-71b326e7fb82` was confirmed deleted; `mtg347-l4-620f4314c806` was
not acquired. Final launcher cost observations are retained with the pilot rather than
reconstructing billing from elapsed time.
Serial quote: USD4.31 first-wave rental, USD12 job/cohort rental budget.

## Concept review (2026-11-13, at `9dbe3f78df`) — next implement direction

No code changed here. Status above ("complete; gate, review and PR remain") is
superseded by this section until the direction below is done.

### Biggest mismatch: a redundant tap is a defect in part 2 and a choice in part 3

Part 2 calls a payment that taps a removable source an over-tap, MTG-342's
bug. Part 3's enumeration (`fixed_outcomes`) counts every such payment as a
distinct outcome ("includes deliberately producing surplus mana"): three
Forests paying {G} are 3 outcomes, 1 without surplus. The case against flat D
(712 outcomes, 422 casts certified above 64, the 420 cap hits) and E's 1,552
all rest on that counting definition. The shape rows hold no count that
excludes redundant taps, so the report cannot say whether one flat choice of
"what stays untapped" fits 64 slots. The recommendation's grouping machinery
follows from an unmeasured premise.

Consequence if surplus is excluded and every tapped unit is spent: the pool
remainder is unchanged and the receipt is fixed by the sources tapped, so the
outcome is the set of sources left untapped, the choice the Task quotes
("choosing which lands to leave untapped"). Pool and receipt choices remain
only where a source makes more than the cost needs (Heartbeat) or mana was
already floating, which is where shape A's prompt appears today.

Direction:

1. Add one count per payment to the existing capture: outcomes over
   non-redundant source sets (no tapped source removable with the cost still
   payable and saved obligations feasible; the same test as
   `record_mana_payment`), by capability and by identity, with the same cap
   and cap-hit rule. Same `PaymentModel` authority; no second payment
   implementation. Rerun `shapes`/`summary` on the `current/` baseline (local
   CPU, USD0), publish, and keep the surplus-inclusive rows as they are.
2. In the report, give D and E both ways: offers median/p95/max, casts above
   64 and cap hits, with and without surplus. State the recommendation from
   those numbers. If non-redundant flat D fits 64 everywhere measured, the
   grouped chooser is not implied by the evidence and the report says so.

### Names that hide what happens

- The recommended design is shape C's entry (a bound automatic plan against
  everything else) in front of shape D's outcomes. The report calls it
  "grouped D" and quotes flat D's decision row (+16–24 decisions/game) while
  noting grouping "can add decisions". Its cost row is C's range (e.g.
  Elemental heuristic 324.69–341.57 against A's 306.18). Name it by what the
  player does and quote that row.
- "Identity collapse loses distinctions" (60–70%) counts two untouched basic
  Forests as different outcomes, alongside the case that matters (a mana
  creature against a land). The report then tells Academy to "use visible
  object joins to distinguish equal-output sources", which offers subsets of
  identical lands. Split the count: sources differing only in object identity
  (same definition and state) against sources that differ in anything else.
  If the capture cannot tell them apart cheaply, say that the 60–70% is an
  upper bound and why.
- The automatic plan in the recommendation is today's first-feasible witness,
  which over-taps in 8–15% of payments. Say that the bound automatic plan
  presupposes MTG-342's fix, or that it would show a plan with a stray float.

### Open product choices (for the report's review questions; none settled)

The Task's deliverable is a recommendation for review, so these go to the end
of the report as numbered questions with the measurements each depends on.
Nobody has answered them; the report must not state an answer as decided.

1. Does the payment chooser offer deliberate surplus tapping (tap a land the
   cost does not need, float the mana)? The engine offers no such action
   today. It is legal in the rules and occasionally matters; including it is
   what pushes flat D past 64. Matters because it decides between one flat
   choice and a grouped chooser.
2. Which sources are the same choice? Exact object always (the report's
   current stance), or equal definition and state. Matters because exact
   identity multiplies offers for identical lands on both the policy and
   Academy, while capability alone merges a mana creature with a land.
3. Is a payment decision at 63–73% of casts (about +5–8% decisions per game,
   mostly answered "automatic") the accepted price of exposing source choice,
   against today's 0.11–0.60% prompts? The alternative noted in
   `scratch/questions.md` (binding override intent at cast selection) is
   uncounted. Learning effects of either stay with MTG-265.

### What must survive

Original and `current/` evidence bytes, the frozen A/B pilot revision and its
switch as a reproduction control, `PaymentModel` as the one payment authority,
no change to defaults, policy input or Academy.
