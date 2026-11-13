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

### Leading hypothesis (to test, not decided)

D, with B's automation for casts with one outcome. It is the only shape that
exposes the land choice Jack named, removes per-pip steps, and stays one
ordinary native decision that Academy can label ("leave Forest untapped",
"spend {R}{G}: converge 2"). Its risk is offer count on five-color boards with
many distinct sources; part 3 measures that. If p95 offers per decision
exceed the 64 policy slots anywhere, D needs grouping or fails.

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

Done (`58e5923c53`, `9f2f1b69b8`, `b63d486f81`):

- Native counter: `Env.info()` keys `mana_payments`,
  `mana_payments_over_tapped`, `mana_spend_decisions_over_tapped`, computed in
  `produce_mana` (`managym/src/flow/mana.rs`); the decision count sits in
  `advance_mana_spend`'s offer arm. Behavior unchanged; replays verify.
- `experiments/mana_spending.py count` (`--games` per deck, `--seed`, `--out`);
  deck sets `elemental`, `tla`, `heartbeat`. Report sections Run it, What is
  counted, Results, Limits; evidence published (`games.jsonl`, `decks.json`,
  `checkpoint-admission.json`) and the manifest committed.
- Over-tap exclusion is descriptive attribution against the original
  denominator, not a post-MTG-342 rate; a post-fix rate needs a matched run
  after MTG-342 lands. The report says so.

Completed in this slice:

- Opt-in `PaymentModel` fixed-source/receipt enumeration and per-payment actual
  spend offers; source-capability and exact-identity counts; work caps.
- Current baseline on the original exact decks/deals: all 336 action sequences,
  stages, offers and counters match; rebuilt merged engine uses new digests.
  Preserve the original bytes and use current digests for shape replay checks.
- 336 verified shape replays; 14,775 payments / 9,089 cast payments; 420 cast
  caps. D reaches 712 exact capability outcomes, 422 casts certify >64. E
  reaches 1,552 priority offers on its completely measured subset. Tables and
  limits are in the report; five current-baseline/shape artifacts are published.
- Report recommends grouped native D after announcements, retaining exact source
  identities and complete choices shared by policy and Academy. No UI changes.
- Arm B and Etude jobs are implemented; debug tests and local release build pass.

Implementation and experiment work are complete:

- Four runs at `00ee188998cb` succeeded at 32 updates. Evaluation runs
  `mana-spending-A-303f5f0a6a59` and `mana-spending-B-fe362b5c294d` completed
  100 matches each; all 800 games valid and replay-checked, no replacements.
- Seed 34701: A 26.0%, B 27.0%; seed 34702: A 28.5%, B 32.5%. These two early
  seeds cannot establish an improvement or equivalence, and do not test D.
- Both evaluation logs show missing-focus ChooseMode warnings. Three recovered
  training resource-limit failures discarded 915 samples. Full receipts and
  limitations are retained in the report; no encoder change was made mid-pilot.
- The report recommends grouped native D with exact source choices and a bound
  automatic plan, shared by policy and Academy. Defaults stay unchanged.
- Final launcher costs and ended-rental confirmations are recorded in the report.
  All 24 pilot evidence artifacts are published and manifest entries retained.
  Gate/review/PR delivery remain the caller's later steps.

## Done when

`experiments/mana_spending.md` states, for Elemental and TLA Limited decks,
spending decisions per game and their share of all decisions, with and without
over-tap (done); compares the shapes on decisions and offers per decision;
shows the small learning comparison with its cost; and recommends one design
with what it changes for the policy and for Academy. Data is published and the
manifest committed.

## Checks run

- `cargo test --manifest-path managym/Cargo.toml --lib over_tap` (debug):
  passed; four positions (Mountain, Mountain, Forest for {1}{G} over-taps;
  Forest, Mountain does not; two Heartbeat positions, the second corrected to
  {2}{G} after the run showed one doubled Forest pays {1}{G} exactly).
- `count --games 8 --seed 0`: exit 0, 336 games, 336 verified replays. After
  the compress move, a rerun wrote a byte-identical `games.jsonl` (`cmp`).
- `tests/eval/test_experiment_files.py::test_etude_finds_each_current_experiment_and_nothing_else`
  and `tests/infra/test_experiment_data_untracked.py`: passed. Ruff, format,
  `git diff --check` passed. Full affected suites belong to gate.

Final publication check: `uv run --no-sync --extra dev pytest tests/infra/test_experiment_data_untracked.py -q` — 1 passed in 2.73s after publishing all 24 pilot artifacts.

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
