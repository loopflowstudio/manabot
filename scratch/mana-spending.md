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

- Arena: auto-tapper picks lands and pool spending; players override by
  tapping lands by hand or clicking pool mana, with Full Control. Reports say
  floating colorless is still spent automatically and restricted mana is hard
  to direct. (Player guides and feedback.wizards.com threads; no official
  description found.)
- Forge: a "Pay Mana Cost" prompt where the player clicks sources, plus an
  Auto button (1.5.7 changelog). Its developer's stated reason: how you tap
  and empty the pool "is rarely of consequence strategically". Users report
  Auto tapping a mana creature they wanted to attack with.
- phase-rs (static reading, v0.123.0, 2026-11-11): its legal-action
  enumeration for AI omits mana actions.
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

Then the **small learning comparison**, sized by part 1: Elemental recipe,
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
  cost of any shape, and pooled mana empties between steps. Counted in part 3
  as an upper bound only.
- A full learning study of the shapes here: belongs to MTG-265.

## Risks and findings

- Share depends on the policy. A random policy casts and holds mana
  differently from a trained one; report each policy separately and never
  pool them.
- A pre-#511 checkpoint has never seen stage 31. Shapes are unchanged, but the
  world pool grew (new cards), so the world binding may refuse it. If it does,
  the report states that and frequency rests on heuristic and random mirrors.
- TLA Limited "decks": use built 40-card decks from the corpus; if the corpus
  holds pools only, sample a fixed list of decks and record the list.
- Enumerating outcomes can blow up; cap it and report hits rather than
  truncate silently.
- Engine Budget CI: a counter in `produce_mana` runs on every cast. Keep it to
  an integer increment plus a check only when surplus exists.
- No worktree `.venv` or built extension existed at kickoff; the first run
  needs `uv venv --python 3.13` and the maturin build from the repo guide.

## Wild success / failure

Success: the report shows a number like "spend decisions are N% of decisions,
M% of them over-tap", a table where one shape has fewer decisions and a real
choice nearly every time it appears, and Academy mock labels a player reads at
a glance. Failure: a design chosen before counting, or one that hides the land
choice again behind automation and is ripped out when a card cares.

## Scope

In: the experiment file and report, the native over-tap counter, the arm-B
switch, published data under `experiments/data/mana_spending/`, the
recommended design.
Out: implementing the design; fixing MTG-342; any MTG-265 protocol; changing
policy-input layout or frozen evidence; Academy UI changes.

Forbidden: a second payment implementation beside `PaymentModel`; counting by
re-deriving rules in Python; a learning claim stated as a finding.

## Slices

### Delete — do not maintain

- Inherited MTG-324 `scratch/plan.md` and `scratch/qa.html` (already removed;
  history at `f4e1822fa`). Preserve its shipped precision behavior and evidence.
- No payment implementation is replaced in this diagnostic slice. Keep
  `PaymentModel`, native spending choices, receipts and observations unchanged.

### Measurement correction (implementation, 2026-11-13)

Excluding decisions attached to over-tapped payments is descriptive attribution,
not an exact post-MTG-342 rate. Some such payments retain strategic choices even
without the redundant tap, and different taps change later trajectories. Report
the non-over-tapped subset against the original denominator; a post-fix rate
needs a separate matched replay/run after MTG-342. The draft report now states
this distinction.

1. **This slice.** Parts 1 and 2: counter, `experiments/mana_spending.py`
   frequency run, first report section.
   Check: `uv run python -m experiments.mana_spending count --games 8 --seed 0`
   prints one row per (deck set, policy) with decisions/game, spend
   decisions/game, share, and over-tap share, and exits 0.
   Native: `cargo nextest run --manifest-path managym/Cargo.toml over_tap`
   (debug) passes a directed test: Mountain, Mountain, Forest paying {1}{G}
   counts one over-tap; Forest, Mountain paying {1}{G} counts none.
2. Part 3 shape table, same games, same command with `shapes`.
3. Learning comparison (launch under `--max-usd 25`), report, recommendation.

## Docs (written 2026-11-13, describe intended behavior)

- `experiments/mana_spending.md`: commands, output columns, what is counted,
  shape table; Results and Recommended design are placeholders.
- `experiments/README.md`: module-table row.

Details the docs settle that the plan left implied: CLI flags `--games`
(per deck), `--seed`, `--out` (default `experiments/data/mana_spending/`,
`games.jsonl`); deck-set names `elemental`, `tla`, `heartbeat`; `Env.info()`
keys `mana_payments`, `mana_payments_over_tapped`,
`mana_spend_decisions_over_tapped`; shape names per unit / automatic /
override / outcome / folded (A–E).

## Done when

`experiments/mana_spending.md` states, for Elemental and TLA Limited decks,
spending decisions per game and their share of all decisions, with and without
over-tap; compares the shapes on decisions and offers per decision; shows the
small learning comparison with its cost; and recommends one design with what
it changes for the policy and for Academy. Data is published and the manifest
committed.

## Implementation checks and remaining work

- `cargo test --manifest-path managym/Cargo.toml --lib over_tap -j 2 -- --test-threads 2`: passed (four positions in one directed test). The initial Heartbeat fixture assumed both lands tap for {1}{G}; corrected it to {2}{G} after the run showed one doubled Forest pays exactly.
- Discovery inventory: `uv run --extra dev pytest tests/eval/test_experiment_files.py::test_etude_finds_each_current_experiment_and_nothing_else -q`: 1 passed. This module has no Etude report/eval callback, so the inventory stays unchanged.
- MTG-183 final (`4f2ad656…`, run `f3748bcabac8`, 331180341 transitions) was downloaded and refused with `UnsupportedPolicyInput`: input v1, while this revision supports v9/v10. Retain the admission receipt with the frequency data; do not rebind it.
- Remaining in this slice: release extension, frequency run, report and artifact publication. Shapes and the learning comparison remain later slices, before the final recommendation. Full affected suites belong to gate.
