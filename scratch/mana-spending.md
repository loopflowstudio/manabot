# MTG-347 — higher-level player control of mana spending

## Outcome and scope

Explore higher-level payment control and recommend a design for review in
`experiments/mana_spending.md`. Implementing the production chooser belongs to
another Task. Keep defaults, policy input, Academy behavior, native payment
execution and the frozen A/B pilot unchanged. No Wave is selected.

All measurement is complete and published: frequencies, the A/B pilot, and the
paired shape counts with and without redundant taps. The report recommends one
native payment-outcome decision with the automatic plan first and asks three
review questions. Do not rerun the pilot or rewrite its evidence. Earlier plan
text, launch notes and per-slice checks are in git `78397f2042` and before
(`ce074e9d37` for the first concept review).

Remaining: gate/review and the caller's delivery steps. No PR created here.

## Demo and done when

A reviewer opens `experiments/mana_spending.md` and, before any table, reads:
the recommended design (one native payment-outcome decision, automatic plan
first), then three numbered questions, each with its options and the measured
price of each. Evidence and limits follow; run logistics sit in a collapsed
appendix. Headless: `sed -n 1,60p experiments/mana_spending.md`.

Done when the report recommends that single decision with D's costs, names no
entry step as part of the design, and every number it quotes is one already in
its tables. The report-text slice (base `caa15caca4`) changes nothing outside
the report and `scratch/`.

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

## Evidence (all published; bytes must survive unchanged)

- Frequency, source `58e5923c53`: 336 games on 21 exact deck lists, eight
  deals per deck/policy, heuristic and random mirrors; MTG-183's input-v1
  checkpoint was refused rather than rebound. Ordinary pool prompts are
  0.11–0.60% of decisions; Heartbeat is 9.7–13.8%.
- Current baseline, rebuilt `0d2233a567`: same actions, counts and offers;
  main's engine/card merge changed digests. `current/shapes/`: 14,775 payment
  rows, 9,089 casts, 420 capped; 422 casts certified above 64
  surplus-inclusive capability outcomes.
- Paired counts, `d18004357a`, `current/non-redundant-shapes/`: same games in
  421.48 s; every original field unchanged. Non-redundant capability D: exact
  max 22, capped lower bounds reach 30, none certified above 64. Exact
  identity: 493 non-redundant casts above 64 (291 exact, 202 capped), exact
  max 252. Non-redundant capability E reaches 75 on one complete window.
  Published `metadata.json` hashes name the `d18004357a` build; the compress
  at `1530995a9e` reproduced `shapes.jsonl` and `summary.json` byte-identically.
- Pilot, source `00ee188998cb`: two seeds per arm, 32 updates, A per-unit
  versus B automatic pooled-unit spending. Evaluations
  `mana-spending-A-303f5f0a6a59` / `mana-spending-B-fe362b5c294d`: 800 valid
  replay-checked games; scores A/B 26/27% and 28.5/32.5%. No improvement or
  equivalence established; no source-choice design tested. Missing-focus
  warnings and three recovered resource-limit failures (915 discarded samples)
  are explicit in the report. All 24 pilot artifacts are published.

## Delete — do not maintain

Nothing remains to delete. The report no longer carries the unconditional
flat-D capacity conclusion, the "grouped D" label or a separate
automatic/override entry in the design; C's measured columns stay as
comparison evidence. Do not add a second enumeration/payment path. Keep
`PaymentModel` authority (`PaymentModel::is_surplus` in
`managym/src/flow/mana.rs` owns the over-tap criterion), the frozen pilot
revision and `MANAGYM_EXPERIMENT_AUTOSPEND` with its README note; the report
says the follow-up replaces the switch. The flat summary keys and the unused
`pool_outcomes`/`source_outcomes` row fields stay because published rows carry
them.

## Design reasoning behind the report (concept review `1530995a9e`, page `scratch/concept-review.html`)

1. **One payment decision, no separate entry.** C is entry + outcome choice;
   its always-automatic bound equals D (Elemental heuristic 317.39 both) and
   every override costs one more decision (up to 321.69). Overflow uses native
   balanced groups where only an outcome commits, after `Decision::NameCard` /
   `NAME_BRANCHING`. `NameScope` (`managym/src/cards/names.rs`) partitions a
   sorted vocabulary by contiguous span; payment outcomes have no such public
   order, so ordering outcomes into groups is left to the implementing Task.
2. **Question 3's cast-selection option is arithmetic, not a measured shape:**
   decisions/game between E (305.60 / 210.91 / 437.26 / 216.22, never
   override) and D; at most one more offer per cast offer (13 on fully
   measured windows). Its price: the automatic path commits before modes,
   targets, X and cost are known.
3. **Question 1 asks where surplus lives.** The non-redundant menu is the set
   of minimal payments MTG-342's tapper picks from; deliberate floating is
   tapping for mana with priority (CR 605.3a). Activated mana abilities
   (`mana_choices()` is some) are already `Action::ActivateAbility` offers at
   priority (`managym/src/flow/action.rs`, `can_activate_mana_ability`) and
   during payment (`mana_ability_actions`); a land's fixed output is tapped
   only by the automatic payer (`PaymentStep::Fixed`, `flow/mana.rs`). The
   priority option's offer cost is unmeasured.
4. The over-tap figure is 6–15% of ordinary payments (400/2717, 538/4430,
   213/2588, 288/4742); earlier text said 8–15%.

## Open product choices (nobody has answered; they do not block this Task)

1. Deliberate surplus tapping: not offered, a priority action, or in the
   chooser? Decides whether the flat menu suffices (exact max 22) or groups are
   required.
2. Which sources are the same choice: exact identity (493 non-redundant casts
   above 64, mostly identical basics) or equal definition and state (needs a
   rules argument; capability alone merges a land with a mana creature)?
3. Where the player opts out of automation: a decision at each payment whose
   outcomes differ (+2.5–7.0% decisions/game non-redundant capability, plan
   visible) or intent at cast selection (no added decisions unless used, plan
   not visible). Learning effects stay with MTG-265.

## Checks

Gate, for the report-text slice:

- `git diff --stat caa15caca4 -- . ':!experiments/mana_spending.md' ':!scratch'`:
  empty (no code, manifest or evidence change).
- `grep -c 'C followed by D' experiments/mana_spending.md`: 0.
- `uv run --extra dev pytest tests/eval/test_experiment_files.py tests/infra/test_experiment_data_untracked.py -q`: pass.
- Each figure quoted in the recommendation and questions appears in a table of
  the same report (317.39, 321.69, 22, 493, 422, 420, 75, 13).

If code changes again: debug `flow::mana` tests
(`CARGO_BUILD_JOBS=2 cargo test --manifest-path managym/Cargo.toml -p managym --lib flow::mana -- --test-threads 2`,
27 passed at `1530995a9e`), rebuild the release extension, and `cmp` a fresh
`shapes`/`summary` against `current/non-redundant-shapes/`.

A checkout with no `.venv` makes `uv run --no-sync` create an empty one and
imports fail. Sync from the lock (`uv run --extra dev`) and keep the existing
`managym/_managym.cpython-313-darwin.so`.

## Compress (2026-11-13, report text)

Prose only, on top of `78397f2042`: the four counting definitions
(redundant/non-redundant, capability/identity) are stated once under "Decision
shapes" and referenced from the paired section; repeated shape restatements,
duplicate pilot caveats and appendix query-plan detail are cut.
Check: all 140 table rows and every heading are identical to `78397f2042`,
all in-page anchors resolve, `git diff --check` clean.
