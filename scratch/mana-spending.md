# MTG-347 — higher-level player control of mana spending

## Outcome and scope

Explore higher-level payment control and recommend a design for review in
`experiments/mana_spending.md`. Implementing the production chooser belongs to
another Task. Keep defaults, policy input, Academy behavior, native payment
execution and the frozen A/B pilot unchanged. No Wave is selected.

All measurement is complete and published: frequencies, the A/B pilot, and the
paired shape counts with and without redundant taps. What remains is one
report-text slice (below): the recommendation and its questions still describe
the design the second concept review replaced. Earlier implementation history
and launch notes are in git `ce074e9d37` and the experiment report; do not rerun
the pilot or rewrite its evidence.

## Demo and done when

A reviewer opens `experiments/mana_spending.md` and, before any table, reads:
the recommended design (one native payment-outcome decision, automatic plan
first), then three numbered questions, each with its options and the measured
price of each. Evidence and limits follow; run logistics sit in a collapsed
appendix. Headless: `sed -n 1,60p experiments/mana_spending.md` shows the
summary, the recommendation and the start of the questions.

Done when the report recommends that single decision with D's costs, no longer
names an entry step as part of the design, asks the three questions in the form
below, and every number it quotes is one already in its tables. Nothing outside
the report and `scratch/` changes.

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

## Evidence already complete

- Original frequency source `58e5923c53`: 336 games on 21 exact deck lists,
  eight deals per deck/policy. Heuristic and random mirrors; MTG-183's input-v1
  checkpoint was refused rather than rebound. Ordinary pool prompts are
  0.11–0.60% of decisions; Heartbeat is 9.7–13.8%.
- Current baseline from rebuilt `0d2233a567`: same actions, counts and offers;
  main's engine/card merge changed digests. All 336 games replay verified.
  `current/shapes/` holds 14,775 payment rows, 9,089 casts, 420 capped casts;
  422 casts already certify more than 64 surplus-inclusive capability outcomes.
  Those bytes and the original baseline must survive unchanged.
- Pilot source `00ee188998cb`: two seeds per arm, 32 updates each, A per-unit
  versus B automatic pooled-unit spending. Evaluations
  `mana-spending-A-303f5f0a6a59` / `mana-spending-B-fe362b5c294d`: 800 valid
  replay-checked games. Scores A/B 26/27% and 28.5/32.5%. No improvement or
  equivalence established; no source-choice design tested. Missing-focus
  warnings and three recovered resource-limit failures (915 discarded samples)
  remain explicit in the report. All 24 pilot artifacts are published.

## Delete — do not maintain

Remove the report's unconditional flat-D capacity conclusion and its misleading
“grouped D” label. Replace with the paired measurements and an automatic/override
entry followed by an outcome choice (C then D), whose cost includes the entry.
Do not preserve a second enumeration/payment path. Keep all existing evidence,
PaymentModel authority, source/action behavior, frozen pilot revision and switch.

## Review direction and product choices

Concept review at `9dbe3f78df` found redundant taps counted as choices in part 3
but attributed as defects in part 2; that iteration added both counting rules
(text in git `ce074e9d37`).

Concept review at `1530995a9e` (2026-11-13), page `scratch/concept-review.html`.
Report text only: no code, run, artifact or evidence change.

### This slice: report text (kickoff 2026-11-13 kept this direction)

Kickoff re-derived the quoted numbers from the report's own tables and read the
engine precedents; findings are inline as *Checked*.

1. **One payment decision, automatic plan first; no separate entry.** The report
   defines C as entry + outcome choice, so "C followed by D" names C twice. Its own
   paired table shows the entry buys nothing where the menu fits: C's
   always-automatic bound equals D (Elemental heuristic 317.39 both) and every
   override costs one more decision (C up to 321.69), while non-redundant
   capability D is 2 / 4-7 / exact max 22 offers. Recommend one native
   payment-outcome decision whose first offer is the bound automatic plan,
   identified as automation. Where permitted outcomes exceed the action capacity,
   the remaining offers are native balanced groups and only an outcome commits
   (the `Decision::NameCard` / `NAME_BRANCHING` precedent). The entry step is that
   grouping's degenerate case, not a concept of its own. Keep the measured C
   columns as evidence; rewrite the recommendation, its quoted costs (D's, with
   C's as the price of an entry) and question 3 to match.
   *Checked:* D capability lower bounds over A are +3.7% / +7.0% (Elemental
   heuristic / random) and +2.5% / +4.5% (TLA), so "2.5-7.0%" holds; ordinary
   p95 is 4-7. The precedent is real and reusable in shape: `NameScope`
   (`managym/src/cards/names.rs`) publishes a balanced, complete partition
   where only a leaf commits. It partitions a sorted vocabulary by contiguous
   span; payment outcomes have no such public order, so the report may cite the
   precedent but must leave "how outcomes are ordered into groups" to the
   implementing Task, not present it as solved. The summary bullets, the shape
   table's C row and "Decision shapes" prose also say "C followed by D" in
   effect; change wording there, never the measured columns.
2. **Give question 3 both sides in numbers.** Override intent at cast selection
   ("cast X" / "cast X, choosing payment") is bounded by published columns
   without a run: decisions/game lie between E (never override) and D (always);
   a priority window gains at most one offer per cast offer (measured maximum 13
   offers per window). State its price beside it: the automatic path commits
   before modes, targets, X, the cost and therefore the plan are known, so the
   plan is not inspectable when chosen. Label these as arithmetic bounds, not a
   measured shape.
3. **Question 1 asks where surplus lives, not only whether.** The non-redundant
   menu is the set of minimal payments, the same set MTG-342's fixed tapper picks
   from. Deliberate floating is a different act (tapping for mana with priority,
   CR 605.3a), and putting it in the payment menu is what produces the 422 casts
   above 64. Present the options as: not offered; a tap-for-mana action at
   priority (uncounted here); or in the payment chooser (measured, needs groups).
   This is a proposal inside the question, not an answer.
   *Checked:* the report's "The engine has no explicit action for it today" is
   true only of fixed sources. Activated mana abilities (`mana_choices()` is
   some) are already `Action::ActivateAbility` offers at priority
   (`managym/src/flow/action.rs`, the `can_activate_mana_ability` branch) and
   during payment (`mana_ability_actions`); a land's fixed output is tapped only
   by the automatic payer (`PaymentStep::Fixed`, `flow/mana.rs`). So the
   priority option extends an existing action to fixed sources; say so, and
   scope that sentence. Its offer cost is still unmeasured.
4. **Lead with the decision.** Order: summary, recommended design, the three
   numbered questions, then evidence, limits. Move commands' output details,
   launch history, cost receipts and warning-log notes to a collapsed appendix.
   Retitle for the design question (file name unchanged).

Unchanged on purpose: `MANAGYM_EXPERIMENT_AUTOSPEND` and its README note stay as
written; the report already says the follow-up replaces the switch.

Open product choices, for the report's review (nobody has answered them; they do
not block this Task, whose outcome is the report that asks them):

1. Deliberate surplus tapping: not offered, a priority action, or in the chooser?
   Decides whether the flat menu suffices (exact max 22) or groups are required.
2. Which sources are the same choice: exact identity (493 non-redundant casts
   above 64, mostly identical basics) or equal definition and state (needs a
   rules argument; capability alone merges a land with a mana creature)?
3. Where the player opts out of automation: a decision at each payment whose
   outcomes differ (+2.5-7.0% decisions/game non-redundant capability, plan
   visible) or intent at cast selection (no added decisions unless used, plan
   not visible). Learning effects stay with MTG-265.

## Final implementation and results

- Existing PaymentModel capture now records non-redundant capability and exact
  identity outcomes on the same traversal/cap. Saved obligations use the actual
  post-tap board, matching over-tap attribution. No runtime default or shared
  policy/Academy behavior changed.
- `d18004357a` replayed all 336 current-baseline games, 14,775 payments and 9,089
  cast payments, in 421.48 s. Every original payment/folded field matches the
  earlier capture; all final digests, spend offers and counters agree.
- Non-redundant capability D: exact maximum 22, capped lower bounds reach 30,
  zero casts certified above 64; 420 capped casts remain unknown. Grouping is
  not implied on the exact capability subset. Exact identity still certifies
  493 non-redundant casts above 64 (291 exact, 202 capped), exact max 252.
  Non-redundant capability E reaches 75 on one complete window.
- All three artifacts under `current/non-redundant-shapes/` are published.
  Original/current shape evidence and frozen pilot bytes remain unchanged.
- Report names C followed by D, quotes entry/override costs, makes grouping
  conditional on reviewed surplus/equivalence choices, states MTG-342's
  dependency and ends with all three open review questions.
- Remaining: the report-text slice above, then gate/review and the caller's
  delivery steps. No PR created here.

## Checks

- `CARGO_BUILD_JOBS=2 cargo test --manifest-path managym/Cargo.toml -p managym --lib flow::mana::outcomes::tests -- --test-threads 2`: 5 passed in debug (0.05 s).
- Release wheel rebuilt for CPython 3.13 and extension replaced atomically.
- `uv run --no-sync ruff check experiments/mana_spending.py`: passed.
- Paired `shapes` / `summary`: 336 verified games; all original row projections
  unchanged and new count subset bounds hold.
- `uv run --no-sync --extra dev pytest tests/infra/test_experiment_data_untracked.py -q`: 1 passed (0.56 s).

Gate, for the report-text slice (base `caa15caca4`):

- `git diff --stat caa15caca4 -- . ':!experiments/mana_spending.md' ':!scratch'`:
  empty (no code, manifest or evidence change).
- `grep -c 'C followed by D' experiments/mana_spending.md`: 0.
- `uv run --extra dev pytest tests/eval/test_experiment_files.py tests/infra/test_experiment_data_untracked.py -q`: pass.
- Each figure quoted in the recommendation and questions appears in a table of
  the same report (317.39, 321.69, 22, 493, 422, 420, 75, 13).

At kickoff the checkout had no `.venv`; `uv run --no-sync` then creates an empty
one and imports fail. Sync from the lock (`uv run --extra dev`) and keep the
existing `managym/_managym.cpython-313-darwin.so`.

## Compress (2026-11-13)

- One owner for the over-tap criterion: `PaymentModel::is_surplus`
  (`managym/src/flow/mana.rs`), read by `record_mana_payment` and the
  non-redundant classification in `flow/mana/outcomes.rs`. The script folds its
  four parallel expansion counters into one list and derives the summary's flat
  C/D keys from the `capability` rule.
- Kept: the flat summary keys and the unused `pool_outcomes`/`source_outcomes`
  row fields, because published rows and summaries carry them.
- Check: rebuilt release extension, replayed all 336 games; `shapes.jsonl` and
  `summary.json` are byte-identical (`cmp`) to
  `current/non-redundant-shapes/`. Debug `flow::mana` tests: 27 passed.
  Published `metadata.json` hashes name the `d18004357a` build, not this one.
