# MTG-347 — higher-level player control of mana spending

## Outcome and scope

Explore higher-level payment control and recommend a design for review in
`experiments/mana_spending.md`. Implementing the production chooser belongs to
another Task. Keep defaults, policy input, Academy behavior, native payment
execution and the frozen A/B pilot unchanged. No Wave is selected.

The frequency measurements, pilot and first shape pass are complete. The current
iteration adds the missing non-redundant capability/identity counts, replays the
current baseline, publishes separate artifacts and corrects the recommendation.
Earlier implementation history and launch notes are in git `ce074e9d37` and the
experiment report; do not rerun the pilot or rewrite its evidence.

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

Concept review at `9dbe3f78df` identified redundant taps counted as choices in
part 3 but attributed as defects in part 2. The iteration adds both counting
rules, corrects C-vs-D decision costs and leaves three questions unanswered:
allow deliberate surplus tapping; source equivalence (identity vs definition and
state); accept the price of a payment entry or explore earlier override intent.
The automatic plan depends on MTG-342's fix; learning effects remain MTG-265's.
The complete review text is in git `ce074e9d37`; final measurements and questions
are in the report, not a separate protocol.

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
- Remaining: gate/review and caller's delivery steps. No PR created here.

## Checks

- `CARGO_BUILD_JOBS=2 cargo test --manifest-path managym/Cargo.toml -p managym --lib flow::mana::outcomes::tests -- --test-threads 2`: 5 passed in debug (0.05 s).
- Release wheel rebuilt for CPython 3.13 and extension replaced atomically.
- `uv run --no-sync ruff check experiments/mana_spending.py`: passed.
- Paired `shapes` / `summary`: 336 verified games; all original row projections
  unchanged and new count subset bounds hold.
- `uv run --no-sync --extra dev pytest tests/infra/test_experiment_data_untracked.py -q`: 1 passed (0.56 s).

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
