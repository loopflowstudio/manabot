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

Remaining: the report-scope slice below (concept review at `59c82bc2be`),
then gate/review and the caller's delivery steps. No PR created here.

## Demo and done when

A reviewer opens `experiments/mana_spending.md` and, before any table, reads:
the summary with the recommendation (one native payment-outcome decision,
automatic plan first), then three numbered questions, each with its options as
bullets and the measured price of each. The design, evidence and limits follow;
run logistics sit in a collapsed appendix. Headless:
`sed -n 1,95p experiments/mana_spending.md` (first table is at line 100).

Done when the report recommends that single decision with D's costs, names no
entry step as part of the design, and every number it quotes is one already in
its tables. Also done when (next slice): the design says which payments it
covers and prices the ones that are not casts, and the Forest illustration says
which review question its two-offer menu depends on. The report-text slice (base `caa15caca4`) changes nothing outside
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

## Next implement direction (concept review at `59c82bc2be`; report and `scratch/` only unless noted)

The report matches the earlier review's direction. Two gaps remain between the
design it recommends and what it shows:

5. **The design is one decision per mana payment, and the report prices only
   casts.** The recommendation says "per cast" and "replaces today's per-pip
   prompts for cast payments"; C–E "retain noncast payment behavior". A player
   would then learn two payment interactions (outcome menu for spells,
   automatic tapping plus per-unit prompts for ability costs) with the source
   choice still hidden in the second. Nothing anyone said restricts the Task to
   casts ("letting users decide how to spend mana"). State the design as
   covering every mana payment, and report the non-cast price beside the cast
   one, from the published rows. Read at this review from
   `current/non-redundant-shapes/shapes.jsonl` (`cast: false`): 5,686 of
   14,775 payments are not casts (Elemental 2,496, TLA 3,190, Heartbeat 0);
   3,774 have a mana cost (1,544 / 2,230), 134 of those capped (68 / 66). Of
   the completely enumerated ones, 933 / 1,220 have two or more non-redundant
   capability outcomes (1,017 / 1,452 by exact identity); exact maxima 28 / 22
   capability, 252 / 252 identity. Only 9 / 2 drew a per-unit prompt today.
   So the hidden choice is about as common there as on casts, and the
   headline "+2.5–7.0% decisions per game" is a cast-only figure. Recompute
   these through the experiment's own summary path, give decisions per game
   for all payments next to the cast-only columns, and correct the summary,
   question 3 and Limits wording to say which population each figure covers.
   Published `shapes.jsonl` / `summary.json` bytes stay unchanged: a new
   derived table needs a new output name, not a rewritten summary. If that
   needs a code change in `experiments/mana_spending.py`, it is a read-only
   aggregation over published rows; no games rerun.
6. **The illustration silently answers question 2.** Its menu shows two offers
   for Forest, Forest, Mountain, which treats the two Forests as one choice,
   while the text below says the recommendation "preserves exact identity
   until an equivalence rule is reviewed" (three offers). Say so next to the
   table, as it already does for the redundant tap and question 1.

Assumption (also in `scratch/questions.md`): covering every mana payment is
what "one payment decision" already means, so it is written as direction, not
asked as a fourth question. If the all-payments price changes the
recommendation, say that in the report rather than narrowing the design.

`scratch/concept-review.html` is the page of the first review (`1530995a9e`)
and predates these two points.

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

- `git log --first-parent --name-only caa15caca4..HEAD`: branch commits touch
  only the report and `scratch/` (no code, manifest or evidence change). A
  plain diff from `caa15caca4` now includes main's #520 via merge `67e03ef138`.
- `grep -c 'C followed by D' experiments/mana_spending.md`: 0.
- `uv run --extra dev pytest tests/eval/test_experiment_files.py tests/infra/test_experiment_data_untracked.py -q`: pass.
- Each figure quoted in the recommendation and questions appears in a table of
  the same report (317.39, 321.69, 22, 493, 422, 420, 75, 13, 712, 49–59);
  percentages are ratios of its columns (question 1's 4–11% is surplus D over
  A, 1.037–1.112; 2.5–7.0% is non-redundant D over A; 26 is 2 × 13).

If code changes again: debug `flow::mana` tests
(`CARGO_BUILD_JOBS=2 cargo test --manifest-path managym/Cargo.toml -p managym --lib flow::mana -- --test-threads 2`,
27 passed at `1530995a9e`), rebuild the release extension, and `cmp` a fresh
`shapes`/`summary` against `current/non-redundant-shapes/`.

A checkout with no `.venv` makes `uv run --no-sync` create an empty one and
imports fail. Sync from the lock (`uv run --extra dev`) and keep the existing
`managym/_managym.cpython-313-darwin.so`.

## Report-text passes (2026-11-13)

Prose only, on top of `78397f2042`: the four counting definitions are stated
once under "Decision shapes"; repeated restatements are cut. The three review
questions then moved ahead of the design, their options as bullets rather than
tables. Check: the 126 remaining table rows are the 140 of `78397f2042` minus
the three option tables (14 rows); option text and every quoted figure are
unchanged; all in-page anchors resolve; `git diff --check` clean.
