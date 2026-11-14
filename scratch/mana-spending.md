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

The report-scope slice from concept review `59c82bc2be` is complete.
Next implement slice, report and read-only aggregation only: name the payment
decision the engine already has and price its overlap (section "The payment
decision already exists"). Then gate/review and the caller's delivery steps.
No PR created here.

## Demo and done when

A reviewer opens `experiments/mana_spending.md` and, before any table, reads:
the summary with the recommendation (one native payment-outcome decision,
automatic plan first), then three numbered questions, each with its options as
bullets and the measured price of each. The design, evidence and limits follow;
run logistics sit in a collapsed appendix. Headless:
`sed -n 1,110p experiments/mana_spending.md`.

Done when the report recommends that single decision with D's costs, names no
entry step as part of the design, and every number it quotes is one already in
its tables; the design says which payments it covers and prices the ones that
are not casts; the Forest illustration says which review question its
two-offer menu depends on; and the design names today's payment decision as the
one it extends, with the decisions D double-counts there priced. Since `caa15caca4` the branch changes only the
report, `scratch/` and the read-only `summary --all-payments` aggregation in
`experiments/mana_spending.py`. Published evidence and its manifest stay
unchanged.

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

## All-payment scope (concept review `59c82bc2be`)

The review found the recommendation priced casts only and the two-offer Forest
example assumed source equivalence; its direction is in git
`b0d7d6060e:scratch/mana-spending.md`. The report now covers every mana payment
(the capture marks cast versus non-cast only, not the rules reason) and the
example names its dependency on question 2.

`summary --all-payments` writes `all-payments-summary.json` beside the source
(`payment_scopes.noncast` / `.all`, same four counting rules). It is derived
locally from published rows and not published; the report's tables hold its
results. No games or training were rerun. The all-payment price did not change
the recommendation; the figures are under the open choices below.

The review's 9 Elemental / 2 TLA non-cast prompts count payment boundaries; the
rows hold 10 / 3 per-unit decisions across them, and the report gives both.

## The payment decision already exists (concept review `509878be2f`)

The report presents the recommended decision as new and says "Existing explicit
mana-ability decisions stay at their native boundaries". The engine already
asks at a payment: `PendingChoice::ManaPayment` (`flow/play.rs`,
`pending_choice_action_space` in `flow/action.rs`) and resolution
`Decision::Payment` (`flow/decision.rs`) publish kind `PayOrNot` with one
`ActivateAbility` per activatable mana ability plus `PayCost`, which runs the
automatic payer; Waterbend adds its taps and a resolution payment may add
`Decline`. It is shown only when a mana ability can be activated; otherwise
the payment commits at once.

Read literally, the report's design puts its outcome decision after that one:
activate-or-`PayCost`, then "automatic plan or another outcome". That is the
separate entry step the design rejects, at every payment with an activatable
source. The design implies one decision instead:

- **`PayCost` is the automatic-plan offer.** The recommended decision is
  today's payment decision, shown whenever its offers differ rather than only
  when a mana ability is activatable. The other outcomes are sibling offers
  beside the activations, Waterbend taps and `Decline`; they are recomputed
  each time the decision returns after an activation, as `PayCost` is today.
  "Payment-outcome decision" is not a second concept or a new decision kind.
- **D double-counts there.** D adds one decision per multi-outcome payment to
  a trajectory that already contains the `PayOrNot` decision. From the
  published paired rows (2026-11-13, at `509878be2f`), payments with an
  activated source still present at commit and at least two non-redundant
  capability outcomes, certified / including capped: Elemental heuristic
  149 / 154, random 28 / 28; TLA heuristic 320 / 330, random 99 / 99; Heartbeat
  none. Per game 1.86, 0.35, 4.00, 1.24 (80 games each), against D's
  all-payment additions of 13.5, 24.8, 12.9, 24.1. The capture records that a
  source has abilities, not that an activation was offered, so these bound the
  saving from above; say so.

Direction for the report: in "Recommended design", state the existing decision
and that the design extends it; replace the "stay at their native boundaries"
sentence; add the overlap column or sentence with the merged lower bound beside
D's all-payment figures, keeping published D numbers as they are (they are the
measured counting rule). Compute the overlap inside `summary --all-payments`
from `activated_sources`, not by hand. How grouped outcomes sit beside
activations is for the implementing Task. No games, native code or published
bytes change.

## Open product choices (nobody has answered; they do not block this Task)

1. Deliberate surplus tapping: not offered, a priority action, or in the
   chooser? Decides whether the flat menu suffices (all-payment exact max 28) or groups are
   required.
2. Which sources are the same choice: exact identity (655 non-redundant payments
   above 64, including combinatorial subsets of identical basics) or equal definition and state (needs a
   rules argument; capability alone merges a land with a mana creature)?
3. Where the player opts out of automation: a decision at each payment whose
   outcomes differ (+2.9–11.8% decisions/game across all payments under
   non-redundant capability, plan visible), or intent at cast selection (no
   added cast decisions unless used, plan not visible; non-casts still need
   control). Learning effects stay with MTG-265.

## Checks

Gate, for the report and read-only aggregation slice:

- `git log --first-parent --name-only caa15caca4..HEAD`: branch commits touch
  only the report and `scratch/` before this scope extension; the extension
  also changes the read-only aggregation in `experiments/mana_spending.py`,
  with no native, manifest or evidence change. A
  plain diff from `caa15caca4` now includes main's #520 via merge `67e03ef138`.
- `grep -c 'C followed by D' experiments/mana_spending.md`: 0.
- `uv run --extra dev pytest tests/eval/test_experiment_files.py tests/infra/test_experiment_data_untracked.py -q`: pass.
- Recommendation and review questions distinguish all payments from cast-only
  figures; all-payment intervals and capacity totals match the derived summary.
  Percentages use the unrounded decision counts; group navigation is excluded.

Run at implementation (2026-11-13): `summary --all-payments` played no games;
independent raw-row checks passed for all 48 scope/rule D and C bounds and the
prompt reconciliation. Without `payment_scopes` the output equals the published
`summary.json` byte-for-byte, and SHA256 of the three published paired-evidence
files matches the manifest. Focused
`uv run --extra dev pytest tests/eval/test_experiment_files.py -q -k etude_finds_each_current_experiment_and_nothing_else`:
1 passed. Ruff and `git diff --check` clean. Compress then reworded report
prose only (no figure or code change): `git diff --check` clean.

Realign at `65fb0de8f0` (2026-11-13): rerunning `summary --all-payments`
reproduced `all-payments-summary.json` byte-identically (SHA256 `cb66eb31…`);
its rows give the report's 2.9–11.8%, exact max 28 and 655 identity >64;
the two focused test files pass (6). Nothing to correct.

For the existing-decision slice: the overlap figures in the report equal the
derived summary; without `payment_scopes` the output still equals published
`summary.json` byte-for-byte; the two focused test files pass.

If native code changes again: debug `flow::mana` tests
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
