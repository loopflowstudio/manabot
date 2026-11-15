# MTG-347 — higher-level player control of mana spending

## Implementation status — 2026-11-15

Choice 1 remains unresolved after reading the preceding review and current
source/docs. Missing input: choose the bounded distinct-mana-cost heuristic
(1a) or playable-action preservation (1b). The supplied direction says
"Stop for real unresolved product choices"; neither objective is selected
here. No implementation, sync or publication performed. Existing edits and
the parent stack remain intact. Builds/tests skipped because no code changed.

## Write docs — 2026-11-15

Shipping doc path: `managym/README.md`, automatic-payer paragraph. No doc
edits made: choice 1 remains unanswered, so replacing its promise with the
proposed 1a wording would select a product objective. Exact open question:
should automatic payment preserve distinct mana-cost options with partial
checks, or actual playable cards and abilities with their required resources
and legality? The existing `docs/academy.md` pending-decision and stale-Command
contracts already cover recovery and need no revision.

`lf open --host desktop --file managym/README.md` returned "No Desktop Files
view attached; nothing opened." The docs remain at that path. Publication
and implementation remain stopped under the supplied "Stop for real unresolved
product choices" instruction; the existing review and kickoff edits are kept.

## Kickoff — 2026-11-15

Planning remains blocked on concept-review choice 1 below. The supplied
direction says "Stop for real unresolved product choices." No answer choosing
distinct mana-cost options or playable cards/abilities accompanies it. Neither
the existing code nor permission to pursue and publish resolves that choice.

Source inspection at `b1ecacd974` confirms the conflict:
`Game::payment_followups` in `managym/src/flow/mana.rs` deduplicates hand
costs, caps the probe, and excludes single-use abilities by cost equality;
`Chooser::affordable` in `mana/tap_choice.rs` unions affordability across
possible remainders. The automatic-payer paragraph in `managym/README.md`
instead promises the most hand cards and battlefield abilities affordable.
These objectives can choose different taps. Option 1a changes the stated
promise; option 1b requires a different probe. Neither is an implementation
detail that this plan can settle.

The next slice cannot be selected until that objective is supplied. Keep the
existing concept review and its concrete payment/reconnect acceptance scenarios;
do not substitute the inherited #523 report plan for this child implementation.
No source edits, runtime probes, sync or publication were performed in kickoff.
Frozen evidence, the stack and unrelated work remain untouched. Context is
within both configured limits; inherited history need not be compressed here.

## PR #524 concept review — 2026-11-15

Rechecked in session `session_c0ae492837834478b84ff0a9d5d10323`: HEAD remains
`b1ecacd974`; the live PR #524 description still promises "keeps the most of
the payer's other hand cards and battlefield abilities affordable" and remains
based on `jack/mtg-347-explore-higher-level-player-control-of-mana` (#523).
The supplied follow-on direction contains no answer to choice 1. Outcome stays
**Blocked**, not an implement iteration. Normal payment, failure and reconnect
contracts below remain the acceptance scenarios; no runtime proof was run.

This section governs the current automatic-payer review at `b1ecacd974`.
The measurement-only scope below is inherited from #523; it does not describe
the implementation now on this child branch. #524 remains stacked on #523.
Jack Heart's supplied instruction: "can you run concept review > pursue flows
on the 3 outstanding prs". This phase is review only.

**Outcome: Blocked on the meaning of a preserved follow-up.** The PR promises
the most affordable hand cards and abilities, but implements a bounded score
over distinct mana-cost records. It does not establish that those actions can
actually be taken. Neither the inherited four chooser questions nor the PR
description settles this difference. Do not treat the existing implementation
as a person's decision.

### Experience and concrete semantics

Existing interaction: cast or activate, reach native payment, use `PayCost`
when offered, then finish any native pooled-unit choices. A trivial payment
may be skipped. #524 changes the automatic fixed sources; it adds no source
chooser, preview, cancellation or new Command. With Mountain, Mountain,
Forest paying {1}{G}, the payer needs one Mountain and the Forest, leaving the
other Mountain untapped. With a Forest and a green mana creature paying {G}
and equal follow-up scores, it prefers the Forest. Follow-up score takes
precedence over preserving a creature for combat; this ordering is already
explicit in the PR's stated outcome, not a new question here.

The simpler contract is one native automatic payment with two distinct
responsibilities: legality from `PaymentModel`, preference from `choose_taps`.
Source classes are an internal search compression, not a declaration that
the sources are strategically interchangeable. `produce_mana` commits the
chosen fixed taps; native spending retains ownership of the exact receipt.
Keep pending obligations, source identities and actual pools as authority;
no persisted preference session or browser payment model is needed.

### Findings and next direction

- **The objective needs an explicit product contract.** `payment_followups`
  deduplicates hand entries by `ManaCost`, takes at most 64 records, and omits
  targets, timing within the turn and several nonmana costs. Ability records
  also carry a source reservation when their own cost taps it. Thus three
  cards sharing one cost can count less than two distinct costs. An ability
  with an unpayable discard or sacrifice cost can still influence tapping.
  Calling these "cards kept castable" hides an observable strategic choice.
- **Possible spending is not a promised remaining pool.** `Chooser::affordable`
  unions follow-ups affordable under different enumerated remainders. Those
  follow-ups need not coexist after the player's eventual unit choice. The
  search also bounds remainders and candidate checks, so "most" is a heuristic
  objective, not a global optimum guarantee. Saved obligations receive surplus
  pruning only. These limits belong in usage wording regardless of choice 1.
- **Shared parent probe needs an identity counterexample.** The single-use
  ability exclusion compares `ability.mana_cost == *cost`; two distinct
  sources can have equal costs, and equality does not identify the activation
  being paid for. Next implementation should check two equal-cost abilities
  on distinct sources and exclude only the actual unavailable activation.
  Coordinate that one fixture and `PaymentProbe` semantics with #523; keep
  one owner, without merging either branch or rerunning its measurements.

### Failure and recovery walkthrough

If fixed sources cannot pay, `produce_mana` returns its existing error before
tapping. Search-budget exhaustion retains a payable candidate; it must not
become payment failure or a new human decision. If a player dislikes the
sources after they have tapped, reconnect is not undo: restore the actual
pending spending decision and already-committed taps. A stale or duplicate
Command must leave state unchanged. `docs/academy.md` already states that
contract; leave it alone. A future proof should interrupt between production
and unit spending, reconnect, reject the old Command, and finish with the same
receipt as uninterrupted play. This was inspected as a contract, not executed
in this review. Replay/fork behavior remains tied to the original revision;
do not rewrite frozen traces to make a changed preference agree.

### Proposed usage wording, pending choice 1

For option 1a: "Automatic payment avoids unnecessary fixed-source taps and
uses a bounded estimate of remaining mana options to prefer sources. It scores
distinct mana costs, not the number of playable cards; targets and other costs
can still prevent a follow-up. Ties favor hand costs, then instant-timing hand
costs, then less reluctant sources and colorless output. Saved obligations
only receive surplus pruning. Any remaining mana-unit choice is still yours."

Put the eventual wording in the automatic-payer paragraph of `managym/README.md`
and align the PR description. No skill instruction needs a new workflow.

### Open question for loop-or-next

1. What should "keeps follow-ups affordable" mean?
   - **1a — bounded mana-option heuristic:** retain distinct cost records and
     partial affordability checks, document their limits and fix identity
     attribution. Small scope; it can prefer an option that is not playable.
   - **1b — playable-action preservation:** count actual cards/abilities and
     their required resources/legality. This changes the objective and needs
     a larger shared probe design and bounded execution strategy.

Do not implement either answer as settled. The earlier outcome-menu questions
belong to the separate chooser proposal, not to this automatic-payer patch.
Show next: the equal-cost-card counterexample, a reserved-source ability, and
mid-payment reconnect under the chosen contract. No training or baseline run
is needed to resolve the product choice. Review artifacts only were edited;
implementation, evidence, parent branch and independent MTG350 work were untouched.

## Outcome and scope

Explore higher-level payment control and recommend a design for review in
`experiments/mana_spending.md`. Implementing the production chooser belongs to
another Task. Keep defaults, policy input, Academy behavior, native payment
execution and the frozen A/B pilot unchanged. No Wave is selected.

All measurement is complete and published: frequencies, the A/B pilot, and the
paired shape counts with and without redundant taps. The report recommends
extending today's mana-payment decision, `PayCost` (the automatic plan) first,
and asks four review questions before its first table. Do not rerun the pilot
or rewrite its evidence. Nothing remains to implement in this Task (checked at
`5040b148b4`); the production chooser needs the separate post-review Task. No
PR created here. Earlier plan text, launch notes and per-slice checks are in
git `78397f2042` and before (`ce074e9d37` for the first concept review).

## Demo and done when

A reviewer opens `experiments/mana_spending.md` and, before any table, reads:
the summary with the recommendation (today's mana-payment decision, extended,
`PayCost` as the automatic plan first), then four numbered questions, each with its options as
bullets and the measured price of each. The design, evidence and limits follow;
run logistics sit in a collapsed appendix. Headless:
`sed -n 1,150p experiments/mana_spending.md`.

Done when the report recommends that single decision with D's costs, names no
entry step as part of the design, and every number it quotes is one already in
its tables; the design says which payments it covers and prices the ones that
are not casts; the Forest illustration says which review question its
two-offer menu depends on; and the design names today's payment decision as the
one it extends, with the decisions D double-counts there priced; the report
names that decision as the mana-payment decision rather than by its shared
`PayOrNot` kind, and presents per-source taps inside it as an unanswered
alternative with its bounds. Since `caa15caca4` the branch changes only the
report, its index row, `scratch/` and the read-only `summary --all-payments`
aggregation in `experiments/mana_spending.py`. Published evidence and its manifest stay
unchanged.

## Docs

The docs that ship with this Task are `experiments/mana_spending.md` (the
report), its row in `experiments/README.md` (`count` / `shapes` / `summary`)
and "Offline payment-shape measurement" in `managym/README.md` (the capture
API and `MANAGYM_EXPERIMENT_AUTOSPEND`). The recommended chooser is not built
here, so no guide describes it; `docs/academy.md` and the engine guides stay
as they are until the implementing Task.

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

Nothing remains to delete. Removed the separate-payment-decision description
and the sentence retaining mana-ability decisions as separate boundaries;
published D counts and existing activation/tap/decline offers survive.
The report no longer carries the unconditional
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
  a trajectory that already contains the `PayOrNot` decision. The capture
  records that a source has abilities (`activated_sources`), not that an
  activation was offered, so the overlap bounds the saving from above.

The report states this in "Recommended design" and prices it in "Existing
payment-decision overlap"; published D figures stay beside the merged bounds.
`summary --all-payments` derives `existing_decision_overlap` per scope and
counting rule (`_payment_overlap`): certified / possible payments (Elemental
149 / 154 and 28 / 28, TLA 320 / 330 and 99 / 99, Heartbeat none), a maximum
saving of possible / games (1.93, 0.35, 4.12, 1.24 against D's all-payment
additions of 13.5, 24.8, 12.9, 24.1), and a merged lower bound of D's lower
bound minus certified / games, because a capped row's uncertain decision and
its overlap cancel. How grouped outcomes sit beside activations is for the
implementing Task. No games, native code or published bytes changed.

## Per-source steps and the decision's name (concept review `d947e7e425`)

Implemented in the report. The review's reasoning follows; no games, native
code or published bytes changed.

**1. Name the decision for what it is.** `ActionSpaceKind::PayOrNot` is not
one decision. Its doc comment is "You may pay [cost]"; it is published for the
mana payment of a cast or activation (`PendingChoice::ManaPayment`), a
resolution payment (`Decision::Payment`), the kicker opt-in
(`PendingChoice::KickerChoice`: `PayCost` / `Decline`) and an upfront
mana-ability fee (`Decision::ManaUpfront`: activations only, no `PayCost`).
`PayCost` likewise means "run the automatic payer" at a mana payment, "pay the
kicker" at `KickerChoice`, and "pay mana instead of sacrificing" at
`PendingChoice::SacrificePayment`. A cast's mana payment has no "or not". The
report's summary and plan say "today's `PayOrNot` decision ... with `PayCost`
as the automatic plan", which reads as covering all of these. It should call
the thing the mana-payment decision (decision context stage `payment`), say
once that it is published today under kind `PayOrNot`, and say the kicker
opt-in and the sacrifice-or-mana choice are other decisions whose `PayCost`
is not a payment plan and which the design leaves alone. Whether the kind or
action is renamed is for the implementing Task (serialized values).

**2. The decision the design extends already has a second way to choose
sources, and the report never weighs it.** In the mana-payment decision an
activated mana ability is a step: `ActivateAbility` commits that source and
the decision returns (`mana_ability_actions`; offered only while the pending
obligations stay payable, `activation_witness`). A fixed-output source
(`PaymentSource.fixed`, `PaymentStep::Fixed`) is never offered; only the
automatic payer taps it. The recommendation therefore gives two kinds of
source two controls inside one decision: step offers for activated abilities,
whole-payment outcome offers for fixed sources. The alternative the existing
decision implies is one control: offer each fixed source as a tap step too,
with `PayCost` finishing automatically from what is left. Shapes A–E do not
include it; "Not recommended" does not mention it; the report's own prior art
(Arena full control, Forge manual selection) is this shape; and question 1's
"tap-for-mana action at priority" would bring it into the payment decision
anyway, because the report says activations are offered "at priority and
during payment", after which the outcome menu is a shortcut over a control
that already exists rather than the only control.

The report's "Per-source bounds" table holds the derived figures (published
`current/non-redundant-shapes/shapes.jsonl`; positive-cost payments that are
capped or have at least two non-redundant identity outcomes). Per-source steps
keep exact identity in a flat menu of at most 16 fixed taps on ordinary decks
(25 on Heartbeat) beside the activations and `PayCost`: no groups, no outcome
enumeration or cap, and question 2 becomes labelling. Their price is one
decision per tap plus the finishing `PayCost`, pooled-unit spending left to
per-pip prompts or automation, and the "low-level payment steps" the Task set
out to avoid. Override rates are unmeasured for either shape. The
recommendation stays as it is unless someone answers choice 4. The review's
full direction is in git `5fa62e0d21:scratch/mana-spending.md`.

## Entering a group (concept review `15cb04c339`)

Implemented in the report. Walking the recovery path of the grouped menu: the
design's precedent, `Decision::NameCard`, narrows one way. `NameScope::option`
(`managym/src/cards/names.rs`) offers only child groups or names, and
`Decision::NameCard`'s action space (`flow/decision.rs`) has no offer that
returns to a wider scope. That is harmless for naming, where every leaf is a
name. The mana-payment decision's root also holds the automatic plan, the
activation steps, Waterbend taps and, at an optional resolution payment,
`Decline`. Copied literally, a player who enters a group can no longer take the
automatic plan, activate a mana ability or decline: the group entry itself
turned down automation and made an optional payment mandatory. That contradicts
the report's own "only an outcome commits" and "Every permitted outcome stays
reachable", and a mis-entered group in Academy has no recovery.

The report's one sentence on this, "group navigation offers no cancellation or
undo of committed costs or revealed information", reads either way.

The report now says, as implied by "only an outcome commits" rather than
chosen: entering a group commits nothing, so every root offer stays reachable
from inside it; a shared return offer is allowed by Academy's cancellation rule
and the mechanism (return offer or repeated root offers) is the implementing
Task's; card naming is the precedent for balanced grouping only; leaving a
group is navigation, not undo of a committed payment or activation; and
group-navigation costs, return steps included, are unmeasured. This applies
only if groups are built at all (questions 1, 2 and 4).

Check (2026-11-13): comparison against HEAD showed every table row, numeric
figure and the four opening questions unchanged; `git diff --check` passed.
Compress then removed the report's second statement of the return-offer
choice; prose only.

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
4. How a player names the sources to tap: one outcome offer for the whole
   payment (recommended; one decision, combinatorial menu, groups under exact
   identity) or per-source tap steps in the same decision (flat menu of at
   most 16–25 taps, exact identity, one decision per tap, spend receipt left
   to per-pip prompts or automation). It decides whether outcome enumeration
   and groups are built at all.

## Checks

Gate, for the report and read-only aggregation slice:

- `git log --no-merges --first-parent --name-only caa15caca4..HEAD`: branch
  commits touch only the report, its `experiments/README.md` row, the
  read-only aggregation in `experiments/mana_spending.py` and `scratch/`; no
  native, manifest or evidence change (checked at `9dd498502f`). Without
  `--no-merges` the log includes main's merges (#520 via `67e03ef138`,
  `05d27fb273`).
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

If native code changes again: debug `flow::mana` tests
(`CARGO_BUILD_JOBS=2 cargo test --manifest-path managym/Cargo.toml -p managym --lib flow::mana -- --test-threads 2`,
27 passed at `1530995a9e`), rebuild the release extension, and `cmp` a fresh
`shapes`/`summary` against `current/non-redundant-shapes/`.

A checkout with no `.venv` makes `uv run --no-sync` create an empty one and
imports fail. Sync from the lock (`uv run --extra dev`) and keep the existing
`managym/_managym.cpython-313-darwin.so`.

## Later prose and aggregation passes (2026-11-13)

Each pass after `78397f2042` (questions ahead of the design, the existing
payment decision and its overlap, per-source bounds, group entry) was checked
the same way: every earlier table row and quoted figure unchanged, in-page
anchors resolving, `git diff --check` clean, and where the aggregation changed,
`summary --all-payments` reproducing `all-payments-summary.json` byte-identically
with `summary.json` untouched and the three published evidence SHA256s matching
the manifest. A raw-row projection reproduced all six per-source bounds rows
and all six overlap rows. Last test run:
`uv run --extra dev pytest tests/eval/test_experiment_files.py tests/infra/test_experiment_data_untracked.py -q`: 6 passed.
