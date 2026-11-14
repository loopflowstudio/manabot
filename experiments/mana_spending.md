# How should players choose which mana to spend? — MTG-347

Exploratory. Today a mana payment taps its sources automatically; when the pool
then holds more usable mana than the cost needs, the player chooses which pooled
unit pays each pip (`mana_spend` decisions). That prompt is rare, and it hides the choice
that differs far more often: which sources stay untapped.

- **Recommendation, for review:** one native decision at every mana payment
  whose offers differ: today's `PayOrNot` decision, extended, with `PayCost`
  as the automatic plan, listed first.
  [Recommended design](#recommended-design).
- **Today's prompt is rare.** 0.11–0.60% of decisions on Elemental and TLA
  Limited mirrors; 51–95% of those prompts follow a payment that tapped a source
  it did not need (MTG-342). [Results](#results).
- **The hidden choice is common, and small once redundant taps are set aside.**
  Which sources stay untapped differs in 63–73% of completely enumerated casts
  when redundant taps are included. Without them, a payment offers a median of 2
  outcomes, 4–5 at the 95th percentile on ordinary decks and an exact maximum
  of 28 under the optimistic equal-output collapse across all payments. This
  adds 2.9–11.8% under counting rule D (casts alone: 2.5–7.0%), before
  crediting overlap with existing payment decisions (priced below).
  Keeping every source's exact identity instead certifies 655 payments above
  64 offers, including 493 casts. Incomplete enumerations remain explicit.
  [Matched payment-shape results](#matched-payment-shape-results).
- **A small learning pilot**, per-unit prompts against automatic pool spending:
  two seeds per arm, about USD1.4 of rental, no resolved difference.
  [Learning comparison](#learning-comparison).

Three [questions for review](#questions-for-review) decide the size of the menu
and where a player opts out of automation. This report makes no claim about
learning; the comparison of compound and sequential decisions is MTG-265's.

## Questions for review

Nobody has answered these. Each lists its options with the measured price.

### 1. Where does deliberate surplus tapping live?

Tapping a land the cost does not need, to float its mana, is rules-legal and
occasionally matters. It is a different act from paying: a player may tap for
mana whenever they have priority (CR 605.3a). Without it, the payment menu is
the set of minimal payments, the same set MTG-342's fixed tapper picks from.

- **Not offered.** Under the equal-output collapse: exact maximum 28, no payment
  certified above 64 offers, so a flat menu suffices on completely enumerated
  payments. The 554 payments that hit the enumeration cap remain unknown; their
  lower bounds reach 30. A land's fixed output could then never be floated on
  purpose, as today.
- **A tap-for-mana action at priority.** The same payment menu. Activated mana
  abilities are already offered at priority and during payment; a land's fixed
  output is tapped only by the automatic payer, so this extends an existing
  action to fixed sources. Its cost in offers per priority window is not
  measured here.
- **In the payment chooser.** 655 payments certified above 64 offers (422
  casts), an exact maximum of 712 and a cast-only 95th percentile of 49–59 on
  ordinary decks, so groups are required. Decisions per game rise 4.2–16.7%
  over today before any group navigation (casts alone: 4–11%).

### 2. Which sources are the same choice?

- **Exact object identity.** Without redundant taps, 655 payments certified
  above 64 offers (389 completely enumerated, 266 capped), including 493 casts,
  with an exact maximum of 252; groups are required. The excess includes
  combinatorial subsets of identical basic lands.
- **Equal definition and state.** Not measured: the capture holds outputs and
  resource transitions, not complete object state. It lies between the other two
  options and needs a rules argument before implementation.
- **Equal mana output.** The menu above (exact maximum 28). Not a safe rule by
  itself: it merges a land with a mana creature that could also attack or block.

The recommendation preserves exact identity until an equivalence rule is
reviewed, and does not assume that every subset of identical basics needs its
own visible offer.

### 3. Where does the player opt out of automation?

- **At the payment decision (recommended).** One decision at every mana payment
  whose offers differ: counting rule D gives 2.9–11.8% more decisions per game
  without redundant taps and under equal-output collapse (casts alone:
  2.5–7.0%), before existing-decision overlap. Exact identity raises that to
  4.0–15.4%, before group navigation. The plan is visible when
  chosen, because the announcements determining the cost are complete.
- **At cast selection: "cast X" beside "cast X, choosing payment".** For casts,
  no added decision unless used. Cast-only decisions per game lie between the
  [folded column](#matched-payment-shape-results) (305.60, 210.91, 437.26 and
  216.22 for the four ordinary-deck rows; never override) and the cast-only D
  column (always override). A priority window gains at most one offer per cast offer;
  fully measured windows held at most 13 offers, so at most 26. The automatic
  path commits before modes, targets, X, the cost and therefore the plan are
  known, so the plan cannot be inspected when it is chosen.

The cast-selection option is arithmetic on published cast-only columns, not a
measured shape. It leaves ability costs, resolution payments and foretell fees
needing their own control; the recommended decision there adds the non-cast
increments in the [all-payment table](#all-payment-costs). No earlier opt-out
for those payments was measured. Learning effects of either stay with MTG-265.

## Recommended design

**Extend the existing native payment decision, with `PayCost` as the automatic
plan, listed first.** Today `PendingChoice::ManaPayment` and resolution
`Decision::Payment` publish `PayOrNot` for mana costs: activatable mana abilities
beside `PayCost`, which runs the automatic payer. Waterbend has its own taps,
and optional resolution payments may offer `Decline`. The recommendation adds
payment outcomes to that decision; it introduces no second decision kind.
This is a design for review; implementing it is a separate Task, and nothing
here is built. The learning pilot does not test it.

An illustration, with Forest, Forest and Mountain untapped and a {1}{G} spell
whose modes, targets and cost are already announced:

| offer | what it commits |
| --- | --- |
| Automatic: tap Forest and Mountain | leaves a Forest; the spell records {G} and {R} spent |
| Tap Forest and Forest | leaves the Mountain; the spell records {G}{G} spent |

The player picks one offer and the engine pays once. Tapping all three lands is
not on this menu: it is a redundant tap, the subject of question 1. Two offers
also assume question 2 lets the Forests count as one choice. Under exact
identity, which the recommendation keeps until an equivalence rule is reviewed,
there are three: both Forests, or the Mountain with either Forest.

- **When it appears.** At every mana-payment boundary, after the announcements
  determining its cost (including modes, targets and X where applicable), and
  whenever its offers differ, rather than only when a mana ability can be
  activated. Alternative outcomes distinguish what stays untapped, what floats,
  or what the payment receipt records (colors spent, spend effects). They are
  sibling offers beside the activations, Waterbend taps and any `Decline`,
  recomputed whenever the decision returns after an activation, as `PayCost`
  is today.
  This covers casts, ability costs including upfront mana-ability fees,
  resolution payments and foretell fees. A payment with only one legal offer
  skips it, as any single-action decision does; zero-mana costs add no fixed
  sources.
  It replaces today's per-pip prompts at every such boundary.
- **The automatic plan** is an ordinary bound offer, listed first and identified
  as automation, describing the sources it taps and the pool it leaves. It
  presupposes MTG-342's fix: today's payer taps a source it does not need in
  6–15% of ordinary payments and would present a stray float as the plan.
- **No separate entry step.** A step that first asks "automatic or override?"
  costs one more decision on every override and saves none when the player
  accepts the plan (table below). Where the outcomes fit the action capacity
  they are offered flat.
- **When outcomes exceed the action capacity,** the offers after the automatic
  plan are native balanced groups, and only an outcome commits. Card naming is
  the precedent (`Decision::NameCard`, `NAME_BRANCHING`). Naming partitions a
  sorted vocabulary by contiguous span; payment outcomes have no such public
  order, so how outcomes are ordered into groups is for the implementing Task.
  Every permitted outcome stays reachable; an enumerator's work cap never
  truncates the legal action space, and first-legal truncation is not used to
  fit this chooser. Whether groups are needed at all depends on questions 1
  and 2.

**What it costs.** All decisions per game on the measured trajectories, without
redundant taps and treating equal-output sources as one choice. The design
covers all payments; the cast-only column shows what non-casts add. D adds a
choice even where today's trajectory already has a payment decision; its
published counts below retain that double-counting for comparison.

| decks | policy | per unit (today) | D casts only | D all payments | C all payments, separate entry |
| --- | --- | ---: | ---: | ---: | ---: |
| elemental | heuristic | 306.18 | 317.39–317.54 | 319.68–319.84 | 319.68–325.44 |
| elemental | random | 211.36 | 226.10–226.24 | 236.18–236.32 | 236.18–247.71 |
| tla | heuristic | 437.75 | 448.86–449.06 | 450.62–450.82 | 450.62–457.55 |
| tla | random | 217.50 | 227.35–227.36 | 241.61–241.64 | 241.61–252.69 |

Under that optimistic equal-output collapse, D adds 2.9–11.8% to decisions per
game across all payments (casts alone: 2.5–7.0%). Extending the existing
decision saves at most 1.93, 0.35, 4.12 and 1.24 decisions per game in these
four rows; these are upper bounds, not observed savings
([overlap table](#existing-payment-decision-overlap)). Non-cast payments use
the same interaction.
Completely enumerated payments with multiple outcomes offer a median of 2,
a 95th percentile of 4–5 on ordinary decks and an exact maximum of 28 outcomes.
These menu sizes do not bound incomplete enumerations. The entry step's range
runs from always accepting the plan to always overriding. These are counts on
today's trajectories, not games played under the design, an acceptance rate or
a learning cost; learning effects belong to MTG-265's study. The
[all-payment costs](#all-payment-costs) include exact identity and surplus
choices; group-navigation decisions are additional.

**For the policy,** each outcome and group has native action descriptors for
what remains available, the floating pool and the relevant receipt. The
announced spell or ability and the payment context stay available to
observation and action joins throughout the pending choice, and pending-choice
state reaches the policy input, the learning tokens and the LLM text in the
change that adds it. Group expansion goes through
`PaymentModel`, which keeps restrictions, expiry, spend riders and saved
obligations.

**For Academy,** the browser renders the same offers and bound Commands. Labels
can say "leave this Forest untapped", "float {R}" or "spend two colors", with
visible object joins for the distinctions question 2 retains. The automatic plan
is shown as automation. Keyboard access, stale rejection, reconnect to the
actual pending decision and replay/fork identity are preserved. The selected
sources and receipt commit through native payment execution once; group
navigation offers no cancellation or undo of committed costs or revealed
information.

**Not recommended.**

- *Keep today's per-unit prompts (A).* The source choice stays hidden.
- *Automatic only (B).* Removes the pooled-unit prompts and supplies no source
  control.
- *Fold outcomes into the cast action (E).* Even without redundant taps a
  priority window reaches 75 offers, and many costs are not known at that
  boundary.
- *A browser-only override.* It would divide the shared policy/Academy
  contract.

Mana activations remain offers within the extended payment decision. How
outcome groups sit beside activations is for the implementing Task.
`MANAGYM_EXPERIMENT_AUTOSPEND` remains a reproduction control for this
experiment's frozen A/B revision; the follow-up replaces it with the automatic
offer above. No Academy code, default training behavior or deployment changes
here.

## Run it

```sh
uv run python -m experiments.mana_spending count --games 8 --seed 0 --out .runs/mana-spending-count
```

plays mirror games locally on CPU and prints one row per deck set and policy
(six rows, shown in Results).

`--games` is games per deck and policy. `--seed` fixes both deals and policy
sampling; the same revision and arguments reproduce the rows. `--out DIR`
(default `experiments/data/mana_spending/`) must be empty; the
[appendix](#appendix-run-logistics) lists what it writes. Each finished game is
replayed without a policy, and both its digest and diagnostics must agree. The
run rents nothing. Incomplete games are retained, excluded from the summary and
make the command exit nonzero; none is replaced.

| column | meaning |
| --- | --- |
| games / invalid | completed, replay-verified games / excluded incomplete or failed games |
| decisions/game | decisions shown to either player; single-action decisions are skipped as in every game runner |
| spend/game | `mana_spend` decisions, one per pip that offered two or more pooled units |
| share | spend decisions as a fraction of all decisions |
| over-tap | fraction of spend decisions whose payment tapped a source it could have left untapped |
| non-over-tap-share | spend decisions attached to payments without a redundant tap, divided by all original decisions |
| no-spend games | fraction of games with no spend decision |

## What is counted

- **Deck sets.** `elemental`: the ten decks of the Elemental recipe, each
  against itself. `tla`: the first ten 40-card builds, in retained population
  order, from `experiments/regimes/tla-frontier-c5fba4-w64.json` (a population
  derived from the TLA trophy corpus), each against itself. The 41-card builds
  are skipped; lists and source event IDs are recorded. `heartbeat`: one
  deck with Heartbeat of Spring, where every land makes extra mana, as the
  upper end.
- **Policies.** Both seats play the same policy, with separate seat memory and
  random streams. `heuristic` and `random` run on every deck, against the full
  native legal action space. Policies cast and hold mana differently, so their
  rows are never pooled. The existing MTG-183 final checkpoint is checked for
  admission separately; it cannot run under this revision (see Results).
- **Over-tap.** The engine counts, at each payment, whether some source it
  tapped could be removed with the cost still payable from the pool. Mountain,
  Mountain, Forest tapped for {1}{G} is one over-tap; Forest, Mountain is none.
  The counter is reported in `Env.info()` as `mana_payments`,
  `mana_payments_over_tapped` and `mana_spend_decisions_over_tapped`, and
  changes no behavior. Both ordinary payments and upfront mana-ability fees
  are counted, as are resolution payments and foretell fees. Zero-mana payments
  count as payments but cannot produce spend decisions. The retained `casts`
  count is selected priority cast actions, not a count of all payments.
  Source output includes automatic triggered bonuses; saved
  payment obligations must remain feasible. Counters reset with the game and
  clone independently, outside rules hashes and policy observations.

  This is attribution, not an exact post-fix rate. Over-tapped payments may
  retain real choices after a redundant tap is removed, and changing the tapper
  changes later play. The non-over-tapped subset uses the original denominator;
  measuring MTG-342's fix requires a separate matched run. A redundant tap also
  need not produce a prompt: Mountain, Mountain, Forest for {1}{G} offers only
  one color for each pip and is skipped.

## Results

Frequency run, 2026-11-13, source `58e5923c53eb8914734be46471b58fccff974249`,
world `w17`, `count --games 8 --seed 0`: **336 completed games, 336 verified
replays, no failures or replacements**. Local CPU, no launcher run or rental;
USD0 compute rental. Each policy played eight mirror games on each of ten
Elemental decks, ten sampled TLA builds, and the Heartbeat deck. Deals were
matched between policies within each deck. These are exploratory sample counts,
not a strength or learning result.

| decks | policy | games | decisions/game | spend/game | share | over-tap | non-over-tap-share | no-spend games |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| elemental | heuristic | 80 | 306.18 | 0.61 | 0.200% | 75.51% | 0.049% | 88.75% |
| elemental | random | 80 | 211.36 | 0.54 | 0.254% | 51.16% | 0.124% | 86.25% |
| tla | heuristic | 80 | 437.75 | 0.50 | 0.114% | 95.00% | 0.006% | 81.25% |
| tla | random | 80 | 217.50 | 1.30 | 0.598% | 80.77% | 0.115% | 83.75% |
| heartbeat | heuristic | 8 | 87.62 | 12.12 | 13.837% | 54.64% | 6.277% | 0.00% |
| heartbeat | random | 8 | 146.25 | 14.25 | 9.744% | 75.44% | 2.393% | 0.00% |

Here `over-tap` is a fraction of spending decisions, while the adjacent share
columns divide by **all** original decisions. Excluding over-tap-associated
prompts leaves 0.025–0.263 spending decisions/game on the ordinary decks:
Elemental heuristic/random 0.15/0.26, TLA heuristic/random 0.025/0.25.
The non-over-tapped counts are 12, 21, 2 and 20 respectively. These exclusions
describe the observed trajectories, not trajectories under a fixed tapper.

| decks | policy | spend decisions | attributed to over-tap | payment calls | over-tapped payments | offers median / p95 / max |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| elemental | heuristic | 49 | 37 | 2717 | 400 | 2 / 2 / 3 |
| elemental | random | 43 | 22 | 4430 | 538 | 2 / 3 / 5 |
| tla | heuristic | 40 | 38 | 2588 | 213 | 2 / 4 / 4 |
| tla | random | 104 | 84 | 4742 | 288 | 3 / 4 / 4 |
| heartbeat | heuristic | 97 | 53 | 134 | 38 | 2 / 2 / 2 |
| heartbeat | random | 114 | 86 | 164 | 56 | 2 / 2 / 2 |

Spending prompts are uncommon in these ordinary-deck mirrors (0.11–0.60% of
decisions), and most games have none. Many redundant payments produce no
prompt. Heartbeat changes the frequency substantially: every sampled game has
spending prompts, even though each prompt still has only two offers. Multiple
Heartbeat copies can also make some taps redundant, so Heartbeat's prompts
are not all outside the over-tap category.

The MTG-183 final checkpoint could not supply a trained-policy row: the loader
refused it with `UnsupportedPolicyInput` because it uses input v1, while this
revision supports v9/v10. It was not rebound.

Evidence is published under `experiments/data/mana_spending/` in the
[artifact manifest](data.manifest.json): `games.jsonl`, `decks.json`, and
`checkpoint-admission.json`. Restore it with:

```sh
uv run --extra artifacts python -m manabot.data fetch experiments/data/mana_spending
```

## Decision shapes

Replay the retained deals with the bounded native diagnostic:

```sh
uv run python -m experiments.mana_spending shapes --source .runs/mana-spending-count --out .runs/mana-spending-shapes --cap 16384
uv run python -m experiments.mana_spending summary --source .runs/mana-spending-shapes
```

At every actual payment this enumerates fixed-source subsets, then native unit
spending to completion; each final remainder must preserve saved payment
obligations under `PaymentModel`. Zero-mana payments keep the existing pool
and add no fixed sources. Activated abilities remain explicit native decisions
outside this automatic boundary: their presence is recorded, their strategies
are not enumerated.

Each payment is counted four ways on one traversal:

- **Redundant taps included or not.** The inclusive count allows deliberately
  producing surplus mana. The non-redundant count keeps only source sets where
  removing any selected source's whole output leaves the cost unpayable from
  the pool or a saved caller infeasible on the post-tap board. This is the
  over-tap counter's criterion, not a ban on floating mana: indivisible surplus
  (Heartbeat) and pre-existing pool choices remain.
- **Capability or identity.** Capability treats sources with equal output as
  one choice; identity keeps exact objects. The collapse is a measurement
  definition, not permission to hide object choices.

Separate columns count untapped-source capabilities, floating pools, native
receipts and their joint outcomes. Receipts keep all six spent amounts and
entry counters, which distinguishes more than a particular spell may read.

The per-payment cap counts source-subset and receipt traversal visits. A capped
payment keeps lower bounds for every count, including a zero non-redundant
count, and never enters exact offer quantiles.

For folded offers, every legal cast at an observed priority window is tried on
an exact clone. A probe that reaches exactly one immediate cast payment
supplies its outcome count; any other (targets, modes or costs announced
first, a payment bypass, several payment steps) is unresolved. Windows with an
unresolved or capped cast are excluded from exact folded-offer quantiles;
their lower-bound counts remain in evidence. Neither enumeration changes the
retained trajectory; every final digest must match the frequency run.

| shape | what the player decides |
| --- | --- |
| A, per unit | which pooled unit pays each pip, after automatic tapping (today) |
| B, automatic | nothing; the first feasible unit pays |
| C, entry step | a separate automatic-or-override step when outcomes differ, then an outcome choice only for a nontrivial override; measured as the price of that step |
| D, outcome | one choice among distinct outcomes: what stays untapped or floating, and what the spell's receipt records (colors spent, spend effects); the recommended decision, whose first offer is the automatic plan |
| E, folded | the outcome is part of the cast action ("cast X, leaving Forest") |

The comparison reports decisions per game, offers per decision (median, 95th
percentile, maximum), real choices and cap hits. A and B count all spending
prompts; the C–E columns below replace cast-payment prompts and retain
noncast payment behavior. [All-payment costs](#all-payment-costs) applies C and
D to every mana payment, as the recommendation requires.
C's decision cost depends on how often the player overrides, so it is a range
from always accepting the plan (equal to D) to always overriding. E's decision
count assumes the other announcement steps stay as they are. These are
counterfactual counts on A's trajectories, not games under C–E.

## Matched payment-shape results

2026-11-13, source `0d2233a567`, rebuilt native SHA256
`acd7fbfa00c522afda010a3cec5e533a668173874aee2f95416defa2ad6b81dc`.
The engine/card merge after the original frequency run changes final state
digests. A fresh `count --games 8 --seed 0` under this build completed and
replay-verified all 336 games. All 21 deck lists, every selected action index,
stage count, spend offer and payment counter match the original evidence;
only final digests differ. Original evidence remains unchanged. `shapes` then
replayed this fresh baseline: **336 verified games, 14,775 payments and 9,089
cast payments**, with no digest, payment-count or spend-offer disagreement.
All exact actual payments had at least one enumerated outcome. Local CPU, no rental.

“Real” below means distinct formal outcomes, including surplus production; it
does not assert that every outcome is strategically useful. Source and receipt
percentages use only completely enumerated cast payments. Cap hits keep certified
lower bounds and are included in the adjacent total denominator.

| decks | policy | cast payments | capped | ≥2 joint outcomes, certified | ≥2 untapped-source outcomes, exact | ≥2 receipt variants, exact | identity exceeds capability, exact (upper bound) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| elemental | heuristic | 2062 | 99 | 1527/2062 (74.1%) | 1425/1963 (72.6%) | 874/1963 (44.5%) | 1380/1963 (70.3%) |
| elemental | random | 2589 | 120 | 1932/2589 (74.6%) | 1805/2469 (73.1%) | 1132/2469 (45.8%) | 1708/2469 (69.2%) |
| tla | heuristic | 2027 | 111 | 1326/2027 (65.4%) | 1215/1916 (63.4%) | 846/1916 (44.2%) | 1172/1916 (61.2%) |
| tla | random | 2113 | 38 | 1346/2113 (63.7%) | 1305/2075 (62.9%) | 884/2075 (42.6%) | 1236/2075 (59.6%) |
| heartbeat | heuristic | 134 | 16 | 107/134 (79.9%) | 91/118 (77.1%) | 67/118 (56.8%) | 82/118 (69.5%) |
| heartbeat | random | 164 | 36 | 143/164 (87.2%) | 107/128 (83.6%) | 73/128 (57.0%) | 93/128 (72.7%) |

With redundant tapping allowed, 63–73% of completely enumerated ordinary casts leave
different fixed-source capabilities available. This contrasts with 0.11–0.60% of all
decisions being today’s pool prompts. Equal-output producers also conceal different
exact objects in 60–70% of exact ordinary casts. This is an upper bound on strategically
meaningful loss: it includes two otherwise identical basic Forests as well as a land and
a mana creature. PaymentModel's capture contains resource transitions, not complete
definitions and object state, so it cannot cheaply separate those cases. Output
capability alone is not an approved equivalence rule. Activated sources were present at
212/18 Elemental and 492/53 TLA cast payments (heuristic/random); their activation
strategies are outside these fixed-source counts. None of the observed cast payments
retained a saved caller obligation.

Counterfactual **all decisions per game** on these A trajectories. All capped
cast rows already certify ≥2 outcomes, so D’s decision count is known even
though some offer counts are incomplete. Grouping D would add decisions beyond
this flat, one-decision estimate.

| decks | policy | A per unit | B automatic | C automatic / override | D flat outcome | E folded |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| elemental | heuristic | 306.18 | 305.56 | 324.69–341.57 | 324.69 | 305.60 |
| elemental | random | 211.36 | 210.82 | 235.06–256.07 | 235.06 | 210.91 |
| tla | heuristic | 437.75 | 437.25 | 453.84–468.94 | 453.84 | 437.26 |
| tla | random | 217.50 | 216.20 | 233.05–247.65 | 233.05 | 216.22 |
| heartbeat | heuristic | 87.62 | 75.50 | 88.88–101.25 | 88.88 | 75.50 |
| heartbeat | random | 146.25 | 132.00 | 149.88–166.38 | 149.88 | 132.00 |

Payment **offers median / p95 / maximum**, conditional on complete enumeration
and at least two outcomes. Empirical quantiles use the higher order statistic.
A uses actual prompts. C is the always-override path
(entry offers plus its nontrivial outcome offers). B has no payment offers.

| decks | policy | A | C | D | D casts certified above 64 |
| --- | --- | ---: | ---: | ---: | ---: |
| elemental | heuristic | 2 / 2 / 3 | 2 / 39 / 203 | 10 / 49 / 204 | 82 |
| elemental | random | 2 / 3 / 5 | 2 / 42 / 192 | 10 / 52 / 193 | 79 |
| tla | heuristic | 2 / 4 / 4 | 2 / 45 / 711 | 10 / 59 / 712 | 91 |
| tla | random | 3 / 4 / 4 | 2 / 42 / 631 | 7 / 58 / 632 | 84 |
| heartbeat | heuristic | 2 / 2 / 2 | 2 / 92 / 167 | 19 / 132 / 168 | 37 |
| heartbeat | random | 2 / 2 / 2 | 2 / 77 / 158 | 19 / 105 / 159 | 49 |

Flat D reaches **712** capability outcomes on a completely enumerated cast.
Heartbeat’s conditional p95 is 105–132; ordinary-deck conditional p95 is 49–59.
Because 420/9,089 cast enumerations hit the cap, these conditional quantiles do
not bound the full population. 422 cast payments are certified above 64
outcomes, including capped rows: with deliberate redundant tapping included, a
64-slot policy cannot use flat D everywhere, even under the optimistic
capability collapse. The [paired measurement](#paired-counts-without-redundant-taps)
gives the capacity needed without redundant tapping.

E **full priority offers median / p95 / maximum**, only where every cast at
that priority window reached an immediate, completely enumerated payment:

| decks | policy | complete / observed windows | original offers | E offers | unresolved cast offers | capped cast offers |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| elemental | heuristic | 4575/13509 | 3 / 6 / 10 | 11 / 66 / 408 | 12731 | 227 |
| elemental | random | 3343/5385 | 3 / 6 / 10 | 11 / 66 / 545 | 2180 | 228 |
| tla | heuristic | 1514/9413 | 3 / 7 / 13 | 6 / 55 / 459 | 10346 | 127 |
| tla | random | 2626/4068 | 4 / 8 / 13 | 8 / 112 / 1552 | 1670 | 159 |
| heartbeat | heuristic | 183/199 | 3 / 7 / 9 | 20 / 99 / 169 | 0 | 16 |
| heartbeat | random | 262/356 | 3 / 7 / 8 | 32 / 136 / 242 | 0 | 112 |

E reaches 1,552 offers even on the fully measured subset. Many ordinary-deck
casts require targets or other announcements before their cost is payable;
flattening those into priority would require another announcement design.
The unresolved counts are an explicit measurement limit, not zero outcomes.
E’s offer quantiles do not describe all priority windows. Windows here have at
least one cast offer; other priority windows retain their baseline offers.

Evidence: `current/decks.json`, `current/games.jsonl`, and
`current/shapes/{metadata.json,shapes.jsonl,summary.json}` under the experiment’s
artifact directory.

### Paired counts without redundant taps

2026-11-13, source `d18004357a`, rebuilt native SHA256
`11f90b28c1fdf023cf842216b32f3426ef6b4851b1d5abc86fb46f6bf47c59af`. The paired run
replayed the same **336 games, 14,775 payments and 9,089 cast payments** at the same
16,384-visit cap, finishing in 421.48 seconds locally (USD0 rental). Every final digest,
payment counter and observed spend offer matched the current baseline. A field-by-field
comparison of all 336 rows also found **every original payment and folded-offer field
unchanged**, including surplus-inclusive counts, work and cap hits. All fully enumerated
payments have at least one non-redundant outcome.

Non-redundant, capability and identity are defined under
[Decision shapes](#decision-shapes). None of the sampled cast payments has
saved obligations; the saved-caller criterion is covered by a directed native
fixture. Neither source key measures an approved definition/state equivalence.

D offers below are median / p95 / maximum on fully enumerated casts with at least two
outcomes **under the column's own definition**; the conditional samples therefore
differ. Above-64 counts are certified lower bounds over all casts, including capped
rows. All definitions share the same cap-hit flag: **420/9,089 casts remain
incomplete**, even when a non-redundant lower bound is small.

| decks | policy | source key | D including redundant taps | D non-redundant | casts >64, including / non-redundant | capped casts |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| elemental | heuristic | capability | 10 / 49 / 204 | 2 / 5 / 9 | 82 / 0 | 99 |
| elemental | heuristic | identity | 37 / 1793 / 5825 | 6 / 64 / 252 | 648 / 111 | 99 |
| elemental | random | capability | 10 / 52 / 193 | 2 / 4 / 7 | 79 / 0 | 120 |
| elemental | random | identity | 37 / 2016 / 4064 | 7 / 56 / 252 | 873 / 119 | 120 |
| tla | heuristic | capability | 10 / 59 / 712 | 2 / 5 / 22 | 91 / 0 | 111 |
| tla | heuristic | identity | 37 / 1649 / 4412 | 9 / 84 / 252 | 594 / 163 | 111 |
| tla | random | capability | 7 / 58 / 632 | 2 / 7 / 19 | 84 / 0 | 38 |
| tla | random | identity | 23 / 1136 / 4004 | 8 / 70 / 252 | 475 / 98 | 38 |
| heartbeat | heuristic | capability | 19 / 132 / 168 | 2 / 6 / 6 | 37 / 0 | 16 |
| heartbeat | heuristic | identity | 91 / 3539 / 3968 | 6 / 36 / 78 | 65 / 2 | 16 |
| heartbeat | random | capability | 19 / 105 / 159 | 2 / 5 / 6 | 49 / 0 | 36 |
| heartbeat | random | identity | 97 / 2898 / 4032 | 5 / 27 / 49 | 91 / 0 | 36 |

Non-redundant capability D has an **exact maximum of 22**, with **zero** casts certified
above 64. Capped rows reach observed lower bounds of 30, not complete counts. Grouping
is **not implied by these measured capability outcomes**. The 420 incomplete casts
prevent a claim that all such outcomes fit 64. Retaining exact identity instead
certifies **493** non-redundant casts above 64 (291 completely enumerated, 202 capped);
its exact maximum is 252. That includes combinatorial subsets of identical lands. It
establishes a capacity cost of exact identity, not the strategic value of every
distinction.

For E, quantiles use the same fully measured priority windows as the original E table;
unresolved and capped probes remain excluded under both definitions. Above-64 counts
here count **complete priority windows**, not cast payments. The last column counts
capped cast probes in all observed windows; unresolved probe counts and complete-window
denominators are unchanged from the table above.

| decks | policy | source key | E including redundant taps | E non-redundant | complete windows >64, including / non-redundant | capped cast probes |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| elemental | heuristic | capability | 11 / 66 / 408 | 3 / 8 / 21 | 240 / 0 | 227 |
| elemental | heuristic | identity | 27 / 1409 / 7827 | 7 / 56 / 196 | 1675 / 215 | 227 |
| elemental | random | capability | 11 / 66 / 545 | 4 / 9 / 23 | 176 / 0 | 228 |
| elemental | random | identity | 22 / 2125 / 14579 | 7 / 81 / 475 | 1253 / 218 | 228 |
| tla | heuristic | capability | 6 / 55 / 459 | 4 / 10 / 25 | 54 / 0 | 127 |
| tla | heuristic | identity | 8 / 888 / 7145 | 6 / 72 / 272 | 350 / 87 | 127 |
| tla | random | capability | 8 / 112 / 1552 | 5 / 15 / 75 | 254 / 1 | 159 |
| tla | random | identity | 11 / 1265 / 10622 | 7 / 117 / 552 | 741 / 278 | 159 |
| heartbeat | heuristic | capability | 20 / 99 / 169 | 4 / 10 / 18 | 28 / 0 | 16 |
| heartbeat | heuristic | identity | 48 / 2739 / 3969 | 8 / 37 / 89 | 82 / 4 | 16 |
| heartbeat | random | capability | 32 / 136 / 242 | 4 / 11 / 22 | 72 / 0 | 112 |
| heartbeat | random | identity | 93 / 3662 / 7845 | 8 / 41 / 90 | 143 / 7 | 112 |

Even with non-redundant capability collapse, E reaches **75** in one complete TLA random
window. Exact identity reaches 552. These measurements do not resolve announcements at
the many unresolved windows; folding remains a separate design problem even where its
menu fits.

The corresponding **all decisions/game** without redundant tapping are below. D's
interval reflects incomplete enumeration; C's interval also spans accepting automatic
versus taking every nontrivial override. No group-navigation decisions are included.
Exact identity and capability are alternative measurement keys, not two successive
menus. Noncast payments retain their observed behavior.

| decks | policy | D capability | C capability | D identity | C identity |
| --- | --- | ---: | ---: | ---: | ---: |
| elemental | heuristic | 317.39–317.54 | 317.39–321.69 | 323.51 | 323.51–339.54 |
| elemental | random | 226.10–226.24 | 226.10–231.07 | 233.09 | 233.09–252.71 |
| tla | heuristic | 448.86–449.06 | 448.86–454.96 | 453.09 | 453.09–467.96 |
| tla | random | 227.35–227.36 | 227.35–232.46 | 231.93 | 231.93–245.96 |
| heartbeat | heuristic | 83.00–84.25 | 83.00–89.12 | 87.25 | 87.25–98.12 |
| heartbeat | random | 139.12–142.88 | 139.12–149.50 | 147.88–148.00 | 147.88–161.62 |

Paired evidence is published separately as
`current/non-redundant-shapes/{metadata.json,shapes.jsonl,summary.json}` in the artifact
manifest. Original frequency, `current/` baseline and `current/shapes/` bytes are
preserved. Restore it with the same artifact fetch command above.

### All-payment costs

The recommendation covers all **14,775** payment rows, including **5,686**
non-casts: 2,496 Elemental, 3,190 TLA and none in Heartbeat. Of those non-casts,
3,774 have positive mana cost (1,544 Elemental / 2,230 TLA), with 134 capped
(68 / 66). Zero-mana rows remain in the population and add no source choices.
The published capture marks cast versus non-cast, not the non-cast rules reason;
it cannot separate ability costs, resolution payments and foretell fees here.

This is a read-only aggregation of the paired rows through the same summary
path, not another game run. Reproduce it after fetching the published evidence:

```sh
uv run python -m experiments.mana_spending summary --source experiments/data/mana_spending/current/non-redundant-shapes --all-payments
```

This writes `all-payments-summary.json` beside the source, leaving the published
`shapes.jsonl` and `summary.json` unchanged. It repeats the cast-only summary
and adds `payment_scopes.noncast` and `payment_scopes.all` under the same four
counting rules. The file is derived locally and not published; the tables below
hold its results.

For any selected payment population, let S be its observed per-unit prompts,
N its certified payments with at least two outcomes, and U those with either
at least two outcomes or an incomplete enumeration. With T original decisions
and G games, D is `(T - S + [N, U]) / G`. C has the same lower bound; its upper
bound adds one decision per payment with more than two outcomes or an incomplete
enumeration. Every population retains all other observed decisions, so a
non-cast increment is its D count minus A and adds to the cast-only D count.
Capped rows are treated as in the cast summary.

Non-cast counts below keep each policy separate. Exact choices and offer
quantiles exclude capped rows and use non-redundant outcomes. Prompted payments
count boundaries; prompts count individual per-unit decisions.

| decks | policy | non-cast payments | positive cost | capped | prompted payments / prompts | exact ≥2 capability / identity | capability offers median / p95 / max | identity offers median / p95 / max |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| elemental | heuristic | 655 | 336 | 6 | 3 / 3 | 181 / 206 | 3 / 3 / 6 | 10 / 36 / 45 |
| elemental | random | 1841 | 1208 | 62 | 6 / 7 | 752 / 811 | 3 / 4 / 28 | 10 / 70 / 252 |
| tla | heuristic | 561 | 269 | 3 | 1 / 1 | 139 / 166 | 2 / 4 / 16 | 6 / 35 / 126 |
| tla | random | 2629 | 1961 | 63 | 1 / 2 | 1081 / 1286 | 2 / 4 / 22 | 6 / 55 / 252 |
| heartbeat | heuristic | 0 | 0 | 0 | 0 / 0 | 0 / 0 | — | — |
| heartbeat | random | 0 | 0 | 0 | 0 / 0 | 0 / 0 | — | — |

Of completely enumerated positive-cost non-casts, 933/1476 Elemental and
1220/2164 TLA payments have multiple capability outcomes (63.2% / 56.4%);
identity gives 1017/1476 and 1452/2164 (68.9% / 67.1%). The hidden source choice
is common here too, despite only 9 Elemental and 2 TLA non-cast payments drawing
prompts today. Those boundaries produced 10 and 3 per-unit decisions respectively.

**Non-redundant all-payment decision costs**, before group navigation. The
non-cast increment uses capability: it is what the cast-only D columns leave
out. The first four D and C capability rows also appear in the recommendation's
cost table.

| decks | policy | non-cast D increment | D all capability | C all capability | D all identity | C all identity |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| elemental | heuristic | +2.29–2.30 | 319.68–319.84 | 319.68–325.44 | 326.11–326.12 | 326.11–344.65 |
| elemental | random | +10.08–10.09 | 236.18–236.32 | 236.18–247.71 | 243.90–243.91 | 243.90–274.01 |
| tla | heuristic | +1.76 | 450.62–450.82 | 450.62–457.55 | 455.19 | 455.19–472.07 |
| tla | random | +14.26–14.28 | 241.61–241.64 | 241.61–252.69 | 248.75–248.76 | 248.75–278.48 |
| heartbeat | heuristic | +0.00 | 83.00–84.25 | 83.00–89.12 | 87.25 | 87.25–98.12 |
| heartbeat | random | +0.00 | 139.12–142.88 | 139.12–149.50 | 147.88–148.00 | 147.88–161.62 |

Relative to A, ordinary-deck all-payment D grows 2.9–11.8% with capability
collapse, or 4.0–15.4% with exact identity. The latter also requires group
navigation. These figures do not imply that every identity distinction matters.

#### Existing payment-decision overlap

D adds one decision per multi-outcome payment while keeping every observed
`PayOrNot` decision, so a payment that already had one is counted twice. The
capture's `activated_sources` counts sources with abilities still present at
commit, not whether an activation was offered, so it bounds the saving from
above. Overlap with Waterbend or `Decline` decisions is not priced.

`summary --all-payments` derives `existing_decision_overlap` for each scope and
counting rule; the table uses all payments and non-redundant capability
outcomes. Overlap counts payments with an activated source and at least two
outcomes (certified) or, in addition, a capped enumeration (possible). The
maximum saving is possible / games. The merged lower bound is D's lower bound
minus certified / games, since a capped row's uncertain decision and its
overlap cancel; the upper bound is D's, because no overlap is proved. Group
navigation and the offers added beside activations are not counted.

| decks | policy | games | overlap certified / possible | maximum saving/game | merged decisions/game | D all payments |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| elemental | heuristic | 80 | 149 / 154 | 1.93 | 317.81–319.84 | 319.68–319.84 |
| elemental | random | 80 | 28 / 28 | 0.35 | 235.83–236.32 | 236.18–236.32 |
| tla | heuristic | 80 | 320 / 330 | 4.12 | 446.62–450.82 | 450.62–450.82 |
| tla | random | 80 | 99 / 99 | 1.24 | 240.38–241.64 | 241.61–241.64 |
| heartbeat | heuristic | 8 | 0 / 0 | 0.00 | 83.00–84.25 | 83.00–84.25 |
| heartbeat | random | 8 | 0 / 0 | 0.00 | 139.12–142.88 | 139.12–142.88 |

**All-payment capacity and surplus price.** Non-redundant capability menus
have median / p95 / maximum below, conditional on complete enumeration and
multiple outcomes. Identity's exact maximum remains 252. Surplus-inclusive D
uses capability and is shown alongside its increase over A for the ordinary
decks. Counts above 64 include capped rows; none of the non-redundant capability
counts certifies more than 64. Capped rows' non-redundant lower bounds reach 30.

| decks | policy | all payments / capped | non-redundant capability offers | non-redundant identity >64, exact / capped | surplus capability >64 | surplus D all capability | surplus D increase over A |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| elemental | heuristic | 2717 / 105 | 2 / 4 / 9 | 61 / 50 | 109 | 327.73 | 7.0% |
| elemental | random | 4430 / 182 | 2 / 4 / 28 | 115 / 74 | 189 | 246.69 | 16.7% |
| tla | heuristic | 2588 / 114 | 2 / 5 / 22 | 90 / 78 | 97 | 456.11 | 4.2% |
| tla | random | 4742 / 101 | 2 / 5 / 22 | 122 / 63 | 174 | 250.57–250.59 | 15.2% |
| heartbeat | heuristic | 134 / 16 | 2 / 6 / 6 | 1 / 1 | 37 | 88.88 | — |
| heartbeat | random | 164 / 36 | 2 / 5 / 6 | 0 / 0 | 49 | 149.88 | — |

Across all payments, 554 enumerations remain incomplete. Keeping exact identity
certifies 655 non-redundant payments above 64 (389 exact / 266 capped), compared
with 493 casts alone. Including surplus also certifies 655 capability payments
above 64, compared with 422 casts alone; its exact maximum remains 712.
Questions 1 and 2 therefore still decide whether groups are needed.

## Learning comparison

This exploratory pilot compares today's per-unit decisions (A) with first-feasible
automatic pool spending (B), on the existing Elemental recipe: seeds 34701 and
34702 per arm, 32 matched updates. Training is frozen at `00ee188998cb`.
`MANAGYM_EXPERIMENT_AUTOSPEND` is explicitly 0 for A and 1 for B; source
selection stays the same.

Evaluation retains each arm's spending mode on the shared 50-pair Elemental
heuristic panel: 200 replay-checked games per final checkpoint, with native
seats and decks swapped across four legs per pair.

The pilot asks only whether removing pool-spending decisions visibly changes
early scores or decisions per game on this recipe. It does not expose
fixed-source choice or implement C–E. Updates are matched; completed games and
learner transitions need not be. The training decision/game ratio divides all
recorded environment decisions by completed games, including decisions from
unfinished or failed episodes; it is not mean completed-game length, and
trajectories differ once the policies differ, so the ratios cannot isolate
saved prompts.

| arm | seed | training run | updates | completed games | environment decisions | learner transitions | decisions/completed game |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| A | 34701 | `8a03411493cd` | 32 | 8,593 | 1,414,495 | 1,413,716 | 164.61 |
| A | 34702 | `b4dbac29bda4` | 32 | 8,573 | 1,388,371 | 1,388,371 | 161.95 |
| B | 34701 | `56c051d5616a` | 32 | 8,526 | 1,544,456 | 1,544,320 | 181.15 |
| B | 34702 | `773f339dd238` | 32 | 8,570 | 1,399,363 | 1,399,363 | 163.29 |

All four jobs completed. A/34701 recovered two resource-limit game failures and
discarded 779 samples; B/34701 recovered one and discarded 136. Both 34702 runs
had none. No games were abandoned. These are the native training receipts'
counts, retained alongside their full error summaries; failed attempts are not
silently counted as completed games. The higher B decision ratios are an
observation of different early trajectories, not evidence that skipping prompts
increases game length or compute cost.

Evaluations `mana-spending-A-303f5f0a6a59` and
`mana-spending-B-fe362b5c294d` each completed all 100 matches / 400 games.
**All 800 games were valid and replay-checked; no dropped, incomplete or failed
matches.** No evaluation game was replaced. Each cell below is 200 games;
there were no draws.

| training seed | A wins / games | A score | B wins / games | B score | B − A |
| --- | ---: | ---: | ---: | ---: | ---: |
| 34701 | 52 / 200 | 26.0% | 54 / 200 | 27.0% | +1.0 pp |
| 34702 | 57 / 200 | 28.5% | 65 / 200 | 32.5% | +4.0 pp |

B scored higher on these two early checkpoints. Two training seeds on a recipe
with roughly 0.2% pool prompts establish neither an improvement nor
equivalence, and say nothing about compound decisions or the recommended
payment decision; MTG-265 owns the learning-effect study. Retain A's defaults
for now and take the source-control design to review.

Rental cost about USD1.4 by the launcher's numbers, under the USD25
authorization. Launch history, cost receipts and warning-log notes are in the
[appendix](#appendix-run-logistics).

## Prior art checked

Reviewed 2026-11-13, through primary documentation and static source:

- **Arena.** Wizards documents auto-tap and full-control settings, and its
  Brothers’ War note describes pausing to preview the planned mana for spells
  whose bonuses depend on mana spent. Sources: [mobile FAQ](https://magic.wizards.com/en/news/mtg-arena/mtg-arena-mobile-faqs-2021-01-28),
  [auto-tap flexibility](https://magic.wizards.com/en/news/mtg-arena/mtg-arena-state-game-the-brothers-war).
- **Forge.** Its guide documents an Auto button using the AI payer, manual
  selection of sources and pooled symbols, and retained prompts for
  color-sensitive spells such as Sunburst and Firespout. Source:
  [Forge user guide, Auto-Pay](https://github.com/Card-Forge/forge/wiki/user-guide#auto-pay).
- **phase-rs.** At v0.123.0, the flat `legal_actions` list omits mana actions;
  `legal_actions_full` preserves engine-authoritative per-object mana actions
  for board interaction. Source: [native action projections](https://github.com/phase-rs/phase/blob/v0.123.0/crates/engine/src/ai_support/mod.rs#L2380-L2386),
  [grouped projection](https://github.com/phase-rs/phase/blob/v0.123.0/crates/engine/src/ai_support/mod.rs#L2536-L2543).

The design inference is to keep automation inspectable and source control
explicit, with grouping owned by the engine. Academy’s shared contract requires
all choices to remain reachable by both policy and browser. These references
provide interaction precedents; the capacity argument comes from this report’s
measurements.

## Limits

- Mirror games under scripted or early policies show how often the decision
  arises, not how much a strong player gains from it.
- The shape tables price casts only; [all-payment costs](#all-payment-costs)
  prices non-casts from the same published rows, without their rules reason.
  No earlier opt-out for non-cast payments is measured. All D counts exclude
  group-navigation steps.
- The shape comparison counts offers. It does not measure play or learning
  under a shape that is not implemented. Manual tapping at every priority
  window is a qualitative alternative, without a measured offer count here.
- Ten TLA builds in retained file order are a bounded convenience sample, not
  a representative sample of the entire Limited corpus. The eight Heartbeat
  games per policy are a constructed upper-end example, not a metagame rate.
- No fixed-tapper run or trained-policy frequency run was performed. Native
  instrumentation attributes redundant output; it does not simulate removing
  a tap's events, other triggers or downstream strategic effects.
- Both pilot evaluation logs carry undiagnosed missing-focus encoder warnings
  ([appendix](#appendix-run-logistics)). Valid replay-checked games do not
  establish that every action feature was informative.

## Appendix: run logistics

<details>
<summary>Output files, checkpoint admission, launch history, cost receipts and warning logs</summary>

**Frequency run output.** `count --out DIR` writes one row per game as
`games.jsonl`, with decisions by stage, spend offer counts, casts, payment
counters, action indices, terminal state digest and failures. `decks.json`
records exact lists, source and native hashes, revision, world and seeds.

**Checkpoint admission.** The MTG-183 final checkpoint was downloaded and
checked separately. Its SHA256 is
`4f2ad656ba036a33d527212a7ac2badf2c5b9db136916f730846b07eef2fbea4`; the refusal
is recorded in `checkpoint-admission.json`.

**Pilot caps.** Each training job allows USD3 and one attempt; the serial cohort
`mtg347-mana-learning-serial` launched under `--max-usd 17`. Four training caps
plus two USD4 evaluation caps total USD20, within the USD25 authorization. These
caps are limits, not measured costs.

**Pilot evaluation sources.** `4f1fdb2da3c9` (A) and `a29783f42e36` (B); their
difference is report/scratch text only. Both include the diagnostic constructor
cleanup after training; it changes neither the frozen training input layout nor
payment behavior. Model rows join through Etude's participant seat, rather than
treating the native game seat as player identity. The published summary retains
the aggregation's SQL, query plan and timing.

**Launch history.** An initial parallel launch (`518a4948ea80`) failed
admission when a second L4 was unavailable; none of its four job commands
started. The acquired machine was deleted before the serial retry. The source
and question stayed fixed.

**Warning logs.** Both evaluation logs include `Invalid focus object ID 81 for
action 12` (ChooseMode), among similar warnings. The existing Python encoder
emits this when a focus object has no observation join and encodes -1. The
cause was not diagnosed here; the pilot measures the existing representation
and procedure together, and changing the encoder mid-pilot would have changed
its question. These warnings are not game failures.

**Launcher cost** observations after all acquired rentals ended (2026-11-13):

| use | rental | launcher cost |
| --- | --- | ---: |
| failed parallel admission | `mtg347-l4-71b326e7fb82` | USD0.0013 billed |
| all four training runs | `mtg347-l4-b0805de6f0f4` | USD0.8642 billed |
| A evaluation | `mtg347-cpu32-499f433305cc` | ~USD0.26 |
| B evaluation | `mtg347-cpu32-d193458ef8af` | ~USD0.29 |

These are the launcher's retained rental observations, recorded once; `~` is
its estimate, not a reconstructed bill. The other initial L4 placeholder was
never acquired. All six successful jobs finished and all four acquired rentals
were confirmed ended. The authorized USD25 cap was not approached.

Pilot evidence is under `experiments/data/mana_spending/learning/`: four native
training receipts, the original evaluation blocks and worker receipts, selected
warning log excerpts, frozen command-mode/source provenance, the training and
score summaries, and `launcher-costs.txt`. The artifact manifest publishes these
bytes; the original checkpoints and full replay traces remain in Etude's R2
records under the run IDs above.

</details>
