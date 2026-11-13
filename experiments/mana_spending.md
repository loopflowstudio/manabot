# How often players choose which mana to spend — MTG-347

Exploratory. A cast taps its lands automatically; when the pool then holds more
usable mana than the cost needs, the player chooses which pooled unit pays each
pip (`mana_spend` decisions). This report counts those decisions on the decks
people train on, separates the ones caused by automatic tapping taking more
sources than the cost needs (MTG-342) from other payments. This experiment
measures frequency, bounded replacement shapes and a small exploratory learning
pilot. It makes no claim about learning; the comparison of compound
and sequential decisions is MTG-265's.

## Run it

```sh
uv run python -m experiments.mana_spending count --games 8 --seed 0 --out .runs/mana-spending-count
```

plays mirror games locally on CPU and prints one row per deck set and policy
(six rows, shown in Results).

`--games` is games per deck and policy. `--seed` fixes both deals and policy
sampling; the same revision and arguments reproduce the rows. `--out DIR`
(default `experiments/data/mana_spending/`) must be empty. It writes one row per
game as `games.jsonl`, with decisions by stage, spend offer counts, casts,
payment counters, action indices, terminal state digest and failures. Each
finished game is replayed without a policy; both its digest and diagnostics must
agree. `decks.json` records exact lists, source and native hashes,
revision, world and seeds. The run rents nothing. Incomplete games are retained,
excluded from the summary and make the command exit nonzero; none is replaced.

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

## Decision shapes

Replay the retained deals with the bounded native diagnostic:

```sh
uv run python -m experiments.mana_spending shapes --source .runs/mana-spending-count --out .runs/mana-spending-shapes --cap 16384
uv run python -m experiments.mana_spending summary --source .runs/mana-spending-shapes
```

This enumerates fixed-source subsets at every actual payment, then native unit
spending to completion. Each final remainder must preserve saved payment
obligations under `PaymentModel`. It includes deliberately producing surplus
mana. Activated abilities remain explicit native decisions outside this automatic
boundary; their presence is recorded, and these are not counts of every possible
mana-ability strategy.

Separate columns count untapped-source capabilities, floating pools, native
receipts and their joint outcomes. Receipts retain all six spent amounts and
entry counters; this conservatively distinguishes more than a particular spell
may read. Distinct remainders are retained regardless of receipt relevance. Exact source identity counts expose the
additional distinctions lost by grouping producers with equal output. Such a
grouping is a measurement definition, not permission to hide object choices.
Zero-mana payments retain the existing pool without adding fixed sources.
The per-payment cap counts source-subset and receipt traversal visits. A cap hit
retains lower bounds; incomplete rows never enter exact offer quantiles.

For folded offers, every legal cast at an observed priority window is tried on
an exact clone. Casts that reach one payment immediately supply their outcome
count; casts that first require other announcements (targets, modes or costs)
are marked unresolved. Any probe that does not reach exactly one immediate
cast payment is unresolved, including payment bypasses or multiple payment
steps. Windows with an unresolved or capped cast are excluded
from exact folded-offer quantiles. Their lower-bound counts remain in evidence.
Neither enumeration changes the retained trajectory; every final digest must
match the frequency run.

| shape | what the player decides |
| --- | --- |
| per unit | which pooled unit pays each pip, after automatic tapping (today) |
| automatic | nothing; the first feasible unit pays |
| override | nothing, unless payments differ in outcome; then one choice per cast |
| outcome | one choice among distinct outcomes: what stays untapped or floating, and what the spell's receipt records (colors spent, spend effects) |
| folded | the outcome is part of the cast action ("cast X, leaving Forest") |

The comparison reports decisions per game, offers per decision (median, 95th
percentile, maximum), real choices and cap hits. A and B count all spending
prompts; C–E replace cast-payment prompts and retain noncast payment behavior.
C offers automatic payment or override, with a second outcome decision only if
the override has multiple alternatives. Its decision cost depends on how often
the player overrides, so it is reported as a range. D offers one joint outcome
decision when outcomes differ. E folds those cast outcomes into priority offers;
its decision count assumes the other announcement steps stay as they are.
These are counterfactual counts on A's trajectories, not games under C–E.

## Learning comparison

This exploratory pilot compares today's per-unit decisions (A) with first-feasible
automatic pool spending (B), on the existing Elemental recipe: seeds 34701 and
34702 per arm, 32 matched updates. Training is frozen at `00ee188998cb`.
`MANAGYM_EXPERIMENT_AUTOSPEND` is explicitly 0 for A and 1 for B; source
selection stays the same. Each training job allows USD3 and one attempt; the
serial cohort `mtg347-mana-learning-serial` launched under `--max-usd 17`.
Four training caps plus two USD4 evaluation caps total USD20, within the USD25
authorization. These caps are limits, not measured costs.

Evaluation retains each arm's spending mode on the shared 50-pair Elemental
heuristic panel: 200 replay-checked games per final checkpoint, with native
seats and decks swapped across four legs per pair. Evaluation sources are
`4f1fdb2da3c9` (A) and `a29783f42e36` (B); their difference is report/scratch
text only. Both include the diagnostic constructor cleanup after training; it
changes neither the frozen training input layout nor payment behavior. Model
rows join through Etude's participant seat, rather than treating the native game seat as player
identity. The published summary retains the SQL, query plan and timing.

The pilot asks only whether removing pool-spending decisions visibly changes
early scores or decisions per game on this recipe. Updates are matched;
completed games and learner transitions need not be. The training decision/game
ratio divides all recorded environment decisions by completed games, including
decisions from unfinished or failed episodes; it is not mean completed-game length.
Trajectories differ once the policies differ, so their decision ratios cannot isolate saved prompts.
Two training seeds and a recipe with roughly 0.2% pool prompts do not establish
an improvement, equivalence or a compound-decision learning result. This pilot
does not expose fixed-source choice or implement C–E. MTG-265 owns the planned
learning-effect study.

An initial parallel launch (`518a4948ea80`) failed admission when a second L4
was unavailable; none of its four job commands started. The acquired machine
was deleted before the serial retry. The source and question stayed fixed.
Final run receipts, scores and launcher costs are recorded below.

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

B scored higher on these two early checkpoints. Two training seeds do not
resolve a learning effect, especially on a recipe where pool prompts are rare;
this does not establish that automatic spending improves learning. It supplies
no comparison of the proposed grouped source chooser. Retain A's defaults for
now and take the source-control design to review. The offline aggregation's
`EXPLAIN` uses scans and hash joins over only this fixed 800-game panel (1,600
participant rows), taking 0.0029 seconds; it is not a growing application query.

Both evaluation logs include `Invalid focus object ID 81 for action 12`
(ChooseMode), among similar warnings. The existing Python
encoder emits this when a focus object has no observation join and encodes -1.
The cause was not diagnosed in this experiment. Valid replay-checked games do
not establish that every action feature was informative: the pilot measures the
existing representation and procedure together. Keep that limitation separate
from game failures; changing the encoder mid-pilot would change its question.

Launcher cost observations after all acquired rentals ended (2026-11-13):

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

The MTG-183 final checkpoint was downloaded and checked separately. Its
SHA256 is `4f2ad656ba036a33d527212a7ac2badf2c5b9db136916f730846b07eef2fbea4`;
the loader refused it with `UnsupportedPolicyInput` because it uses input v1,
while this revision supports v9/v10. No trained-policy frequency row is
available, and the checkpoint was not rebound.

Evidence is published under `experiments/data/mana_spending/` in the
[artifact manifest](data.manifest.json): `games.jsonl`, `decks.json`, and
`checkpoint-admission.json`. Restore it with:

```sh
uv run --extra artifacts python -m manabot.data fetch experiments/data/mana_spending
```

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

| decks | policy | cast payments | capped | ≥2 joint outcomes, certified | ≥2 untapped-source outcomes, exact | ≥2 receipt variants, exact | identity collapse loses distinctions, exact |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| elemental | heuristic | 2062 | 99 | 1527/2062 (74.1%) | 1425/1963 (72.6%) | 874/1963 (44.5%) | 1380/1963 (70.3%) |
| elemental | random | 2589 | 120 | 1932/2589 (74.6%) | 1805/2469 (73.1%) | 1132/2469 (45.8%) | 1708/2469 (69.2%) |
| tla | heuristic | 2027 | 111 | 1326/2027 (65.4%) | 1215/1916 (63.4%) | 846/1916 (44.2%) | 1172/1916 (61.2%) |
| tla | random | 2113 | 38 | 1346/2113 (63.7%) | 1305/2075 (62.9%) | 884/2075 (42.6%) | 1236/2075 (59.6%) |
| heartbeat | heuristic | 134 | 16 | 107/134 (79.9%) | 91/118 (77.1%) | 67/118 (56.8%) | 82/118 (69.5%) |
| heartbeat | random | 164 | 36 | 143/164 (87.2%) | 107/128 (83.6%) | 73/128 (57.0%) | 93/128 (72.7%) |

The hidden source choice is common: 63–73% of completely enumerated ordinary
casts leave different fixed-source capabilities available. This contrasts with
0.11–0.60% of all decisions being today’s pool prompts. Equal-output producers
also conceal different exact objects in 60–70% of exact ordinary casts; output
capability alone is not an approved equivalence rule. Activated sources were
present at 212/18 Elemental and 492/53 TLA cast payments (heuristic/random);
their activation strategies are outside these fixed-source counts. None of the
observed cast payments retained a saved caller obligation.

Counterfactual **all decisions per game** on these A trajectories: B removes
all spend prompts; C–E replace only cast-payment prompts. C’s range spans
always taking automatic payment to always taking a nontrivial override. All
capped cast rows already certify ≥2 outcomes, so D’s decision count is known
even though some offer counts are incomplete. Grouping D would add decisions
beyond this flat, one-decision estimate.

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

Flat D has completely enumerated casts with **712** capability outcomes.
Heartbeat’s conditional p95 is 105–132; ordinary-deck conditional p95 is
49–59. Because 420/9,089 cast enumerations hit the cap, these conditional
quantiles are not a bound on the full population. There are 422 cast payments
already certified above 64 outcomes, including capped rows. A 64-slot policy
cannot use flat D everywhere, even with the optimistic capability collapse.

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
artifact directory. The earlier empty failed shape attempt remains local only.

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

## Recommended design

**Recommend a grouped native payment-outcome chooser at the fixed-source
payment boundary (D), after the spell’s modes, targets, X and cost are known.**
This is a design for review; implementing it is a separate Task. The learning
pilot does not test this chooser.

Keep one automatic plan as an explicit, bound native choice, describing the
sources and pool it leaves. Other complete outcomes remain reachable through
native groups, with branching that fits the 64-slot default. A uniquely
payable outcome skips as an ordinary trivial decision. An enumerator’s work cap
must never truncate the real legal action space. Expand groups lazily through
`PaymentModel`; retain exact source identities, restrictions, expiry, spend
riders and saved obligations. Equal current mana output alone does not justify
merging objects. Keep existing explicit mana-ability decisions at their real
native boundaries; do not model their costs or effects as hypothetical UI steps.

For the **policy**, each group and outcome must have native action descriptors
for what remains available, the floating pool and the relevant receipt. Keep
the announced source available to observation and action joins throughout the
pending choice. Carry
pending choice state through the same observations, learning tokens and LLM
text in the change that adds it. Respect the supplied observation capacities;
never use first-legal truncation to fit this chooser. Grouping can add decisions
beyond the flat D estimates, which already add roughly 16–24 decisions/game on
ordinary decks. Whether that helps learning belongs to MTG-265’s study, not to
these counts or the A-vs-B pilot.

For **Academy**, render those same groups and bound Commands. Labels can say
“leave this Forest untapped”, “float {R}”, or “spend two colors”; use visible
object joins to distinguish equal-output sources. Show the automatic plan and
its automation identity explicitly. Preserve keyboard access, stale rejection,
reconnect to the actual pending decision, and replay/fork identity. Commit the
selected sources and receipt through native payment execution once. Group
navigation supplies no cancellation or undo of already committed costs or
revealed information.

Do not flatten outcome products into priority (E): its measured menus already
exceed capacity, and many costs are not known at that boundary. Keeping A
leaves the requested source choice hidden. B only removes pooled-unit prompts;
it does not supply source control. A browser-only override would divide the
shared policy/Academy contract.

The `MANAGYM_EXPERIMENT_AUTOSPEND` switch remains an explicit reproduction
control for this experiment’s frozen A/B revision. The recommended production
design drops forced automatic spending: the follow-up should replace that
switch with the bound automatic-plan choice in the shared chooser. No Academy
code, default training behavior or production deployment is changed here.

## Limits

- Mirror games under scripted or early policies show how often the decision
  arises, not how much a strong player gains from it.
- The shape comparison counts offers. It does not measure play or learning
  under a shape that is not implemented. Manual tapping at every priority
  window is a qualitative alternative, without a measured offer count here.
- Ten TLA builds in retained file order are a bounded convenience sample, not
  a representative sample of the entire Limited corpus. The eight Heartbeat
  games per policy are a constructed upper-end example, not a metagame rate.
- No fixed-tapper run or trained-policy frequency run was performed. Native
  instrumentation attributes redundant output; it does not simulate removing
  a tap's events, other triggers or downstream strategic effects.
