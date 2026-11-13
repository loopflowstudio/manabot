# How often players choose which mana to spend — MTG-347

Exploratory. A cast taps its lands automatically; when the pool then holds more
usable mana than the cost needs, the player chooses which pooled unit pays each
pip (`mana_spend` decisions). This report counts those decisions on the decks
people train on, separates the ones caused by automatic tapping taking more
sources than the cost needs (MTG-342) from other payments. This first slice
measures frequency; comparing replacement shapes and recommending a design
remain planned work. It makes no claim about learning; the comparison of compound
and sequential decisions is MTG-265's.

## Run it

```sh
uv run python -m experiments.mana_spending count --games 8 --seed 0
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
uv run python -m experiments.mana_spending shapes --source experiments/data/mana_spending --out experiments/data/mana_spending/shapes --cap 16384
uv run python -m experiments.mana_spending summary --source experiments/data/mana_spending/shapes
```

This enumerates fixed-source subsets at every actual payment, then native unit
spending to completion. Each final remainder must preserve saved payment
obligations under `PaymentModel`. It includes deliberately producing surplus
mana. Activated abilities remain explicit native decisions outside this automatic
boundary; their presence is recorded, and these are not counts of every possible
mana-ability strategy.

Separate columns count untapped-source capabilities, floating pools, rule-visible
receipts and their joint outcomes. Exact source identity counts expose the
additional distinctions lost by grouping producers with equal output. Such a
grouping is a measurement definition, not permission to hide object choices.
The per-payment cap counts source-subset and receipt traversal visits. A cap hit
retains lower bounds; incomplete rows never enter exact offer quantiles.

For folded offers, every legal cast at an observed priority window is tried on
an exact clone. Casts that reach one payment immediately supply their outcome
count; casts that first require other announcements (targets, modes or costs)
are marked unresolved. Windows with an unresolved or capped cast are excluded
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

Planned after the frequency and shape slices: one small run on the Elemental
recipe compares today's per-unit decisions with automatic spending: seeds
34701 and 34702 per arm, 32 matched updates, scored against the heuristic,
launched under the aggregate USD25 authorization. Each training job allows at
most USD3 and one attempt; the launch uses `--max-usd 17`, reserving up to USD8
for evaluation. The native `MANAGYM_EXPERIMENT_AUTOSPEND=1` switch takes the
first feasible unit; A explicitly sets it to 0. Source selection stays the same.
Evaluation uses the same switch as training, on the shared 50-pair Elemental
heuristic panel (200 replay-checked games per final checkpoint). It asks only whether removing the
spend decisions visibly changes early learning or steps per game on this
recipe. Two seeds cannot support a claim about either shape.

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

## Recommended design

Not selected in this frequency slice. The next slice must measure source-choice
outcomes as well as pooled spending: today's low prompt rate says nothing about
the hidden choice of which lands to leave open. The shape comparison and small
learning comparison remain required before a final recommendation for policy
inputs and Academy.

## Limits

- Mirror games under scripted or early policies show how often the decision
  arises, not how much a strong player gains from it.
- The shape comparison counts offers. It does not measure play or learning
  under a shape that is not implemented.
- Ten TLA builds in retained file order are a bounded convenience sample, not
  a representative sample of the entire Limited corpus. The eight Heartbeat
  games per policy are a constructed upper-end example, not a metagame rate.
- No fixed-tapper run or trained-policy frequency run was performed. Native
  instrumentation attributes redundant output; it does not simulate removing
  a tap's events, other triggers or downstream strategic effects.
