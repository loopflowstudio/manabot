# How often players choose which mana to spend — MTG-347

Exploratory. A cast taps its lands automatically; when the pool then holds more
usable mana than the cost needs, the player chooses which pooled unit pays each
pip (`mana_spend` decisions). This report counts those decisions on the decks
people train on, separates the ones caused by automatic tapping taking more
sources than the cost needs (MTG-342) from other payments. This first slice
measures frequency; comparing replacement shapes and recommending a design
remain planned work. It makes no claim about learning; the comparison of compound and
sequential decisions is MTG-265's.

## Run it

```sh
uv run python -m experiments.mana_spending count --games 8 --seed 0
```

plays mirror games locally on CPU and prints one row per deck set and policy:

```text
decks       policy     games  decisions/game  spend/game  share  over-tap  no-spend games
elemental   heuristic     80               …           …      …         …               …
elemental   random        80               …           …      …         …               …
tla         heuristic      …               …           …      …         …               …
tla         random         …               …           …      …         …               …
heartbeat   heuristic      8               …           …      …         …               …
```

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
- **Over-tap.** The engine counts, at each payment, whether some land it
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

The following comparison is planned for the next slice; `shapes` is not yet a
CLI command. It will enumerate offers from the engine's payment model for the
same deals. Only today's per-unit shape is implemented.

| shape | what the player decides |
| --- | --- |
| per unit | which pooled unit pays each pip, after automatic tapping (today) |
| automatic | nothing; the first feasible unit pays |
| override | nothing, unless payments differ in outcome; then one choice per cast |
| outcome | one choice among distinct outcomes: what stays untapped or floating, and what the spell's receipt records (colors spent, spend effects) |
| folded | the outcome is part of the cast action ("cast X, leaving Forest") |

The comparison will report decisions per game, offers per decision (median, 95th
percentile, maximum) and the fraction of decisions with at least two distinct
outcomes. Outcome enumeration is capped; casts that reach the cap are counted
in their own column, never truncated.

## Learning comparison

Planned after the frequency and shape slices: one small run on the Elemental
recipe compares today's per-unit decisions with automatic spending: two seeds
per arm, matched updates, scored against the
heuristic, launched under `--max-usd 25`. It asks only whether removing the
spend decisions visibly changes early learning or steps per game on this
recipe. Two seeds cannot support a claim about either shape.

## Results

Not yet run.

## Recommended design

Follows the results: one design, with what it changes for the policy's action
space and for Academy.

## Limits

- Mirror games under scripted or early policies show how often the decision
  arises, not how much a strong player gains from it.
- The shape comparison counts offers. It does not measure play or learning
  under a shape that is not implemented.
