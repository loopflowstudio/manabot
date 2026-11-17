# One token per game thing: v10 rows against input v11 (MTG-344)

> The code that ran this experiment was removed from main; the last commit with it is `d2d3fa7753`.

**Exploratory.** This run asks how the new layout trains against the old one
on the current recipe, and what it costs; its numbers are not a finding.

## Question

Input v11 gives each game thing one row of the policy's object table: cards
outside the battlefield and stack, permanents that carry their own
characteristics, and one stack segment for spells and abilities. In v10, a
battlefield creature is two attention tokens (a card row and a permanent row
joined only by list order), a stack spell is a card row and a stack ability is
no token. Does a policy trained on v11 play the same Elemental games better or
worse than one trained on v10, at the same number of updates, and what does
v11 cost to encode and run?

## Method

- Arms: the ten-deck Elemental recipe (`elemental_variants.elemental_recipe`:
  two-seat self-play over uniform deck pairs, width 64, two attention layers,
  256 streams) with `observation.policy_input_version` 10 or 11 and nothing
  else changed (`require_single_axis`). Both carry the same turn state, so the
  arms see the same information in different rows.
- Two shared seeds per arm (34401, 34402), 200 updates planned, one RTX
  A5000 each (L4 and RTX 4000 Ada had no capacity), Etude runs
  `ab6897b250fc`, `b3a65969050b` (v10) and `96f142a125e2`, `943990a1fd8a`
  (v11) at `eb604cc29e`.
- Evaluation: scheduled checkpoints (updates 20, 40, 75, 150) and each
  run's final checkpoint against the native heuristic over the 50 fixed
  Elemental pairings, four legs each (200 games per checkpoint), Etude eval
  runs `one-token-layout-v1-c08486c12446` and
  `one-token-layout-v1-763f8bc45436`.
- Strongest confound: two seeds per arm. A difference smaller than the
  seed-to-seed spread of one arm says nothing.
- Cost cap: USD 25 for training and evaluation, under the launcher's cap.
- Encode and forward costs and row occupancy: `uv run python -m
  experiments.one_token_layout measure` on one laptop (below).

## Results

At matched updates the two layouts are indistinguishable against the
heuristic. Over both seeds (400 games per arm and update, ± one standard
error):

| Update | v10 rows | v11 one token |
|---:|---:|---:|
| 20 | 0.225 ± 0.021 | 0.200 ± 0.020 |
| 40 | 0.305 ± 0.023 | 0.278 ± 0.022 |
| 75 | 0.385 ± 0.024 | 0.398 ± 0.024 |

Every difference is within about one standard error, and v11's early
deficit reverses by update 75. Final checkpoints, one seed each: v10 0.470
(139 updates), 0.535 (156); v11 0.510 (136), 0.395 (115). The chart is
`experiments/data/one_token_layout/report/one_token_layout.html`.

Throughput per learner transition matched: both arms collected 3.3–3.4 M
transitions per hour, and their time splits (collection 1.42–1.45 h,
learning 42.7–45.0 min of 2.18 h) agree. v11 updates took longer only
because they held more transitions (52–61 k per update to update 75 against
45–50 k for v10), which follows the games the policies played rather than
the layout.

Laptop measurements (`uv run python -m experiments.one_token_layout
measure` at that commit, 8,507 observations of 40 random Elemental games,
`experiments/data/one_token_layout/costs.json`):

| | v10 | v11 |
|---|---:|---:|
| Card rows per side, p99 / max | 43 / 48 | 31 / 44 |
| Permanent rows per side, max | 25 | 25 |
| Stack rows, p99 / max | 1 / 4 | 3 / 5 |
| Python `encode_observation_into`, median µs | 109.4 | 109.4 |
| CPU forward, batch 256, median ms | 95.7 | 86.9 |

Native `encode_into` over 6,935 curated observations (median / p90 µs):
main's v10 5.88 / 8.42, v11 6.29 / 8.92.

## Cost

Training $2.35 (four runs, $0.57–0.60 each); evaluation $0.30 (two passes,
$0.08 and $0.22). Total $2.65, under the USD 25 cap.

## Limits

- Exploratory: two seeds, 115–156 updates of a small model; the standard
  errors cover game noise, not seed variation.
- No head-to-head: one arena game encodes both seats with one observation
  layout, so a v10 checkpoint cannot play a v11 one.
- The legacy arm runs through the deprecated v9/v10 fork; its encodings match
  452d7b5357 byte for byte on 12,403 curated observations.

## Log

- 2026-11-14: two runs (v10 seed 34401, v11 seed 34402) were refused at
  machine creation and re-queued as `b3a65969050b` and `943990a1fd8a`.
- 2026-11-14: the A5000 ran 50–60 s per update against the 40 s budget, so
  every run ended at its wall-clock watchdog (139, 156, 136, 115 updates);
  the arms are compared at the updates all four reached.
- 2026-11-14: the v10-against-v11 head-to-heads were removed after the first
  evaluation pass failed on them ("players have different observation
  ABIs"); 232 heuristic matches from that pass are kept.
