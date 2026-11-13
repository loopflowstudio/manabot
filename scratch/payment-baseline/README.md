# Payment affordability: baseline (step 1, measure)

Commit `e82a20033` (v0.2.3), macOS arm64 laptop, load average 35-48 throughout.
Stock binaries: `cargo build --release --example bench_steps --example bench_search`
(the Engine Budget build), kept as `bin/*.e82a20033`. Instrumented binaries:
same commit plus `instrumentation.patch` with `--features benchmark`
(`bin/*.instrumented`); the patch is uncommitted in the working tree and every
line is behind the feature. Digests are unchanged in every run of both builds.

## Verdict

Worth optimising. Payment code is about a quarter of `bench_steps` and over a
third of `bench_search`, and on this workload none of it needs a search:
every one of the 522,421 (steps) and 13,686,633 (search) plans was answered
correctly by arithmetic over the pool and the fixed outputs (0 mispredictions),
because no payment model in either benchmark contains a source with a searched
ability.

## 1. Share of CPU

| | bench_steps | bench_search |
|---|---|---|
| payment, sampled (stock, `sample` 1 ms, inclusive under `flow::mana`) | 26.9% (658/2444) | 38.6% (9458/24500, seconds 5-35 of seed 1) |
| - plan/witness search | 18.3% | 27.8% |
| - model construction and the rest of `flow/mana.rs` | 8.6% | 10.8% |
| payment, timers (instrumented; model + plan scopes only) | 0.89 s: 21.2% of its own 4.2 s, 25% of stock 3.5 s | 26.2 s: 27.1% of its own 96.8 s, 35% of stock 74 s |
| - plan | 0.66 s (1,270 ns each) | 19.3 s (1,412 ns each) |
| - model | 0.23 s (414 ns each) | 6.9 s (467 ns each) |
| everything else | ~73% | ~61-65% |

Timer shares over five instrumented `bench_steps` runs: 21.1-21.3%. The two
methods agree within a few points; the sampler also counts caller glue in
`mana.rs` (obligation vectors, dropping models and paths) that the timers skip.
Ceiling: deleting all payment work would make `bench_steps` at most ~1.37x and
`bench_search` at most ~1.6x faster.

The search costs 2-3x what model construction costs. A 9-source model takes
~0.4 us; a payable plan takes ~2.5 us because the witness taps one fixed source
per recursion, cloning the search state and inserting it in a hash set each
time (4.6 calls per payable plan; malloc/free/memmove/SipHash lead the self-time list).

## 2. Distribution (bench_steps; bench_search is within a few points, see `search.report.txt`)

- 5.14 plans and 5.49 models per step; per step p50 3, p90 12, p99 36; 17% of steps do none.
- Per priority legal-action computation: mean 1.08 plans, p90 3, max 12; 41% do none. 76% of plans come from there but only 37% of plan time.
- Per plan: 2.47 witness calls, p50 1, p90 6, max 11.

Arithmetic class of each plan (prediction checked against the search result):

| class | plans | answer | calls/plan | plan time |
|---|---|---|---|---|
| pool + every source's best output < mana value | 43.7% | no | 1.00 | 3.7% |
| waterbend cost (not predicted; model is all fixed sources) | 32.2% | 53% yes | 3.64 | 60.4% |
| pool + all fixed sources pay | 16.7% | yes | 4.56 | 32.8% |
| fixed sources cannot pay, nothing else to search | 5.4% | no | 1.00 | 1.7% |
| pool already pays | 2.0% | yes | 2.00 | 1.5% |
| needs search / nested activations | 0 | | | |

- 0.0% of models contain a source with a searched ability; mean 11 sources (p90 24, max 37).
- `waterbend_payment_after_taps` is 15.7% of plans and 46.9% of plan time: the waterbend
  candidate loop builds the same model and re-searches once per candidate (5.3 calls, 3.8 us each).
- Rebuilt models: 74.6% of builds produce a model already built earlier in the same step;
  61.4% have the same player and (characteristics, tapped) revisions as an earlier build in the
  step (what a revision-keyed cache of unreserved sources would reuse), with 0 cases where the
  revisions matched but the sources differed.

## 3. Numbers to beat (stock build)

| | user CPU s, median (range) | instructions retired, median (range) | runs | digest |
|---|---|---|---|---|
| bench_steps | 3.565 (3.48-4.00) | 62.47 G (62.44-62.68 G) | 10 | 15843333459759757527 |
| bench_search | 74.10 (73.42-75.47) | 1314.96 G (1314.83-1316.81 G) | 5 | 13733780288109658817, 5337676601429031145, 10035331835025021780 |

Runs alternated search/steps (first five `bench_steps` runs back to back). 101,650 steps, 0 capped, 0 errors.

## Limits

- User seconds on this host vary 15% for `bench_steps` (one 4.00 s outlier; the other nine are within 8%)
  and 2.8% for `bench_search`. Instructions retired (`/usr/bin/time -l`) vary 0.4% and 0.15%
  and are the sensitive comparison; a change must still be confirmed in user seconds, interleaved
  against `bin/*.e82a20033`, since fewer instructions is not always less time.
- One workload: the curated UR Lessons vs GW Allies decks. They have no modal lands, filters or
  mana creatures with chosen outputs, so "0% need search" is a fact about these decks, not the
  card pool; waterbend's 32% share is likewise theirs. Any shortcut needs the search as its fallback.
- The `bench_search` sample covers 30 s of seed 1, not the whole run; the timer figure covers all of it.
- Timer scopes exclude classification and fingerprinting, but the instrumented run is 20-30% slower
  overall, so its own-elapsed shares understate and the vs-stock shares slightly overstate.
- "Same revisions" ignores pending-choice reservations (applied after the base sources), life and
  activation-used flags; no mismatch was observed here, which does not prove a cache key sound.

## Files

`runs/` raw outputs and `time -l` reports; `steps.report.txt`, `search.report.txt` full counter
tables; `*.sample-shares.txt` and `runs/*.sample.txt` profiles; `tools/` the two parsers;
`instrumentation.patch` the feature-gated counters.
