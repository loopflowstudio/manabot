# TLA trophies

## Population audit, 2026-11-20

The retained 17Lands Premier Draft dump
contains 562,924 distinct draft/match pairs from 101,496 drafts. Of these
drafts, 8,763 have at least seven observed wins. The source SHA-256 is
`6bb12e5cf17e57a2624ca8ac83f15fd4415ea347aa58eac0ed82feb1b17d536b`.

| Observed losses through the seventh win | Drafts |
| --- | ---: |
| 0 | 1,135 |
| 1 | 3,240 |
| 2 | 4,383 |
| 3 | 5 |

The initial first-N derivation stopped at draft
`163f0b88025e494a8ab91cb30fa8bf37` (source data rows 134079–134088).
Its third loss was match 8 and its seventh win was match 10.
Scanning each draft to its final match also found 12 drafts with eight wins,
and 19 with exactly seven wins and a tenth match. These counts describe
different conditions and can overlap with the five above. First-N selection stops
reading a draft once it selects the seventh win, so excluding only those
five would still admit anomalous records.

A full-draft selection requiring exactly seven wins, at most two observed
losses, a final win and a final match number of at most nine yields 8,732
drafts, excluding 31 of the 8,763 candidates.

## Frozen training and evaluation inputs

The full-source builder selected 8,732 trophies from all 562,924 rows.
Each trophy retains the main deck and sideboard from its final winning
match. The pool split uses the existing salted draft-ID ranking across all
8,732 drafts. Playability was checked after assignment against every card in
each pool and human build.

| Split | Drafts | Playable pools |
| --- | ---: | ---: |
| Training | 7,859 | 7,859 |
| Held out | 873 | 873 |

A fresh checkout can reproduce both files from the retained source.

```bash
uv run --extra artifacts python -m manabot.data fetch experiments/data/mtg213/corpus-v1/games.csv.gz experiments/data/mtg213/corpus-v1/source.json
uv run python -m experiments.tla_corpus experiments/data/mtg409/corpus-v1 --source-dir experiments/data/mtg213/corpus-v1 --all-trophies
uv run python -m experiments.tla_pools experiments/data/mtg409/corpus-v1/corpus.json experiments/data/mtg409/pools-v2/corpus.json --held-out 873
```

## Packed L4 pilot

The solo control ran six updates at revision `197609239`. The packed runs
use `96a817842`, which adds local evaluation evidence retention to the same
training code. Each seed trains fresh v11 w64 in float32, with 64 streams,
one producer, both learning seats, damped updates, closed lists and
independent deck draws. Checks every five updates probe up to 256 attack
and 256 block decisions. Every ten updates, each EMA checkpoint plays the
same two held-out deck pairs against the first-64 heuristic, crossing seats
and decks for eight replay-verified games per seed. Local Etude evaluation
uses the delivered checkpoints and incurs no rental charge.

Etude 0.4.2 sets the slot count on the on-demand template. The solo control
used a 12-vCPU / 80-GB minimum. The two- and three-slot templates request
one L4 with at least 6 vCPUs and 48 GB each. The solo and packed observations
come from separate hosts. They measure these rental configurations, including
their host differences.

### Which packing to use

Two seeds on one L4 gave the best return in the initial pilot. The table
uses updates 2–6 for every seed, after its first update. Each seed's throughput is the
increase in learner decisions divided by elapsed wall seconds between the
first and sixth update records, including the update-5 probe. Aggregate
throughput is the sum of the simultaneously running seeds' rates. All three
rentals were priced at USD0.69/hour. The last column expresses steady-state
throughput at that price. Setup and delivery are included in the launcher
costs below.

| Seeds per L4 | Wall seconds/update per seed | Aggregate decisions/s | Million decisions/USD |
| ---: | --- | ---: | ---: |
| 1 | 45.2 | 332 | 1.73 |
| 2 | 34.6, 33.1 | 899 | 4.69 |
| 3 | 53.2, 56.6, 60.8 | 789 | 4.12 |

Median collection time per seed rose from 16–17 seconds at two slots to
39–47 seconds at three. The largest observed per-process CUDA allocator
reserve over the six updates was 0.25 GiB, and process RSS reached 6.44 GiB.
The probe took 1.32–1.35 seconds on the two-slot machine at update 5. The
three-slot calibration stopped after every seed supplied the six-update
window. There was no fourth-slot trial.

### Launcher record

| Experiment | Job and seeds | Final updates | Estimated USD |
| --- | --- | --- | ---: |
| `tla-trophies-cal-solo` | `trophy-w64` (40901) | 6 | 0.0779 |
| `tla_trophies` | `trophy-w64` (40901, 40902) | 30, 30 | 0.1911 |
| `tla-trophies-three` | `trophy-w64` (40901, 40902, 40903) | 9, 9, 9 | 0.1395 |

Etude estimated USD0.4085 across all three rentals. The launch passes capped
the solo control at USD1.25, the two-seed experiment at USD4.50 and the
three-seed calibration at USD3.75. All rentals ended.

### What the graphs showed

[Open the Etude report](../data/mtg409/report/tla_trophies.html).
[Checkpoint behaviour screenshot](../data/mtg409/report/behaviour.png).
[Combat probe screenshot](../data/mtg409/report/combat-probe.png).

The two training seeds finished 30 updates each. Both produced attack and
block gaps and all eight board-edit shifts at updates 5, 10, 15, 20, 25 and
30. At update 30, the attack gaps were 2.74 and 3.36 and the block gaps were
3.24 and 3.15. Neither reached the study's lock threshold of about 9.
For the opponent's 9/9 untapped creatures, the attack gap shifted by −0.085
and −0.041 and the block gap by +0.527 and +0.168.

All 11 behaviour curves had observations at updates 0, 10, 20 and 30 for
both seeds. The eight distinct EMA checkpoint digests produced the same
action sequence on each of the eight fixed games. Their scores were 0/8 at
every checkpoint. They declared 50% of offered attacks and 66.7% of blocks,
and spent 25.4%, 51.5% and 48.6% of available mana across the three turn
bands. The audit compared retained action sequences across checkpoints for
each deal and leg. All 64 games passed replay verification. The update probe
reads the raw learner, while evaluation reads the EMA, advanced once per
update at rate 0.999. These two held-out pairs give a narrow view of strength.

For the separately budgeted USD100 study, start with two independent seeds
per L4, repeat the packing check on the allocated hosts, and spend the longer
window on additional independent seed pairs and a broader fixed held-out
deal set. Evaluate the raw and EMA exports at matched checkpoints to measure
how long the behaviour changes take to reach the averaged policy.

The rendered files can be restored or the page regenerated from the run
records with these commands.

```bash
uv run --extra artifacts python -m manabot.data fetch experiments/data/mtg409/report
uv run --extra artifacts --extra evaluation etude report experiments/tla_trophies.py --source r2://manabot/jobs/tla_trophies --out scratch/qa/tla-trophies
```

## Thread and batch calibration

On the six-vCPU L4, native game stepping took about 2 seconds of a
32–43 second update. Collection's rollout recording rose from about 0.6 s
with two seeds to 17.7 s with three, at a one-minute host load of 20.
This motivates testing CPU contention and larger policy and learner batches.

Both calibration arms request six updates on two slots, with three Torch,
OpenMP and MKL threads per seed. Arm A uses 64 streams and 64 rows per
learner and bootstrap call. Arm B uses 256 streams and 512 rows per call.
The comparison uses aggregate learner decisions per dollar over updates
2–6, peak allocated and reserved CUDA memory, and rollout recording time.

All seven seeds completed six updates. Each row uses updates 2–6 for
steady-state throughput. Decisions per dollar uses USD0.69/hour.
The decision count covers all six updates, and the final column is
Etude’s estimate for the whole rental, including setup and shutdown.

| Arm | Slots | Threads/seed | Streams / rows | Decisions in six updates | Decisions/s | Million decisions/USD | Rental USD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 2 | 3 | 64 / 64 | 177,025 | 562 | 2.93 | 0.1196 |
| B | 2 | 3 | 256 / 512 | 721,697 | 1,307 | 6.82 | 0.1453 |
| B | 3 | 2 | 256 / 512 | 1,087,736 | 1,979 | 10.33 | 0.1905 |

Values below follow seed order, starting at 40901. Update time is wall
time between the first and sixth updates divided by five. Recording and
load are medians over updates 2–6. CUDA memory is the per-process peak
over all six updates.

| Experiment | Seconds/update | Rollout recording (s) | Host load (1 min) | CUDA allocated / reserved (GiB) |
| --- | --- | --- | --- | --- |
| `tla-trophies-cal-a` | 53.9, 50.9 | 8.40, 5.01 | 9.06, 8.58 | 0.182 / 0.242, 0.190 / 0.250 |
| `tla-trophies-cal-b` | 94.7, 89.5 | 8.04, 3.41 | 14.35, 16.31 | 0.280 / 0.359, 0.280 / 0.340 |
| `tla-trophies-cal-b-s3` | 89.4, 93.2, 92.0 | 3.69, 3.73, 3.62 | 35.59, 35.59, 36.25 | 0.287 / 0.361, 0.283 / 0.340, 0.277 / 0.348 |

The selected configuration packs three seeds on one L4, each with two
threads, 256 streams and 512-row learner and bootstrap calls. Larger
batches gave 2.33 times the throughput per dollar at two slots. Packing
a third seed then raised it another 51%. These comparisons used separate
rental hosts. All three rentals ended, for USD0.4554 in calibration and
about USD0.87 including the original pilot.

## Three-seed training run

Etude 0.4.3 admitted seeds 40901–40903 together on one L4 at USD0.69/hour.
The quote was USD7.59 for 10.75 training hours plus 15 minutes of setup.
The job caps total USD7.80. Each seed gets one attempt. The launcher placed
all three on `mtg265-l4-s3-22b1fe1ca732` at revision `4c38775ec499`.

Each seed targets 400 updates, with raw checkpoints at initialization and
every 100 updates. The evaluation reserves up to USD1.30 for one CPU32
rental. Each raw checkpoint plays the same 32 held-out deals, with four
seat and deck assignments per deal.

On the six-vCPU L4, three calibration seeds reserved about 1.05 GiB of
CUDA memory combined while one-minute host load reached 35.6. The earlier
rise in collection time with a third seed points to a CPU bottleneck. A
host with more vCPUs per GPU may support more seeds.

All three seeds reached update 400 on their first attempt. The launcher
confirmed deletion of the L4 at 05:14 UTC on November 21 and estimated
USD3.63 for the whole rental.

| Seed | Attack gap at 400 | Block gap at 400 | Attack gap shift, own creatures 9/9 | Attack gap shift, opposing creatures 9/9 untapped |
| --- | ---: | ---: | ---: | ---: |
| 40901 | 6.87 | 17.64 | +3.64 | −4.25 |
| 40902 | 11.33 | 21.19 | +5.92 | +0.57 |
| 40903 | 6.68 | 0.76 | +3.52 | −1.12 |

At update 400, making its own creatures larger increased the attack gap
for every seed. Block preferences varied substantially across seeds. Two
had mean block gaps above 17, while the third remained below 1.

### Held-out evaluation through update 400

Each of the 15 raw checkpoints completed all 32 scheduled deals against the
heuristic. The four crossed games per deal give 128 replay-verified games per
checkpoint and 1,920 overall. The fresh Etude render showed zero failed or
unplayed deals. Every point on the strength curve and all 11 behaviour curves
has 128 games per seed. [Open the final Etude report](../data/mtg409/report/raw-400/tla_trophies.html).

| Mean across three seeds | Init | 100 | 200 | 300 | 400 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Score against the heuristic | 5.5% | 58.3% | 63.3% | 62.8% | 69.3% |
| Land played when able | 84.85% | 100% | 100% | 99.96% | 100% |
| Cast a spell when able | 84.1% | 95.9% | 96.2% | 97.3% | 95.7% |
| Mana spent, turns 1–3 | 35.3% | 49.5% | 50.4% | 51.5% | 52.1% |
| Mana spent, turns 4–6 | 71.8% | 78.7% | 79.3% | 79.9% | 80.8% |
| Mana spent, turns 7+ | 71.6% | 67.6% | 71.2% | 70.3% | 71.4% |
| Attackers declared | 43.3% | 99.4% | 99.6% | 99.4% | 96.8% |
| Blockers declared | 60.0% | 99.7% | 95.1% | 94.0% | 88.0% |
| Chump attacks | 18.3% | 9.5% | 8.0% | 8.5% | 7.3% |
| Chump blocks | 41.9% | 32.7% | 24.5% | 24.7% | 20.9% |
| Spells cast on the opponent's turn | 3.5% | 4.3% | 3.7% | 3.6% | 3.8% |
| Turns per game | 19.59 | 19.01 | 18.89 | 18.88 | 18.69 |

The block declaration average at update 400 hides a split. Seeds 40901 and
40902 declared every offered block on the held-out deals. Seed 40903
declared 64.1%, consistent with its lower self-play block gap. The probe ran
80 times per seed and used 53–54 seconds, or 0.28–0.31% of active wall time.
[Strength](../data/mtg409/report/raw-400/score.png) ·
[Land plays](../data/mtg409/report/raw-400/behaviour.png) ·
[Chump attacks](../data/mtg409/report/raw-400/chumps.png) ·
[Combat log-odds](../data/mtg409/report/raw-400/combat-probe.png) ·
[Attack shifts](../data/mtg409/report/raw-400/attack-shifts.png) ·
[Block shifts](../data/mtg409/report/raw-400/block-shifts.png).

The final CPU pass completed its 44 outstanding deals on its first attempt.
Etude confirmed shutdown of `07c490-cpu32-61bccc01213a` at 05:23 UTC on
November 21 and estimated USD0.0927 for the whole rental. The pilot,
calibrations, long training run and final CPU pass have USD4.59 in
identified launcher estimates. Earlier evaluation results were already in
storage. Their rental charges could not be verified from this launcher's
records, so the whole-task cost remains unconfirmed.
