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

## Long-run admission

At the measured 89–93 seconds/update, 400 updates take about ten hours
per seed. On November 20, the quoted 10.5-hour window plus 15 minutes
of startup cost USD7.4175 for one L4, or USD2.4725 per slot at full
occupancy. RunPod uses no reservation increment.

Etude 0.4.2 rejected the USD2.50 and USD2.75 per-seed quotes because its
admission check budgets the whole machine against each seed's cap. At
USD7.42 per seed, the quote fit only one seed under the experiment limit.
Queued attempts also use the whole-machine rate to bound their duration.
The rental itself is budgeted from the pooled caps, and runtime charges
are split by occupancy.

A local Etude 0.4.2 fake-provider check changed the new-rental admission
price to one third of the machine rate. At USD0.69/hour with zero setup,
each seed requested one hour for USD0.23. Three seeds received the full
hour together. A lone seed was also admitted, but received only 20 minutes.
With four seeds and two available machines, the first three received 40
minutes together and the fourth received 20 minutes. The scheduler divided
the pending dollars evenly between machines. Admission must check that the
seeds placed together fund the rental window it promises them.

The long run awaits the Etude fix. The local smoke registered two
checkpoints, and the resolved schedule test covers all 32 held-out deals
per raw checkpoint per seed. The pilot graphs above are the latest
measured behaviour trajectories.
