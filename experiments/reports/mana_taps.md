# Do fixed-source mana taps at priority and during payments change TLA learnability?

**Offering fixed-source mana taps made the w64 policy weaker on these TLA deals and slowed training per update.** After 50 updates the on arm scored 0/192 against the heuristic. The off arm scored 14/192 (7.3%). Head to head, the on arm scored 39/192 (20.3%) and lost the comparison in all three seeds. The on arm already trailed at initialization. Its score against the heuristic stayed at zero, while the off arm improved by two scored games across three seeds. This measures an early strength disadvantage. The longer-run learning rate remains unresolved.

Both arms trained on all 1,000 TLA trophy decks with the same w64 recipe, BF16, and undo disabled. Each seed pair used one L4. The common target was reduced from 250 to 50 updates based on the previous on-arm pace and the USD35 whole-experiment cap. The comparison uses EMA exports at updates 0, 25 and 50. Each checkpoint played the same 16 fixed deck pairs, with seats and decks crossed over four legs. Training seeds are the three units of evidence. Each eligible source has its own tap action, including identical lands. The policy sees the first 64 legal actions. Both choices affect how often a starting policy taps.

## Strength by update and active training time

Each cell lists the observations at updates **0 / 25 / 50**. Scores are points out of 64 scored games unless another denominator appears. At roughly 0.7–0.85 active hours, the on arm scored 0/64 for every seed at update 25. The nearest off-arm observations were 5/64 at 0.83 hours, 4/64 at 0.70 hours and 4/64 at 0.85 hours, respectively.

| Seed | Off vs heuristic | On vs heuristic | On vs off | Off active hours | On active hours |
| --- | --- | --- | --- | --- | --- |
| 21501 | 4 / 4 / 5 | 0 / 0 / 0 | 13 of 60 / 11 / 16 | 0 / 0.43 / 0.83 | 0 / 0.82 / 1.48 |
| 21502 | 4 / 4 / 5 | 0 / 0 / 0 | 16 / 9 / 15 | 0 / 0.70 / 1.15 | 0 / 0.75 / 1.29 |
| 21503 | 4 / 4 / 4 | 0 / 0 / 0 | 12 / 16 / 8 | 0 / 0.45 / 0.85 | 0 / 0.75 / 1.32 |

Etude completed all 432 scheduled matches. Of 1,728 game legs, 1,724 contributed scores across 431 complete deals. One leg of seed 21501's initialization head-to-head encountered `PanicException: current object appearance: Stale` on deal 385011. Scoring excluded all four legs of that deal. All endpoint deals were complete.

On the four replayed legs of fixed pair 0 against the heuristic, every on-arm checkpoint saw a priority float on 234/322 decisions (72.7%) and chose one on 128/322 (39.8%). Of 128 mana produced that way, 118–119 units (92.2–93.0%) expired. Every off-arm checkpoint offered zero priority floats over 241 decisions. These counts come from one fixed deck pair. The full spot-check table covers every seating, including head-to-head, with 216 seat rows linked to 108 replays.

## Training performance through update 50

Each pair used the same L4 host and seed. Seed 21501 shared its GPU until
the off arm finished. On the two other hosts the off arm finished before
the on arm started. Hours include active collection and learning.
Decisions/game and decisions/s use total retained decisions divided by
completed games and active seconds, respectively.

| Seed | Decisions/game off → on | Decisions/s off → on | Seconds/update off → on | Active hours off → on |
| --- | ---: | ---: | ---: | ---: |
| 21501 | 198.3 → 387.0 | 875 → 929 | 59.7 → 106.6 | 0.83 → 1.48 |
| 21502 | 198.5 → 374.7 | 635 → 1,035 | 82.5 → 92.7 | 1.15 → 1.29 |
| 21503 | 208.2 → 373.7 | 890 → 1,006 | 61.6 → 95.2 | 0.85 → 1.32 |

The median paired increase was 89% in decisions/game, 13% in decisions/s
and 55% in seconds/update. The complete-games collector retained more
decisions in each on-arm batch. Host conditions still varied over time.
For seed 21502 the off arm's first update alone took 705 seconds.

## Reading the checkpoint spot checks

Etude's `mana_spot_checks` table keeps one row per seat and leg of the
first fixed evaluation pair at each seating. Divide `float_offered` and
`float_chosen` by `surfaced` to get the shares of chooser decisions.
Offers count only actions in the policy's first 64 slots. The
`replay_sha256` names the gzip-compressed game record in the match's blobs.

`regular_expired` and `combat_expired` count units actually lost at their
respective native pool boundaries, including automatic steps. Replacement
effects can keep some or all of a pool. For the share of priority-floated
mana lost, divide `priority_expired_min` and `priority_expired_max` by
`priority_produced`. A zero denominator leaves that share undefined. Equal
bounds give an exact count. The bounds use unit totals, so spending or
partial retention from a mixed pool can leave a range even when color or
a spending restriction would narrow it.

### Local w64 smoke at `eaaa473837`

The CPU smoke trained seed 21501 for one update per arm and evaluated
initialization and the endpoint. All 24 legs across the six seatings
completed and passed replay. All 48 seat rows resolved to their retained
replays and their surfaced counts matched the recorded decision counts.

Against the heuristic, the four legs gave identical counts at updates 0
and 1 within each arm. Each row below describes one checkpoint.

| Arm | Decisions offering a float | Decisions choosing a float | Priority-produced units lost | All regular units lost | Combat units lost |
| --- | ---: | ---: | ---: | ---: | ---: |
| Off | 0 / 241 | 0 / 241 | 0 produced | 9 | 7 |
| On | 234 / 322 (72.7%) | 128 / 322 (39.8%) | 118–119 / 128 | 129 | 0 |

Local Etude runs were `312dfa8ab7b3` (off), `0c379dda01b7` (on) and
`mana-taps-paired-654696ec2472` (evaluation). These counts exercise the
measurement on one deck pair.

## Widened native treatment on 40 seeded deals

Random self-play averaged 508.25 surfaced decisions per game with the option
on and 227.60 with it off (2.23×). Against the heuristic, the random seat's
treatment changed the game average from 171.30 to 157.75. In random self-play
a priority float was offered at 12,885 of 20,330 decisions (63.4%) and chosen
at 8,182 (40.2%). These are uniform random choices from the full native
space.

| Players | Taps | Decisions/game | Skipped share | Decisions offering a priority float | Priority floats chosen | Scalar steps/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Random vs random | off | 227.60 | 73.4% | 0 | 0 | 3,136 |
| Random vs random | on | 508.25 | 60.5% | 12,885 | 8,182 | 4,718 |
| Heuristic vs random | off | 171.30 | 67.6% | 0 | 0 | 4,566 |
| Heuristic vs random | on | 157.75 | 65.4% | 1,694 | 929 | 5,710 |

All 40 games in each row completed. Each row replayed the same deck pairs
and post-selection random streams with reset seeds 1000–1039. Both random
seats received the treatment in self-play. Against the heuristic only the
random seat received it, alternating seats. Steps/s includes scalar observation
delivery and counting the offers on this machine. It does not measure learner
updates or GPU throughput.

## Earlier priority-only CPU rehearsal

| CPU rehearsal, two completed games per arm | Decisions/game, taps off | Decisions/game, taps on | Priority taps as a share of surfaced decisions, taps on |
| --- | ---: | ---: | ---: |
| w64 | 225.5 | 479.5 | 49.7% |
| w160l8 | 225.5 | 570.5 | 53.5% |

These two-game FP32 rehearsals used two CPU streams and one update per arm.
They exercised the earlier priority-only option at revision `206f60ca85`.

## Launch record

On 2026-11-18 UTC, Etude quoted USD5.33 for the four training jobs sharing a
Lium RTX 4090 in sequence. Each job had a USD1.95 cap. A separate cpu32
evaluation had a USD1.50 cap. The launch used revision `206f60ca85` and
`--max-usd 10`.

Launch `c7e08dd0607c` failed admission. Lium returned a timezone-naive
creation time for pod `76be0130-a892-48b6-982b-0c55c89d0855`. Etude
reported an invalid provider reply, recorded no training runs and could not
reconcile machine `lium-4090-e29369481ff0`. The pod was deleted through
Lium and its ID subsequently returned HTTP 404. The statement endpoint also
returned 404. The launcher's final cost is pending.

There are no GPU checkpoints or scored evaluation deals from this launch.

On 2026-11-18 UTC, Etude quoted USD3.10 for four 52½-minute RunPod L4
training jobs. No RunPod rental was started. The quote allowed USD3.16
for training and USD1.50 for cpu32 while the earlier Lium rental retained
a USD5.33 cap. Its statement still returned HTTP 404.

The requested comparison covers legal mana taps shared by bots and people
in Academy. At the earlier priority-only revision, a funded cast drops surplus tap
choices, resolution payments omit ordinary fixed-source taps, and Academy
starts both seats with priority taps off. The local w64 smoke above used the widened treatment.

### Three-seed RunPod comparison at `3f1d731a5f`

On 2026-11-18 UTC, Etude quoted all six 250-update runs on RunPod RTX 3090s
at USD0.50/hour. The first wave quoted USD17.12. The six seven-hour jobs
had a combined USD27.00 job cap under the launcher's USD28.17 training limit.
This leaves the earlier Lium rental's USD5.33 cap and USD1.50 for evaluation
within USD35. RunPod offered cpu32 at USD0.96/hour. Single-slot and two-slot
RTX 4000 Ada quotes both returned no available machines.

Launch `090cf96c199e` requested seed 21501 on single-slot RTX 3090s.
RunPod had no machine at admission. Etude started neither arm. The
two-slot quote for seed 21502 also returned no capacity. These attempts
produced no solo or packed throughput measurements.

Etude next quoted USD21.99 for the first wave of six single-slot RunPod L4
jobs at USD0.69/hour. Every job kept the 250-update target with a 6.5-hour
wall limit and a USD4.67 cap, including startup. The six job caps total
USD28.02.

Launch `2896671bcf5f` admitted all six runs. The four machine rental ceilings
Etude actually recorded were USD6.90, USD7.65, USD9.14 and USD13.63
(USD37.32 combined). The recorded maximum exposure exceeded the training
allowance even before evaluation. The launch was cancelled. Etude confirmed
all four L4 machines deleted by 07:30 UTC and estimated their respective
costs at USD0.14, USD0.17, USD0.15 and USD0.16. The earlier Lium statement
remains unresolved.

| Seed | Last launcher update, off / on | Updates with stored metrics, off / on |
| --- | --- | --- |
| 21501 | 4 / 2 | 3 / 1 |
| 21502 | 13 / none | 12 / none |
| 21503 | 8 / none | 7 / none |

Run addresses are `mana_taps/mana-taps-w64-{off,on}@{seed}`. All 23 sealed
training rows are in the [published CSV](../data/mtg385/launch-2896671bcf5f/training.csv).
At its first update the on policy chose priority taps on 38.6% of surfaced
decisions. Only initialization checkpoints reached storage. There is no
scored trained comparison or GPU packing measurement from this launch.

### Paired L4 training at `68d3a20b09`

On 2026-11-18 UTC, launch `6c36bc30a50d` placed both arms of seed 21501
on machine `mtg385-l4-pair-fe9567457c20`. Its recorded rental ceiling was
USD4.485 at USD0.69/hour. The runs share one L4 with at least 12 vCPUs and
80 GB of host memory. The previous 21501 runs recorded process RSS of
16.55 GiB off and 31.37 GiB on, which left little room for workers on the
original 48 GB template.

All six runs completed 50 updates and exported at 0, 25 and 50. At the
previous on arm's first-update pace, 250 updates would take about 38 hours
on that host. The complete-games collector closes games after its transition
quota, so equal update counts can contain different numbers of decisions.

Two additional packed rentals for seed 21502 failed admission. RunPod
acquired no machine for either L4 launch `1dfbfa40f973` or A40 launch
`6b04a0e6a69b`. Single-slot L4s were available for sequential training.
Launch `cb970eff0754` for seed 21502 and launch `e377ebc1adc0` for seed
21503 each recorded one USD8.7975 rental ceiling. The three acquired
ceilings total USD22.08 against a remaining USD27.55 training allowance.

Etude ran the evaluation as `mana-taps-paired-l4/evals-229c77cb7099` in
launch `302f3d92743c`, with a USD1.20 cpu32 rental ceiling. Training run
addresses are `mana-taps-paired-l4/mana-taps-w64-{off,on}@{seed}`.
Etude confirmed deletion of all four machines. Its machine ledger on
2026-11-18 showed USD0.7925, USD1.1239 and USD1.0536 billed for seeds
21501–21503, respectively, and approximately USD0.37 for evaluation.
The earlier cancelled RunPod run and unresolved Lium charge are recorded
above. The final whole-experiment bill remains open.

## Published results

Fetch the five artifacts with
`uv run --extra artifacts python -m manabot.data fetch experiments/data/mtg385/paired-l4`.
The interactive report can switch its horizontal axis between updates and hours.
[Learning curves](../data/mtg385/paired-l4/mana-taps-paired-l4.html) ·
[Training updates](../data/mtg385/paired-l4/training.csv) ·
[Scored games](../data/mtg385/paired-l4/games.csv) ·
[Spot checks](../data/mtg385/paired-l4/mana_spot_checks.csv) ·
[108 replay blobs named by SHA-256](../data/mtg385/paired-l4/replays.tar)
