# Do fixed-source mana taps at priority and during payments change TLA learnability?

**The effect on w64 learning and training performance is still unmeasured.**
The comparison uses three matched training seeds (21501–21503), all 1,000
TLA trophy decks and the same 16 evaluation pairs per checkpoint. Each seed
trains w64 in BF16 with and without the additional fixed-source taps, targeting
250 updates. The USD35 whole-experiment cap includes startup and evaluation.
Undo is disabled. The learning outcomes are strength against the heuristic
over updates and hours, and head-to-head strength at matched updates.
Training performance is measured in decisions per game, decisions per second
and seconds per update. At initialization and each evaluated checkpoint, the
on-arm spot checks will cover how often the acting policy sees a float, chooses
it, and leaves floated mana unspent when the pool empties.

## Widened native treatment on 40 seeded deals

Random self-play averaged 508.25 surfaced decisions per game with the option
on and 227.60 with it off (2.23×). Against the heuristic, the random seat's
treatment changed the game average from 171.30 to 157.75. In random self-play
a priority float was offered at 12,885 of 20,330 decisions (63.4%) and chosen
at 8,182 (40.2%). These are uniform random choices from the full native
space. Initialization and learning curves for the w64 policy remain pending.

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
The four planned charts and the result artifacts await a completed comparison.

On 2026-11-18 UTC, Etude quoted USD3.10 for four 52½-minute RunPod L4
training jobs. No RunPod rental was started. The quote allowed USD3.16
for training and USD1.50 for cpu32 while the earlier Lium rental retained
a USD5.33 cap. Its statement still returned HTTP 404.

The requested comparison covers legal mana taps shared by bots and people
in Academy. At the rehearsed revision, a funded cast drops surplus tap
choices, resolution payments omit ordinary fixed-source taps, and Academy
starts both seats with priority taps off. Training and evaluation await
a revision that exercises those choices through the shared action space.
