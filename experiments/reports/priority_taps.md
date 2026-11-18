# Do fixed-source mana taps at priority change TLA learnability?

Exploratory. **Learnability remains unmeasured for both w64 and w160l8.** The
GPU launch stopped before the first training command. The local CPU rehearsal
showed that offering taps increased the number of decisions per game, but it
covered only one update per arm. It cannot establish how quickly either model
learns to beat the heuristic.

| CPU rehearsal, two completed games per arm | Decisions/game, taps off | Decisions/game, taps on | Priority taps as a share of surfaced decisions, taps on |
| --- | ---: | ---: | ---: |
| w64 | 225.5 | 479.5 | 49.7% |
| w160l8 | 225.5 | 570.5 | 53.5% |

The treatment adds one choice per untapped fixed-output source at priority.
Identical Forests each get their own choice. Payment choices stay the same
and undo is disabled. The rehearsal used FP32 and two CPU streams. The planned
comparison uses BF16 on GPU, one training seed per arm, the same 1,000 trophy
decks and 16 evaluation pairs drawn from that population.

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
