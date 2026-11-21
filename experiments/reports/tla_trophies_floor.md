# Forward-KL floor on TLA trophy policies

This exploratory comparison asks whether adding a forward KL to the damped
update pulls a saturated combat policy back and changes its held-out play.
Etude admitted all eleven full training runs on 2026-11-21 UTC.

## CPU smoke, 2026-11-20

At update 400 of MTG-409's raw trophy run (`4c38775ec499`), seed 40902 had
the largest mean attack and block gaps (11.33 and 21.19) across the three
seeds. The earlier probe recorded signed means. Both pullback arms start
from `trophy-w64-raw@40902/out/policies/00000400-raw.safetensors` in experiment
`tla_trophies`. The other seeds' gaps are in the
[MTG-409 report](tla_trophies.md#three-seed-training-run).

Each of the five jobs completed one local CPU update at seed 40902 with two
streams. Etude verified the delivered warm-start file against SHA-256
`896ee236e0227ae4b94f68e2732b8955a2a7c7c1cb9f9415e62a11296c7d3ce4`.
The two update-zero policies had identical arrays. We collected one two-stream
batch with that starting policy and rescored the same positions with each
arm's update-one raw policy. Saturation counts decisions with |declare logit
− decline logit| > 9.2.

| Raw policy | Attack gap | Saturated attacks | Block gap | Saturated blocks |
| --- | ---: | ---: | ---: | ---: |
| Shared start | 12.68 | 15/19 | 16.98 | 2/2 |
| One update, ε = 0 | 11.41 | 14/19 | 16.24 | 2/2 |
| One update, ε = 0.01 | 9.20 | 7/19 | 12.65 | 1/2 |

These counts describe one seed, one update and one small batch. Each of the
three fresh policies also played one held-out deal at update one, crossing
decks and seats. Etude completed three matches with zero failures and rendered
the behaviour and score panels with the corrected smoke tracking directory.
Each arm lost all four games, declared 40% of offered attackers and declared
68.75% of offered blockers. The first render read an older local tournament
alongside the smoke and left the panels empty.

## Full comparison

The recipe at `e5d6e045e` declares freshly initialized ε = 0, 0.002 and
0.01 arms with identical seeds 40901–40903, up to 400 updates and raw
checkpoints every 100. Each checkpoint plays all 32 held-out deals against
the heuristic. Both warm starts use seed 40902, fresh Adam and a fresh damping
schedule for up to 100 updates. Compare attack and block gaps, saturated
shares, board-edit shifts and forward KL at matched updates. Compare the
fresh arms' held-out scores and behaviour per seed and across seeds.

Etude's combined L4 quote printed USD23.69, including queued commitments,
against USD24.98 in declared job caps and a USD25 experiment ceiling. Only
the three control seeds had immediate placements. A separate quote placed
both warm starts together at USD1.035 each for 2.75 hours plus startup. The
evaluation runs locally. Rental spend for this implementation smoke was USD0.

At 05:37 UTC on November 21, Etude admitted all eleven runs at revision
`6ae723eb8421` under the refreshed USD23.69 quote and USD25 ceiling.
The three control seeds started on `mtg265-l4-s3-9fe9ece0494c`. The other
eight runs entered the same rental queue. The run addresses in experiment
`tla_trophies_floor` are `control@40901` through `control@40903`, the same
three seeds under `floor-0.002` and `floor-0.01`, and `pullback-0@40902`
and `pullback-0.01@40902`.

### Held-out controls at initialization

All three update-zero controls played each of the 32 distinct deals once,
with four seat/deck assignments per deal. The 96 matches completed without
a failure (`tla-trophies-held-out-32-3dc0bcc7f65f`). The rendered per-seed
cells each contain 128 games. Declaration rates divide declarations by the
number offered.

| Seed | Score vs heuristic | Attackers declared | Blockers declared |
| --- | ---: | ---: | ---: |
| 40901 | 7/128 | 43.40% | 59.87% |
| 40902 | 7/128 | 43.30% | 60.13% |
| 40903 | 7/128 | 43.32% | 60.00% |

### Does the zero-weight control reproduce MTG-409?

At initialization the two studies agreed for each seed on score, attack
declarations and block declarations. By update 100, all six policies
declared at least 98% of offered attackers and 99% of offered blockers.
At 400, the same two seeds in each study still declared every blocker.
Seed 40903 recovered a smaller block gap and declined many blocks in both
studies. Each value below uses 32 held-out deals and 128 crossed games per
seed, or the corresponding learner's update-400 combat probe. The historical
run is `tla_trophies/trophy-w64-raw` at `f9c19a2656` and the fresh control is
`tla_trophies_floor/control` at `6ae723eb8421`.

| Seed | Study | Score | Attack declared | Block declared | Attack gap | Block gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 40901 | MTG-409 | 71.9% | 91.8% | 100% | 6.87 | 17.64 |
| 40901 | Fresh control | 67.2% | 93.5% | 100% | 9.84 | 15.38 |
| 40902 | MTG-409 | 68.0% | 99.7% | 100% | 11.33 | 21.19 |
| 40902 | Fresh control | 60.2% | 99.9% | 100% | 10.88 | 24.01 |
| 40903 | MTG-409 | 68.0% | 99.0% | 64.1% | 6.68 | 0.76 |
| 40903 | Fresh control | 65.6% | 97.3% | 57.0% | 7.71 | −0.82 |

The fresh control reproduced the combat pattern. Its mean final score was
64.3%, five percentage points below the historical 69.3%. The zero-weight
loss regression checks numerical compatibility directly. The training runs
show how much their learned outcomes differed even with matched seeds.
[MTG-409 report and archived curves](tla_trophies.md#held-out-evaluation-through-update-400)

### Training window for ε = 0.002

On November 21, Etude recorded exit 75 for `floor-0.002@40901` at update
321 after assigning it a 13,252-second placement with limit kind
`over_budget`. Its deadline was 13:59 UTC. The job log also retained a
native trigger-mode panic. The launcher recorded a saved stop, and the
last scheduled raw checkpoint from that attempt is update 300. Seeds
40902 and 40903 both reached 400 on their seven-hour shared placements.
