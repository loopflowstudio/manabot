# Forward-KL floor on TLA trophy policies

This exploratory comparison asks whether adding a forward KL to the damped
update pulls a saturated combat policy back and changes its held-out play.
The full training and held-out comparison is pending.

## CPU smoke, 2026-11-20

At update 400 of MTG-409's raw trophy run (`4c38775ec499`), seed 40902 had
larger mean attack and block gaps (11.33 and 21.19) than seed 40901 (6.87 and
17.64). The earlier probe recorded signed means only. Both pullback arms
therefore start from `trophy-w64-raw@40902/out/policies/00000400-raw.safetensors`
in experiment `tla_trophies`.

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
the behaviour and score panels.

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
