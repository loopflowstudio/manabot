# MTG-135 interim size and batching evidence for MTG-122

Delivered 2026-10-18 UTC. This is an interim evidence handoff, not the final
training recommendation. MTG-122's execution hold remains in force. Nothing in
this handoff starts its complexity study or changes its accepted design.

The [report](experiments/grid-early-learning-013.md), [complete summary](experiments/data/grid-early-learning-013-expanded-summary.json)
and [chart](experiments/data/grid-early-learning-013-scores.svg) cover all 25
declared cells: 23 completed 131,072-transition endpoints and two four-stream
deadline cutoffs. Both evaluation cohorts passed their full artifact, source,
trace and replay audits: 2,944 games in 92 checkpoint/opponent panels. Every
complete cell has one training seed and only four paired deal blocks.

At 64 streams, 64×2 scored 53.125% against greedy and 84.375% against random;
384×2 scored 62.5% and 84.375%. The latter is a promising candidate, not a
reliable architecture ranking. At 512 streams, all models were weak at this
short experience endpoint. More transitions per second do not establish more
learning per dollar, and the four-stream results show that Adam update count
alone does not explain learning quality.

The [frozen longer protocol](experiments/protocols/longer-learning-017.md) is now
running in MTG-135: all five architectures at 64 and 256 streams plus 64×2 at
four streams, two fresh training seeds, 524,288 transitions each, and four raw
checkpoints. Its exact recipe, competence thresholds and time-based selection
rule were fixed before admission. GPU caps are reserved within MTG-135's $50;
MTG-121's separate $5 startup work is independent. The report records the
account-capacity admission deviation without changing the scientific recipe.

## How this transfers to the complexity study

Both designs use the existing GRU semantic-card encoder, visible-state attention,
flat action offers and scalar masked-mean value. The grid's source recipe binds
the product `ur-lessons-vs-gw-allies` semantic pack. MTG-122 instead requires its
common research catalog, exact native-content bindings and fixed learnable tensor
shapes across all complexity arms. These are related model families, not
interchangeable checkpoint identities. Keep MTG-122's admission checks and fresh
initializations; do not copy weights or transport-ID buffers from this grid.

Changing streams also changes optimizer grouping, empty-filter frequency,
staleness and EMA updates per transition in the present learner. A final handoff
must give the exact transition budget and schedule mapping, not simply replace
four streams with a larger number while retaining an iteration budget. Its
measured score and timing apply to this w6 world. MTG-122 must retain its planned
bounded calibration across all three pools before estimating a full study's cost.

The copied files are the working evidence in this checkout. `manifest.json`
binds their exact bytes and the source report commit. Raw game/trajectory evidence
remains retained by MTG-135; its manifests and locations are named in the report.
The final recommendation remains pending longer-run retention, evaluation and
analysis, and will be delivered separately without replacing this dated record.
