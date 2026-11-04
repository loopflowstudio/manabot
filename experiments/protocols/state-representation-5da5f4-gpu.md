# GPU horizon comparison — Task 5da5f4

Prospective, 2026-11-04; fresh initialization, distinct from the retained CPU
study. Jack Heart requested actual GPU learning toward a sustained plateau,
within the original cumulative $25, and authorized a necessary Etude SDK release
and Vast use. No winner promotion, merge or MTG-213 allocation is authorized.

## Frozen comparison

`experiments/state_representation_gpu.py` is the executable declaration. Pin its
source commit before launch. Both arms use current-world authored Allies/Lessons,
width64/depth2, policy FiLM, no recent events or temporal memory, v7 shared effects
and explicit joins. Pool versus GRU changes effect and continuation reduction
jointly, as in the CPU study. New source/rules and 512 action slots make this a
prospective study, not a continuation of CPU weights or scientific identity.
Other observation axes remain 60 cards, 40 permanents, 128 facts, two focus
objects, 32 events and the delivered v7 effect capacities. No hidden truncation.

Four fresh paired seeds 74201–74204, alternating treatment order within seed.
32 streams, one complete game per stream per update, both seats, cross-balanced
curriculum. Target 2,048 updates: 65,536 games per replicate, sixteen times the
previous horizon. Record actual decisions, optimizer exposure and completion.
CUDA float32 training only, 32 GiB process memory watchdog. No CPU substitution.
Use the existing absolute damped-policy schedule, identical between arms, with
no exhausted 128-update horizon mapping. Its learning rate is clipped between
5e-6 and 1e-4 and remains 1e-4 through this horizon; tau follows absolute updates.
A flat curve therefore cannot be attributed to a zero learning rate.

Export initialization and every 256 updates, plus final raw weights. Evaluate
these checkpoints after each replicate's training, on deals 85001–85032 with
both deck assignments and both seats (128 games/checkpoint), against the
source-pinned heuristic. This measures a predeclared curve; it does not select a
checkpoint or alter training. Final update 2,048 alone additionally scores
untouched deals 86001–86064 (256 games). CPU float32 inference, four active games,
batch32, 1 GiB residency, 120 seconds/game, 10,000 commands and ordinary replay.
The evaluation reserve is 1,200 seconds shared across all points of a replicate.
Missing games or points remain missing, never zero scores or substituted seeds.

## Plateau and uncertainty

At the fixed final horizon, examine the three successive 256-update intervals
from 1,280 through 2,048. For EACH arm and EACH interval, compute score gains
within each of the four training seeds, then the mean and 95% Student-t interval
(df3). A sustained plateau requires all six intervals entirely inside
[-0.05,+0.05]. A nonsignificant slope alone fails this positive bounded-change
criterion. This is a conditional small-sample diagnostic, not a guarantee of
saturation beyond the observed horizon. Report the complete curves, initialization
gains, per-seed results, within-seed paired-deal uncertainty, and limitations.
All 72 curve endpoints must complete before a qualified plateau claim. Flat
poor play can satisfy the diagnostic but does not establish useful competence.

Final architecture contrast uses only the separate held-out update-2,048
scores: paired GRU-minus-pool seed means and df3 95% interval, plus training time
and exposure. No automatic selection, broad equivalence or causal attribution
to just one reducer. Four seeds and one authored matchup limit generalization.
There is no optional stopping or score-based extension. If the bounded run
cannot reach its target or establish a plateau, retain the evidence and report
that the requested plateau objective remains unmet. A further horizon requires
a prospective amendment before new scoring, within the same unspent allowance.

## Allocation and failure accounting

Prior known provider bill $0.07258259388618171; keep the existing $6 ancillary
reserve pending settlement. Maximum new scientific rental reservation $17 total
(eight attempts, at most 2.5 hours and $2.125 each), plus at most $0.50 excluded
CUDA admission: cumulative reservation $23.57258259388618171, leaving $1.4274.
A quote above the scientific envelope is refused. Provider disk and transfer
must be included; Vast offers must have free upload/download under the supported
SDK selection. No network volumes. Jobs get a 6,000-second training watchdog,
1,200-second evaluation/finalization reserve and remaining setup/cleanup time
inside their 2.5-hour leases. Wall stops export learner state and are explicitly
incomplete for this comparison. No failed-seed retries or survivor selection.

Use task-local provider/storage configuration and the released Etude launcher;
retain its version separately from the worker's locked SDK/runtime. Supported
provider guardians and controller cleanup own each rental. Check deletion and
final billing, retain uncertain costs as reserved, and preserve every attempt.
Do not launch a second provider while acquisition on the first is uncertain.
An excluded seed74200 CUDA admission can check actual collection/update and
throughput before launch; its score never selects treatments or scientific seeds.
