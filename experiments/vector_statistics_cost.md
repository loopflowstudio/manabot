# Optional vector statistics cost

Exploratory implementation check, 2026-11-17. Jack Heart selected "1a and 2b":
independent optional-analysis failure and custom Python questions on every vector
game. The intended claim is optional-hook cost, including no-hook framework
cost, on a bounded real buffered workload; no inference or training claim.

Compare kept `76510eaf3f` without hooks, edited without hooks, and edited with
CardPerformance, Behaviour and a per-game decision counter. Same release build,
interpreter, dependencies and supplied observation buffers. Four vector streams
play sixteen complete Mountain-only games each at seeds starting 7, action 0,
with native trivial-decision skipping. Completed streams are masked while
others finish. No recordings, native summary files, policy inference, network
or artificial contention. Five rotating repetitions, one fresh subprocess per
cell, no concurrent cells. Time reset, buffered stepping, stream synchronization,
close and final aggregation; exclude imports, vector/buffer allocation and
post-run output hashing. Process CPU and elapsed time are separate metrics.
All arms must agree on game/decision count, winner sequence and final native
state digests. Report errors, medians and full ranges; overlapping ranges do
not establish a difference. This fixture is cheaper than a real neural training
loop and overstates hooks' share of such a loop. It does not exercise rich
combat boards or expensive user code. No GPU overhead claim.

## Results

**No measurable disabled-framework cost; enabled questions add CPU work.**
Frozen source/harness: `a409c94ff3b77f2eab8d10611e6d97743c07fef6`.
Kept source: `76510eaf3f`. Apple M4 Max, CPython 3.13.15, shared host.
All 15 cells completed: **960 games, 10,560 decisions, zero errors**. Winner
sequences and final native state digests matched exactly in every cell.

| Arm | Median CPU seconds / 64 games | CPU range | Median elapsed seconds | Elapsed range |
| --- | ---: | ---: | ---: | ---: |
| Kept, no hooks | 0.075061 | 0.065582–0.089283 | 0.365405 | 0.255930–0.519896 |
| Edited, no hooks | 0.083403 | 0.077970–0.085768 | 0.313312 | 0.237250–0.548380 |
| Edited, three questions | 0.185938 | 0.151913–0.194828 | 0.691590 | 0.517264–1.394685 |

Disabled CPU and elapsed ranges overlap the kept version: no measurable
framework difference under the stated range criterion. Enabled CPU ranges do
not overlap disabled ranges. The observed medians differ by **0.102535 CPU
seconds per 64 games** (about 1.6 ms/game, or 146 microseconds/decision), and
these questions use 2.23 times the disabled CPU work in this cheap fixture.
That is the cost of doing the requested questions, including full native facts
and validation, not a tax paid by users who register none. Elapsed ranges
still overlap; do not claim a reliable throughput ratio from this host.

The analysis increment is about 55% of enabled CPU work in this fixture, so
even deleting all of it could improve that CPU-work metric by at most roughly
2.23 times. Neural inference/learning and richer game states change the share;
there is no supported training or GPU percentage. Arbitrary user questions
may cost much more than these three. Initial reducer cloning/allocation and
memory footprint are outside this panel; vector streams require independent
reducer copies and retain compact per-game rows until their capture is released.

An unreported profile attributes substantial enabled work to terminal exposure
receipt parsing/validation and reduction in `CapturedVector._end_analysis`;
`decision_observations` adds active-view materialization. These are the opt-in
source paths, absent when questions are not registered. The profile's elapsed
values are not benchmark samples. An easier admission cell completed 16 games /
176 decisions; the measured cell does four times that work (64 / 704).

CPU seconds by repeat 0–4:

| Arm | Samples |
| --- | --- |
| Kept, no hooks | 0.071784, 0.078909, 0.075061, 0.065582, 0.089283 |
| Edited, no hooks | 0.082788, 0.077970, 0.085768, 0.083403, 0.083538 |
| Edited, questions | 0.194828, 0.188196, 0.177525, 0.151913, 0.185938 |

Gameplay SHA256: `8ca338e29bada996582c7346368548069900188f18e6e2fdcdfd965e3f72e3f6`.
Raw local evidence: `.runs/vector-stats-measured/{measurements,summary}.json`.
The pre-fix release extension used for the kept archive predates these Rust
changes; its native sources are identical to those at the kept commit.

Reproduce with isolated kept sources and their matching release extension:

```sh
uv run --no-sync python experiments/vector_statistics_cost.py run \
  --before .runs/vector-stats-before --output .runs/vector-stats-repeated
```

Validation: 34 capture/reducer/writer checks, 12 evaluation reducer checks,
four debug Rust observation/reassignment checks, library Clippy with Python
bindings, Ruff and diff whitespace checks passed. The Rust extension was
rebuilt in release and replaced atomically before Python/vector measurements.
Run `experiments/vector_statistics_cost.py` from its frozen revision
`a409c94ff3b77f2eab8d10611e6d97743c07fef6`. Both measured sources use
the dense vector transport.
