# MTG-193 open questions and assumptions — 2026-10-26

Design: `scratch/19d240-batch-shared-evaluation.md`. MTG-184's landed notes
live at `ce43a5cf7:scratch/questions.md`.

- **Assumption — compute identity unchanged:** grouping across matchups stays
  under `shared-eval-v1` because v1 never fixed batch composition (it already
  varies with arrival order). If a reviewer reads the contract as fixing
  per-matchup grouping, bump `EvalCompute.version`; that is the only change.
- **Assumption — served-child RSS target ≤ 90 MiB:** derived from local import
  measurements (47 MiB light set vs 216 MiB today); confirm on Linux.
- **Open — oversubscription culprit:** not reproducible on macOS. The rental
  thread census by role and thread name decides which pool caused ~1,000
  threads on `cpu32`; the bound is enforced regardless.
- **Coordination:** the before/after calibration runs through MTG-183's
  rental runner and allowance (< $1.50); this Task launches no rentals itself.
- **Gap — code anchors still load Torch in-child:** heuristic/random/scripted
  players are built through `flat_mc`/`players`, which import Torch. The accepted
  scope (MTG-183's model-vs-model cohort) is fully served and Torch-free;
  splitting `flat_mc` so code-anchor games are light is a candidate follow-up.
- **Assumption — supervisor census recorded, not enforced:** the group-worker
  parent hosts Etude callbacks and checkpoint preparation whose threads this
  change does not own; failing an allocation at close would discard results.
- **Assumption — budgets:** game/replay 4, serving 16 (CUDA driver/profiler
  threads). Local counts are 1; the rental census confirms or revises them.
- **Stacked on MTG-188 (Jack Heart, 2026-10-26):** this branch merges
  `jack/mtg-188-measure-inference-startup-decision-latency-and-game` (PR #381).
  Its scratch files (`mtg188-context.md`, `inference-perf-demo.md`, …) are the
  parent Task's context, not direction for MTG-193. MTG-188's serving/worker
  timing was ported into the split modules (`serving.py`, `serving_protocol.py`,
  `native_game.py`); its service-start failure test now injects a `ChildProcess`
  launch failure instead of a `multiprocessing` start failure.
