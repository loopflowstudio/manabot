# b015d5 update measurement (PR #435, merged as `45cc961a2`)

Full engine-measure notes, review history and raw tables: `git show
645b9be21:scratch/b015d5-update-measure.md` and the `b015d5` sections of
`645b9be21:scratch/questions.md`. Results live in `experiments/training-update-time.md`.

Still used by the throughput experiment:
- CPU behavior digest (seed 25201, 24 streams, 3 updates, `digest2.py`, logs in
  `.lf/logs/b015d5-measure/`, gitignored): raw / EMA / optimizer / attempts
  `76e926e3ea4fcde1` / `2a31f25602a86b9c` / `05eb51c04b5e2d14` / `cb230744dcd7c850`,
  72 games, 14,311 decisions. Step 2's packing digest compares against this shape.
- Laptop timing is noise-bound (load 13-31; 90.8-106.3 s/update on identical code);
  only same-machine pairs and per-phase µs/decision are claims.
- CPU dispatch counts (not GPU evidence): one collection sampling call 802 aten ops /
  37 host syncs (forward 629 / 17), ~722 calls per update; learner ~760 ops and ~31
  syncs per minibatch outside the GRU. The probe's `cuda-counts` phase measures the
  CUDA equivalents.
- Host screen: median `time/collection/environment_step_and_encoding` over updates
  2-11, 6.9-7.7 s fast vs 10.6-12.4 s slow; above 9 s is slow. A run never changes
  machines; replace a slow pod with `etude cancel`, then relaunch (never delete first).
