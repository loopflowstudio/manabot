# PPO CUDA baseline (MTG-338)

Exploratory throughput measurement, not playing-strength evidence. Intended
claim to test: “Current PPO sustains roughly 0.26 optimizer steps/s on an
NVIDIA L4 at the Elemental w64l2 recipe, 256 streams × 64 transitions, both
seats, TF32, bucketed shapes and live tokens; backward and clipping account
for roughly three quarters of update time.” This expectation comes from the
historical MTG-325 run. No implementation candidate exists yet.

The restored MTG-325 harness is the starting point. This baseline adds five
unprofiled repeats after warmup, a separate cProfile run and one-minibatch
CPU/CUDA operator trace, optimizer/exposure/error counts and final-weight
digests outside timing. Every repeat restores initial weights and a fresh Adam
optimizer on the same collected batch. Collection is timed separately. Eight
streams × eight transitions provide an easier real workload as a harness check.
The launcher cap is $4 for one hour on one L4, with no retries.

Timing includes the actual update and a final CUDA synchronization. Warmup,
copies, digesting and the profiled update are excluded. A future implementation
comparison must alternate both arms at least five times using a frozen harness
and verify behavior; this baseline alone cannot establish a speedup.

- 2026-11-13: the shared `l4-etude` ceiling ($0.60/hour) was below the
  available L4 offer ($0.69/hour). Created `mtg338-l4` from that template,
  with one machine and a $0.75/hour ceiling, retaining the $4 launch cap.
- The local launcher database refuses SDK 0.2.16 as a different writer
  version. Launcher commands use its existing 0.2.15; workers retain the
  repository's locked SDK. The database is unchanged.
- Run `8f56c7ef9fe9`, revision `5853054ab`, was cancelled after roughly nine
  minutes of worker execution without a completed timing sample. The launcher
  reported approximately $0.12. Direct inspection found an active Python
  process and 14,066 MiB of L4 memory allocated; the provider summary incorrectly
  reported zero GPU memory. External stack inspection was refused because the
  container lacks `SYS_PTRACE`; no stack was obtained. These are missing
  measurements, not zero-throughput observations.
- The next baseline starts afresh: progress markers identify collection and
  each update, and a separate diagnostic invocation enables periodic internal
  stack dumps plus the operator profile. The five timing repeats run in a
  separate process without either profiler or traceback timer.
- Before a clean comparison, the restored harness also needs the production
  runtime settings it omitted: `torch.set_num_threads(stage.execution.threads)`
  (one in this recipe, as in `training/execution.py`) and `seed_random`'s
  deterministic cuDNN setup. The forthcoming baseline records these settings.
  This corrects the measuring instrument, not the PPO implementation; no
  speedup relative to the historical harness can be claimed from it.
