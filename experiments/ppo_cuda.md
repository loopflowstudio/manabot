# PPO CUDA baseline (MTG-338)

Exploratory throughput measurement, not playing-strength evidence. Intended
claim: “Current PPO completes X optimizer steps/s on an NVIDIA L4 at the
Elemental w64l2 recipe, 256 streams × 64 transitions, both seats, TF32,
bucketed shapes and live tokens.” No implementation candidate exists yet.

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
