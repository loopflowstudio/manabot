# MTG-338 measurement assumptions

- The optimize Graph is at its first measure step and HEAD has no runtime
  change. Measure the current implementation as the baseline; do not invent
  an optimization during this step. Use the existing L4 workload and a $4 cap.

- At optimize-plan, no working design exists in this checkout. Use
  `scratch/design.md` for the implementation handoff and keep the single
  experiment report in `experiments/ppo_cuda.md`. Choose the one-method CUDA
  embedding substitution first; defer other gather sites until it is measured.

- At implement, keep this step to the runtime candidate and focused lookup
  checks. The Graph's later measure/gate steps own the recovered equivalence
  harness, paired-run instrumentation, real CUDA execution and broader model/
  training acceptance. No speedup or CUDA equivalence is claimed locally.

- At paired measure, compile the kept method verbatim from `734f2cd5c` beside
  the candidate in one process. All other runtime and native code is identical;
  resetting the method is outside timing. Use the recovered CPU harness on the
  rental, never full game/training checks on the laptop. Retain full first-step
  raw gradients and first/final weights in separate audit runs.

- Paired measurement stopped after authenticated launch admission and a separate
  availability read both reported no L4 capacity. Do not substitute another GPU
  for the requested L4. Restore the kept implementation under the measure skill's
  keep rule, retain the failed admission evidence, and leave no queued work.
