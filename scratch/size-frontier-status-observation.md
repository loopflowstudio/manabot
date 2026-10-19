# Study006 status observation, 2026-10-16

At approximately 18:36 UTC, Jack Heart asked how much training had finished and
when all three experiments would finish. Read-only Etude reads confirmed:

- Numerical run bd16b4a8e512 succeeded. Local metadata covers 16 collection/update
  iterations for each of 64 and 512 streams plus the declared replay replicas.
- Profile replacement 5aec98e566b2 succeeded. Recovered metadata reports completed
  training at 128 updates for each of small, middle and large. Full binary recovery
  and final analysis are separate from these successful worker outcomes.
- Original small-13561 run dd9172bae366 remains running, with empty log output and
  no progress clocks. Its current iteration is unknown; elapsed rental time does
  not establish how much training completed.
- Replacement reference f3470da00668, middle-13561 8c9a349e0d2d and small-13562
  d54203a3342e failed CUDA initialization and the RTX 5090 admission check before
  training. These are a new failure after the output-directory repair. Logs report
  CUDA unknown error; do not infer the provider or driver root cause from that alone.
- Watcher PID 55927 was alive. The original Task Session
  ae7607971f6a4c8f8b3ab55b9f35b754 was present and idle in the fresh Session read.

No additional jobs were launched or existing controllers changed by this status
inspection. The learning comparison has no reliable completion ETA until GPU
admission is repaired. Its protocol allows up to 12 hours for subsequent CPU arena
evaluation; this is a cap, not a measured forecast. Keep MTG-135 incomplete.
