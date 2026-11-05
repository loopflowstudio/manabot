# Vast diagnosis and admission — completed evidence (summary)

Full note, with every measurement and limitation:
`85cf29970:scratch/vast-host-diagnosis.md`. Evidence is manifest-published under
`experiments/data/state-representation-gpu-5da5f4/` (`trace-delivery-failure/`,
`cuda-traces/`, `batched-admission/`, `batched-plateau/`).

- `b153cc2e5a77` (`2b2061c8d`): CUDA references passed; one float32 routing
  gradient exceeded atol 1e-5 (1.34e-5 in 1/192). No learner. $0.013 reported.
- `af601877e` added independent float64 comparisons, with no learner changes.
  `911423cce90b` passed all four references and six routing cases, retaining
  complete inference and learning traces (59 outputs verified). $0.030.
- `37c1523a98ed` matched admission: four updates per arm on a 3090, 49 outputs
  verified, projection 55.67h / $13.77 at $0.2474/h. $0.052.
- `e51f61b92931` was lost at 13:09 UTC. The recovery audit shows our launcher
  deleted the rental after one unhealthy read (also `59185580`, `59152469`).
  Etude PR84 retains a rental through a 15-minute outage.
- First-step CUDA traces justified no further runtime repair. Concurrency
  slowdown was plausible contention, not causally profiled.
