# bb4160 blockers and decisions — 2026-10-28

- **World contradiction, execution stopped.** The exact requested MTG-183
  checkpoint (`4f2ad656ba036a33d527212a7ac2badf2c5b9db136916f730846b07eef2fbea4`,
  1,071,166 bytes) was fetched and verified from
  `r2://manabot/jobs/f3748bcabac8/checkpoints/learner_transitions-331180341.pt`.
  It binds **w13**, policy-input v1. This checkout builds **w14**;
  `CheckpointOpponent` rejects it with `checkpoint requires an explicit
  compatible w14 world binding`. The directive requires current-world admission
  and forbids changing the opponent or forcing admission. A historical w13
  experiment or a different current-world opponent needs an accepted design
  revision; neither is assumed here. No rental or scored game was started.
- **Etude unavailable.** Installed pinned SDK 0.2.10 refuses
  `/Users/jack/.etude/launcher.db` as written by another version. No database,
  authentication, dependency pin or other Task's execution was changed. This is
  a separate launch prerequisite, not the reason for world rejection.
- **Independent scope.** Versioned LLM session admission/request/reply binding
  is implemented and tested. Sampling, vLLM service routing, checkpoint version
  propagation, cohorts, paid execution and settlement remain unfinished.
  Training overlap and asynchronous learning are deferred: this evaluation
  needs frozen players, and PRs #389/#383 remain open.

Obsolete MTG-207 rendering notes remain at
`c5d406e3d2a701525c15c8500a30f7da7896cc40:scratch/questions.md`.
