# Exploratory TLA trophy kickstarting trial: decaying forward-KL teacher loss

This exploratory run tests whether kickstarting fresh self-play policies (`w64` and `w160l8`) with a linearly decaying forward-KL loss toward a frozen trained `w64` teacher (`tla_trophies_floor/tau-x2@40902` update 800, 0.824 vs heuristic) accelerates learning and hands off cleanly to pure self-play on the complete 7,859-pool TLA trophy split. No committed prediction or threshold is declared; observations are directional and will not be cited as findings without a dedicated replicated study.

## Design choices & assumptions

- **Distillation loss:** Forward KL `policy_weight * scale * KL(π_teacher ‖ π_student)` over legal offers on actor-selected rows (Schmitt et al., 2018), matching the mode-covering gradient `π_student - π_teacher` and cleanly zero-baselined in nats. Optional value-head distillation (`value_weight`, default `0.0`) is supported in `Kickstart` (scalar MSE or categorical WDL cross-entropy) and left off in this initial policy-only sweep.
- **Fade schedule:** `Kickstart` decays monotonically from 1.0 at update 1 to exact `0.0` once `fade_updates` (or `fade_transitions`) elapse, supporting `"linear"` (default) and `"cosine"`. Once faded to exact zero, teacher admission and forward passes stop completely and the teacher is released from device memory. `tau`, `reference_forward_kl`, and learning-rate schedules stay untouched.
- **Teacher forward placement:** Teacher forwards run under `no_grad` inside learner minibatches on the ~25% advantage-selected rows (with `teacher_admitted = AdmittedRollout(teacher, batch)` so student and teacher architectures/readings/tables are fully independent). Because `DampedPolicyLearning` runs 1 epoch over ~25% retained rows, this does ~4× less teacher inference than collection-time evaluation and needs zero extra rollout storage.
- **Behavior policy:** Pure on-policy student self-play (`behavior="current-self"`), avoiding off-policy importance-sampling bias and giving DAgger-like teacher supervision on the student's visited states.
- **Experiment scope & budget (`experiments/tla_trophies_kickstart.py`):** Four arms to 400 updates (`fade_updates=200`, leaving 200 pure self-play updates after handoff) — `w64-ks-0.2-f200` and `w64-ks-0.5-f200` (2 seeds: `40901`, `40902`, compared against retained R2 `tla_trophies_floor/tau-x2` checkpoints with no redundant fresh `w64` baseline) plus `w160l8-fresh` and `w160l8-ks-0.2-f200` (seed `40902`, testing cross-architecture transfer from the `w64` teacher to a 17× larger student). Estimated total cost ~$12–$16 (under the USD 25 exploratory cap).

## Status

- **2026-11-26:** Kickstarting implementation (`Kickstart`, `SelfPlay.teacher`, decaying forward-KL/value loss, Etude resume validation, diagnostics, unit tests) and exploratory experiment module [`experiments/tla_trophies_kickstart.py`](../tla_trophies_kickstart.py) prepared.
- **2026-11-27 05:58 UTC (launch, commit `bb403b819627`):** Verified real-teacher (`tla_trophies_floor/tau-x2@40902` u800, SHA-256 `3456250f4fb6fe9ee8ef4fb0d2700b1196cceb97a44eac394b012a9c91a2e26f`) loading and fade-to-zero locally on CPU for both `w64` and `w160l8`. Sized A40 placement hours/caps in `bb403b819627` (`max_hours=8.0`/`max_usd=2.65` for `w64`, `16.0`/`5.20` for `w160l8`) and launched all six runs under `--max-usd 25` on three 2-slot RunPod NVIDIA A40s (`mtg446-a40-48g-s2`, $0.59/hr, planned first-wave ceiling $19.32):
  - `tla_trophies_kickstart/w64-ks-0.2-f200@40901` & `@40902` (`mtg446-a40-48g-s2-158b2582c037`)
  - `tla_trophies_kickstart/w64-ks-0.5-f200@40901` & `@40902` (`mtg446-a40-48g-s2-c1a6cb441892`)
  - `tla_trophies_kickstart/w160l8-fresh@40902` & `w160l8-ks-0.2-f200@40902` (`mtg446-a40-48g-s2-ec23e9a44ef6`)
