# MTG-301 plan (working notes)

Steers received (full text in .lf/prompts/*steer.md):
1. Add an option-3 (fixed-shape lookups) arm; verify CPU outputs/grads within float tolerance before timing.
   Patch rebased onto 92560a1e0 (now experiments/l4_step_profile_option3.patch).
2. Report ends with "Implications for an accelerator-first redesign"; run w160l8 and a larger random-init size
   if budget allows; measurements only, rank directions. USD 3 cap.
3. MTG-302 bucketed arm: include only if its branch is ready before the rental.

Design: one L4 job. Stage A in-situ frozen w160l8 update 1 on CUDA (HEAD code) with hooks:
profiler windows over collection inference calls and learner minibatches, sync-debug windows
with stack attribution, input captures. Stage B offline replays of captured inputs per arm
(baseline / option3) x size (w160l8 / larger random init), subprocess per arm.

## Run
- 2026-11-12 10:23:53 UTC: Etude run 6441cc948c6a, cohort mtg301-l4-profile, machine l4-etude-349b5d17199f,
  RunPod pod 26qog3f0upfmuy, $0.59/h, revision 8b60fe289. One earlier launch (10:18) refused for stock; no pod.
- Run 6441cc948c6a failed in ~2 min (stage failure; finally-block bug lost stage logs). ~$0.03.
- Side effect: at 10:24:24 UTC, 30 s after this launch, Etude dispatched the already-queued MTG-267 run
  tla-frontier-c5fba4-cpu-split-v1-53dd18c20036 onto cpu32 pod n8nv3sul7f2m8l ($0.96/h). Its owner had queued it;
  not cancelled. Launch/wait admit all waiting work, not only the named cohort.
- Steer 2026-11-12 (MTG-300 done, negative): catalog encoded once per weights version (198 encodes / 198 steps);
  report catalog kernel time per learner step as encoder cost, not removable duplication.
  Real CUDA candidates: validation syncs (MTG-299), padded/compacted shapes (MTG-302).
- Steer (MTG-302 ready, PR #477): bucketed arm via MANABOT_BUCKETED_SHAPES=1. Needs the branch's engine changes,
  so a second job runs at a merge of the MTG-302 branch with this experiment (candidate=bucketed).
- Steer: env var is a transition flag; report must end recommending either a typed Execution field
  follow-up (serialized only when set, CLI, receipts; delete env var) or removing the switch.
- Budget raised to USD 10 total, still one machine/window. Plan: one hand-held L4 rental (1.5 h, max $1.00),
  two queued jobs: option3 at 2de5f4131 (main 92560a1e0 + experiment), bucketed at 06f2d6f18
  (MTG-302 branch 54cd65420 merged with experiment). Local CPU smoke passed for both; bucketed CPU parity ok.
- 11:27 UTC rented l4-etude-6fc061afd2e6 / pod jvkaqth5y6wyt9, window to 12:57, max $1.00
- Done 12:40 UTC: option3 run d8ad784530d7 (src 4a002968e), bucketed run 1999b57daf19 (src e55157ac5 on local
  branch mtg301-bucketed-run, not pushed). Rental terminated; posted billing $0.143, window bound ~$0.80.
  Evidence published to experiments/data/l4_step_profile; report experiments/l4_step_profile.md; follow-up Task c85f69.
- Compress pass: HEAD's experiments/l4_step_profile.py equals what ran at 4a002968e/e55157ac5 except runs()'s
  `hours` parameter; left unchanged so the merged module stays the evidence.
