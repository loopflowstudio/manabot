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
  Evidence published to experiments/data/l4_step_profile; report experiments/l4_step_profile.md; follow-up Task MTG-312 (c85f69).
- Compress pass: HEAD's experiments/l4_step_profile.py equals what ran at 4a002968e/e55157ac5 except runs()'s
  `hours` parameter; left unchanged so the merged module stays the evidence.
- Realign: MTG-302 landed on main as bf4e280f7; `git diff 54cd65420 bf4e280f7 -- manabot/model` is empty, so the
  bucketed arm measured main's code (noted in the report's runs section). Remaining: push `mtg301-bucketed-run`
  (questions.md) at publish.

## Concept review (2026-11-12) — next implement direction
The measurements, decision and MTG-312 recommendation stand; the report's shape does not yet read as
science (question → plain answer → one result per question → cost → limits; provenance in a collapsed
appendix). Report-only edits, no new numbers:
1. Order: Answer, then Results as the brief's four questions with their letters — (a) syncs, (b) GPU idle /
   launch-bound (today unlabeled inside "Where a window's time goes"; name it, with the small-call host-bound
   finding), (c) catalog, (d) kernel compute and the arms — then scale, Decision, Implications, one-line cost,
   Limits.
2. Move "Runs and cost" (run/pod ids, torch versions, the four dated failure/admission lines), the numerical
   gates and the Evidence/reproduce block into a closing `<details>` appendix. Keep the cross-job baseline
   agreement sentence beside the arms table, where it qualifies the comparison.
3. Leave `experiments/l4_step_profile.py` and the option-3 patch unchanged (they are what ran).
Done: report reordered (Answer → (a)–(d) → Scale → Decision → Implications → cost → Limits → <details>
appendix); every number of 035b58723's report retained.
Re-review (2026-11-12, HEAD 6ed9622e2): all three items hold in the report; the baseline-agreement sentence sits
under the arms table. No further concept change; no open product choice. Publish still needs the `mtg301-bucketed-run` ref pushed (questions.md).

## Performance review (2026-11-12)
Checked against `summary.json`/`job.json`/in-situ traces of both jobs:
- Replay pass totals, learner medians, sync counts (118/208; bucketed 117/192), baseline repeats
  (≤2.1% learner, ≤3.6% inference), parameters/weights digests per arm, the $0.30 and $1.40 per-100-update
  arithmetic, and the "≈18 s" small-call tail (in-situ, recording subtracted per row) all reproduce.
- Added to the report: bucketed job's profiled in-situ calls 122–137 had 45–1,330 ms of host time
  outside any torch op (option-3 job 17–22 ms); window table uses the option-3 job; replays unaffected.
- Unproved, already stated as such: no CUDA baseline-vs-baseline numerics floor; no decision digest or
  matched-trace game evidence for bucketed; whole-update savings are projections; no learning evidence.
  Default adoption stays a person's decision at demo; MTG-312 (typed Execution field) carries the switch.
