# Combat and mana decision shapes on the Small TLA model

Exploratory study. Compare the existing combat decisions and smart automatic
mana payer with combat pick-then-done and optional per-source mana taps. The
frontier Small model trains for 500 updates on 900 TLA pool decks; 100 decks
are held out, with 50 fixed paired matchups against the heuristic. Two seeds
per arm (26501, 26502) make six training runs. Evaluation includes updates
50, 100, 200, 300, 400 and the final checkpoint. No learning result is available.

## Authorization and existing execution

In conversation `9840d3d8-9b50-41b1-b74f-00c65efd0216`, Jack Heart said
"no i meant 55 dollars" (2026-11-14 12:14:59 UTC), followed by
"maybe 500 updates then" (12:15:20 UTC). The retained continuation summary
also quotes "anyway this is approved". The USD55 total covers calibration,
training and evaluation; it is not an additional USD55 for each arm.

Preparation is committed at `05b2b64622`. The whole-set combat arm was
omitted because the current policy action representation cannot describe an
arbitrary set of combatants. These are ordinary sequential policy decisions;
this study does not restore the removed compound neural decoder.

On 2026-11-14 at 22:03 UTC, launcher records contained no decision-shape
runs, no local launch process was active, and RunPod reported no pods.
Earlier L4/Ada/4090/A4000 attempts had not produced a running calibration.
A fresh A5000 offer was available at USD0.27/hour with two runs per machine;
the A4000 offer had no capacity. A new eight-update, two-run baseline
calibration was submitted as `mtg265-cal-a5000-20261114`, capped at USD1.
Submission was initially preparing the exact git source, not training.

Reserve up to USD49 for training and USD5 for evaluation, with USD1 for
calibration. The final launcher quote and measured calibration determine the
training hours and placement. Do not duplicate a live cohort or spend beyond
the remaining total. Preserve all failed and partial attempts.

## Calibration repair, 2026-11-14

Runs `6574e52d4932` and `21b2a10d2022` failed before training. The numerical
suite passed 40 checks, then its tiny continuation fixture failed in CUDA
attention backward with `Invalid max_seqlen_k:202` on RTX A5000, Torch
2.15.0+cu132. Both receipts and logs remain in Etude. The rental was terminated;
the launcher's retained estimate is about USD0.02 for the machine.

Live-token attention supplied the padded slot count as the maximum sequence
length, even when the packed batch contained fewer tokens. Tighten that
upper bound to the smaller of the slot count and packed token total. This
does not change attention membership or require a device-to-host read. The
sparse-batch forward/backward regression also runs in the CUDA admission
gate; that gate is not disabled. This is consistent with the efficient
backward constraint documented in
[PyTorch issue 129523](https://github.com/pytorch/pytorch/issues/129523).
The focused local live-token tests passed (5 passed, 1 CUDA skip); GPU
verification and successful calibration remain required before full training.

The launcher validates a job's own dollar allowance before sharing its cost.
Use USD16 per job and a USD48 ceiling for the complete six-run cohort, with
two jobs per A5000 and up to three machines. A 48-hour maximum allows slower
packed jobs to reach 500 updates; jobs stop at that target. The machine price
ceiling is USD0.30/hour. Calibration and evaluation stay within the USD55
total; these ceilings are not measured bills or evidence of completion.

The repaired Ada attempt (`728d1b881b40`, `b095dc8808e1`, source
`e3e15828cc48`) passed CUDA numerical admission, including sparse attention
backward and optimizer checks. Training then refused tracking because the
launcher bound the experiment to `decision_shapes`, while the job declared
`decision_shapes_calibration`. Keep that shared experiment name; mark short
calibration regimes with a `-calibration` suffix instead, which also keeps
their checkpoint tags out of scored evaluation. No study updates ran in
either attempt. The Ada rental remains held under its USD0.80 ceiling while
preparation continues.
