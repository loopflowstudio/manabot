# Combat and mana decision shapes on the Small TLA model

**Current launch: `mtg265-study-20261114d`, source `ee0281c0d402`.** All six
jobs are admitted on three provider-confirmed L4s, two jobs each. Each uses
256 streams and targets 500 updates on the common v10 input. The three arm
smokes completed two updates each; no study learning result is available yet.

| Arm | Seed 26501 | Seed 26502 |
| --- | --- | --- |
| Baseline | `baae8dd0fb5d` | `3cae72a9ae8d` |
| Combat pick-then-done | `8b1208dffc09` | `86569bef127e` |
| Manual mana taps | `ef6848121d91` | `5b578c31fbea` |

The three held rentals each have a USD16 cap and 23-hour timer: USD48 maximum
hardware, with USD5 reserved for evaluation and preparation below USD1,
within the USD55 authorization. The supply `mtg265-l4-held` permits manual
rental only (`max=0`), so this cohort cannot acquire additional machines.
The rental caps, rather than the launcher's duplicate per-job reservation,
are the spending authority for these already allocated machines. Jobs have
a 22-hour allowance and USD15.50 per-job ceiling. A local supervisor waits
for this cohort and then terminates exactly these three manual rentals:
`mtg265-l4-held-08fc662039b4`, `mtg265-l4-held-60c5dd4c4bee`, and
`mtg265-l4-held-996fb972dd40`. The timers remain independent of that supervisor.

Read current progress with `uv run etude status mtg265-study-20261114d`.
The stopped attempts below remain history; do not relaunch them.

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

The tracking-fixed calibration (`bd63b6fe1c88`, `099e3658b035`, source
`eb88310ef319`) reached collection but failed at update zero: native admission
reported `policy input v10 required` for mana that cannot cast nonartifact
spells. The 900-deck regime had inherited v9 from the historical frontier.
Fresh arms now use v10 consistently, retaining width 64, two attention layers,
four heads and 256 streams. No historical checkpoint or evidence is rebound.
The earlier frontier curve is only a planning reference, not a same-input
control for this study.

The six v9 study runs in `mtg265-study-20261114` were stopped before training:
four cancelled, two capacity-refused, all USD0 in the launcher. Preserve those
records. Verify the corrected input with each arm before replacing that cohort.

## Successful smoke and final placement

At source `ee0281c0d402`, all three current-input calibration runs completed
two updates: baseline `362f5ba6146d`, combat `9729cdd9186d`, and mana taps
`a767d947ac05`. These used 256 streams; only the numerical gate's tiny
continuation fixture uses two streams. The baseline TrainingRun records
144.08 seconds for two updates, including collection and learning, excluding
machine setup. This short smoke supports placement planning, not a sustained
throughput or strength claim. The calibration rental was terminated.

The initial v10 cohort `mtg265-study-20261114b` exposed conservative launcher
budget reservation per job despite shared GPUs. Its 48-hour jobs made later
rentals too short, and the worker correctly refused with `remaining allocation
cannot cover the frozen regime`. Stop that cohort; all its attempts ended at
update zero or before training. Its acquired machines were terminated.

The next attempt was `mtg265-study-20261114c`, source `ee0281c0d402`, with the
same six seeds/arms and 500-update target, but a 28-hour maximum per job and
USD9 job ceiling. The whole cohort retains its USD48 ceiling. Placement is
two jobs per Ada GPU, up to three machines, under USD0.28/hour. The launcher
quoted USD23.73 for that full maximum first-wave rental window, including
startup; this is a quote, not a bill. The scientific question, damping horizon,
population and evaluation schedule are unchanged. Earlier failed/cancelled
attempts remain in Etude and are not reused as completed study runs.

All six `...14c` jobs were capacity-refused before execution, at USD0.
A5000s were unavailable. A40 and RTX2000-Ada create requests returned HTTP400;
termination was requested and provider inventory confirmed no pods acquired.
The replacement L4 placement above was acquired successfully. Provider
inventory confirms 16 vCPUs and 86–94 GB RAM per machine at USD0.69/hour.
The question, six seeds/arms, 500-update target and evaluation schedule remain
unchanged; only hardware and admission allowances changed.
