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
