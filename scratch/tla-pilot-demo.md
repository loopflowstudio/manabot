# TLA pilot endpoint demonstration — 2026-11-05 UTC

MTG-213, demo decision pass 1. **Advance to the caller's next delivery step.**
This is an autonomous review of the preliminary pilot's actual outputs, not
human acceptance, Task completion, or satisfaction of the full learning criterion.

## Experience and intended progress

The experience is to train a manabot on randomly paired decks from the frozen
134-deck trophy population, export its checkpoint within the bounded allowance,
and inspect paired evaluation and retained failures. The current design is the
[v10 protocol](../experiments/protocols/tla-trophy-213-v10.md), with accepted
decisions in [questions.md](questions.md). The
[report](../experiments/tla-trophy-pilot.md) and
[execution handoff](tla-pilot-postmerge-verify.md) describe the preceding pass.

Earlier passes lacked a trained export and endpoint evaluation. The preceding
execution produced both on NVIDIA L4, Run `ae76c17951c3`, source
`fe83a6b068dace2187613e1bf124f9677b077ff3`. This is concrete progress. The
100-update target remains unmet; a wall-budget pause at 16 updates is not a
completed 100-update study. No new paid run or historical match retry was used
for this demo.

## Actions and observed results

1. Ran the real analysis CLI against the retained Etude store:

   ```sh
   uv run --extra artifacts --extra reports python -m experiments.tla_pilot_analysis --source experiments/data/mtg213/gpu-v10-20261105/evaluation/store --out .runs/mtg213/demo-v10-analysis.json
   ```

   It exited successfully. Parsed output exactly equals the retained
   [analysis.json](../experiments/data/mtg213/gpu-v10-20261105/analysis.json),
   including pair bootstrap intervals and all failed/missing slots. All 64
   endpoint games completed: 9/32 wins against the matched heuristic and 26/32
   against initialization. Full-schedule results remain `complete=false`,
   `success=null`: 92 valid of 288 expected, 196 missing, one failed callback.
   The callback's missing `ETUDE_OUTPUT_DIR` remains a visible failure.

2. Read and hashed every v10 file listed in
   [the manifest](../experiments/data.manifest.json): all **315** local files
   match their published size and SHA256 entries. Raw and admitted evaluation
   checkpoint bytes both hash to
   `235c3d9f3310047e0a1bdc1eff18a0c9dfb5eeab1a0d2c577355a2f25f104648`.
   This checks local evidence integrity against the manifest; it is not a new
   remote artifact-store fetch or a fresh checkpoint gameplay run.

3. Recounted the actual
   [attempt ledger](../experiments/data/mtg213/gpu-v10-20261105/training/game-attempts.jsonl)
   and compared it with the training receipt and summary. All counts agree:
   4,205 settled attempts, 4,201 completed, two engine errors and two resource
   limits; 1,881 discarded samples and 754,114 valid learner transitions.
   Recomputed peak rolling 100-game error rate is 1%, below the selected 25%.
   The training receipt confirms CUDA, 16 updates and 2,954 optimizer steps.
   The retained failures demonstrate finite capacity and rules defects, while
   continued completed games and updates demonstrate game-local recovery.

4. Inspected the dated
   [settlement](../experiments/data/mtg213/gpu-v10-20261105/settlement.json):
   at 01:33:52 UTC, pilot inventory and unsettled hardware were empty after
   teardown. The [summary](../experiments/data/mtg213/gpu-v10-20261105/summary.json)
   records USD0.438120592 estimated rental cost, USD0.913464481 cumulative
   billed-plus-estimated cost and USD5.839145926 conservative reservations under
   USD10. These remain dated estimates, not fresh inventory or final invoices.

## Feedback, judgment and next action

No person participated in this headless demo; there is no new human observation
or approval. Jack Heart previously selected a preliminary pilot, first-64 legal
support shared across opponents, and game-local recovery at 25%. The latest
Task comment `81ec3f17-30e6-4821-a77e-75888b77f65d` directs continued delivery of
v10 while reserving a full-TLA successor for a damped-learning discussion.

Acting under that direction, the review judgment is **advance**: the bounded
pilot now supplies the trained endpoint and paired evaluation it previously
lacked, and the report accurately preserves its limitations. There is no
concrete implementation gap in this results-delivery pass requiring a backward
edge. No design change is agreed or proposed by this demo. No HTML presentation
or browser proof was required on this headless, non-UI surface.

The next action is delivery review of the existing report and manifest, using
the exact analysis reproduction and artifact checks above. Keep the two native
target-selection failures open for rules investigation and independent behavior
witnesses; final billing reconciliation is also outstanding. The 192 unreached
milestone games and four missing initialization games cannot be repaired by
relabeling new games. Scientific improvement, held-out generalization and
Elemental-relative learning pace remain unestablished. A new training protocol
and its learning-rule choice remain separate prospective work.

No server, watcher, rental or other background process was started by this demo.
