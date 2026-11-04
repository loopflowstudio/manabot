# TLA pilot post-merge verification — 2026-11-04

Decision: **Fix forward / iterate**. The merged enabling repair holds;
MTG-213's empirical acceptance remains incomplete. Return to implementation
in this Task. No new product choice or spending authorization is required.

## Evidence on main

PR #423 merged at 20:43:37 UTC as
`8e8d702f807d61675fc1bca46417eee69316daa0`. A fresh fetch confirms this is
origin/main. Checkout HEAD `6da60684cb628ec9b431dbbb9d7dc91348743622` and main
have the identical tracked tree `b78da9f8f947750effadd4a89bfaf81f0654f916`;
the verification therefore exercises main's source, without switching the
managed checkout. All required PR checks passed, including debug Rust,
integration, Academy and Engine Budget.

Fresh command:

```sh
uv run --extra dev --extra artifacts --extra evaluation pytest tests/eval/test_tla_pilot.py tests/training/test_action_cutoff.py tests/verify/test_native_buffer_capacities.py tests/env/test_observation.py tests/sim/test_heuristic.py -q
```

Result: **196 passed in 23.13 seconds**. These checks exercise real v9 runtime
admission, frozen-contract rejection, retained-prefix scalar/vector/Python
parity, original Command execution, masks and collection, matched heuristic
support, and paired-analysis accounting. The controlled native heuristic case
checks the preferred excluded action and the best retained action. No new
implementation defect was exposed. Existing full debug CI supplies the native
rules proof; the verification did not rerun the full Rust workspace.

## Acceptance and progress since the preceding pass

Accepted outcome comes from comment 1540e0ad, superseding the deferred original
brief: actual trophy-corpus training and evaluation, an admitted trained
checkpoint, and a qualified Elemental-relative learning analysis. Comment
ffd29028 subsequently requires the same first-64 legal support for heuristic,
candidate and initialization anchor. Merged v9 implements that requirement.

The preceding demo intended to deliver the legality/capacity/cutoff repair and
then execute a fresh bounded pilot. Its notes were cleared during delivery;
they remain readable at `d2ff60e6a^:scratch/tla-pilot-demo.md` and
`d2ff60e6a^:scratch/tla-pilot-verify.md`. Delivery, green CI, matched-support
integration and these fresh main checks are concrete progress. Missing old
notes from the working directory are not evidence of no progress.

The retained v5 training receipt at
`experiments/data/mtg213/gpu-v5-20261104/training/run.json` independently records
CUDA execution, 3 updates, 585 optimizer steps and 149260 learner transitions,
then `PyAgentError: observation capacity exceeded: actions 256 > 64`.
Its only exported stage artifact is initialization; monitoring checkpoints are
empty. The retained analysis records 32 valid initialization games (5 wins,
27 losses), 288 expected games, 256 missing, `complete=false`, `success=null`.
It explicitly reports no successful trained endpoint. These historical results
remain valid evidence of execution and failure, not of completed learning.

The merged PR body, v9 protocol and report all identify v9 as prospective.
No v9 execution receipt or trained checkpoint was found in the retained local
pilot evidence. Consequently the concrete remaining gap is an admitted trained
checkpoint with complete paired comparisons and a learning report. Advancing
would incorrectly mark that accepted outcome complete.

## Implementation continuation — 2026-11-04 20:50 UTC

Fresh v9 source freeze completed: Run `2171156493ca`, source
`6da60684cb628ec9b431dbbb9d7dc91348743622`, identical to the verified merged
tree. No executable change or deletion was needed.

**Delete — do not maintain:** obsolete demo-boundary instructions, completed
concept-review proof requests and CI repair narration in `scratch/questions.md`
have been replaced by current decisions and git references. No code deletion
targets remain. Superseded prepared Runs stay frozen evidence, never launch targets.

Fresh quotes refused Vast RTX5090 (price above USD0.81/hour), RTX3090 and
RunPod L4 (no capacity). A6000 availability was false. Configured A4000
availability was true at USD0.25/hour, but its subsequent quote also refused
capacity. No submission or paid allocation occurred. Final inventory at
20:49:51 UTC found no active pilot run, no RunPod pods and no unsettled hardware.
The live Vast rental59159114 is the independently owned representation Run
886233843d34; its storage packet verifies ownership, and it was untouched.
Accounted prior cost remains USD0.32710677365251912; conservative reservations
remain USD3.9139916049382716. New spending is USD0.

Evidence: `experiments/data/mtg213/gpu-v9-preflight-20261104/`, including
prepared-run, inventory, refusal logs, launch scripts and outcome. Local
operational scripts remain in `.runs/mtg213/v9-continuation/`. The named blocker
is provider capacity within the existing price limits, not an implementation
failure or a need for new product direction. The accepted outcome is incomplete.

Validation: all 15 preflight files published to the artifact store;
`uv run --extra dev pytest tests/infra/test_experiment_data_untracked.py -q`
passed (1 test), `git diff --check` passed, and scratch is within both context
limits. No background process or paid rental was started by this continuation.

## Remaining implementation work

Continue from the prepared v9 Run and original 134-deck population. Its source
is frozen; any executable amendment needs a successor Run. Never launch or rebind superseded
v6/v7/v8 prepared Runs. Reconcile current provider inventory, billing and
compatible quotes before one allocation within USD1 and cumulative USD10.
The recorded USD0.32710677365251912 accounted cost and
USD3.9139916049382716 conservative reservations are historical, not fresh bills
or supply evidence. Existing nested Doppler credentials and published isolated
Etude 0.2.14 control client remain the documented route.

Execute the bounded pilot, retain admitted trained exports and actual learner
progress, verify teardown and whole-rental accounting, complete the paired
schedule, and report uncertainty and Elemental/world/hardware differences.
Keep failed attempts and missing historical slots unchanged. Any newly exposed
repair continues in this Task from main as a new PR, preserving #423 as history.
No unrelated follow-up Task is needed for this in-scope acceptance gap.

No paid allocation, server or watcher was started. Documentation-only compression
reuses the 196 passing focused checks; gate owns any further acceptance checks.
`git diff --check` passed; `lf context` reports memory and scratch below both
token and byte limits. No runtime tests rerun for these prose-only changes.
