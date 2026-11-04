# TLA pilot continuation — reconciled 2026-11-04 UTC

MTG-213 remains incomplete. V3/256 failed with 270 distinct facts before its
first GPU learner update; no trained endpoint exists and 256 scheduled evaluation
games are missing. Inventory access, v3 source freezing and the bounded launch
requested by the earlier iteration feedback have happened. Repeating those
steps unchanged would repeat a known failed contract.

## Accepted outcome and constraints

Jack Heart's 2026-11-03 direction selects preliminary self-play and evaluation
on independently sampled trophy decks from the same frozen 134-entry population.
It includes actual GPU learner updates, an admitted trained checkpoint, complete
paired evaluation and an honest comparison of learning per decision/compute/wall
time with the evidenced Elemental setup. An honest negative result is possible;
raw simulation throughput and initialization scores do not establish learning.
Live fetching and held-out-deck generalization remain deferred.

His $10 cumulative paid-compute cap includes setup, failures, retries and cleanup;
shipping enabling changes after checks is authorized. He confirmed the immediate
larger-capacity hotfix, with the operator selecting 256 after his question.
The accepted next repair is shared-effect fixed tensors, owned by Task 5da5f4
(see below); no later numeric capacity choice is accepted. V1/v2/v3 source, protocols, attempts and evidence stay frozen;
no truncation, deck exclusion, failed-game replacement or inherited budget applies.

## Completed implementation and execution

PR #416 merged as `3b122de9078c985f846a0f3c5a3a42ebca3b1182`; its tree
`3e696c396defb93716b3a428a1bef66b9abe71ab` equals pre-merge source `7b8bb3273`.
CI run 41752523278 and 38 focused corpus/pilot/discovery tests passed. Etude froze
that merged source for run `c216c6be7037`, cohort `mtg213-pilot-v3`. Subsequent
checkout changes retain the results and reconcile context; they are not its pin.

The v3 runtime is w17/card digest
`dcb9d47c674a657176c2a9d56e016288d1e175729aa09cd60f0dc333633330ed`,
with v6/256, unchanged learning settings, seeds and eight-pair/four-leg schedule.
`experiments/tla_pilot_v3.py:frozen` enforces that exact input configuration.
PR #415 expanded admission: a read-only audit now admits 306 of the original
1,000 lists, including all 134 original entries. The pilot population is unchanged.

`experiments/tla-trophy-pilot.md` owns the result narrative and accounting.
Its linked `experiments/data/mtg213/gpu-v3-20261104/` receipts retain CUDA checks,
171 completed-but-unoptimized games, 86 active stream prefixes, and the 270-row
scalar/vector/Python overflow replay. Nine Drill activations over fifteen
permanents create two distinct keyword-removal facts each. Diagnostic parity at
512 covers this counterexample only. Fresh initialization completed 32 valid
baseline games (3 wins / 29 losses, all replay-passing); analysis retains all
288 expected games, `complete=false`, `success=null`. All 120 evidence files are
artifact-published in the manifest; advertised `etude-result.json` was absent.
Local scripts and intermediate receipts remain in `.runs/mtg213/v3-continuation/`.

## Operational evidence and remaining budget

Etude 0.2.13 via isolated `uv tool run` restored inventory decoding of
`input_undelivered`; the frozen worker retained 0.2.12. Provider access used
nested Doppler `manabot/prd` (artifacts), then `etude/prd` (RunPod).
L4 `l4-762887e71be1` / pod `z84dbb8r0fw52v` cost $0.59/hour with a $0.7375
maximum reservation and $1 cohort ceiling within the existing $10 cap.

A network outage stalled cloud evaluation discovery before any match and delayed
cleanup. The canonical CLI evaluated a byte copy of the run's tracking prefix;
no failed match was retried. The existing L4 owner settled under its supply lock
and confirmed deletion at 02:26:33 UTC. The unrelated provider pod and mini supply
were left untouched. All local processes started by that continuation ended.

Prior provider bills total $0.07092813; the full v3 rental estimate is $0.09958535,
including idle cleanup. Cumulative billed-plus-estimated compute is $0.17051348;
the smaller per-run estimate ($0.03342127) omits some rental time. All three
reservations ($2.2125) remain held against $10 until final billing. These are
retained settlement observations, not fresh availability or final v3 billing.

## Selected repair and remaining work

Comment 42a12145-a070-4afb-8735-4d7bf56116a9 supersedes the unresolved
512-versus-representation choice: Jack Heart approved shipping shared-effect
fixed tensors, encoding effects once and routing them to affected object
embeddings before attention. Task 5da5f4 owns this bounded prerequisite. Do not
duplicate its live implementation or retry v3 unchanged. The last dependency
read (`lf task status 5da5f4 --json`, 2026-11-04 04:14–04:16 UTC) showed an
active implementation Session and no PR. A shipping retry was not delivery.
Recheck delivery before binding a prospective pilot; generic v6's 512 default
does not choose its contract.

## Delete — do not maintain

No product path is replaced in this continuation. The shared-effect owner
removes the representation paths it replaces. Preserve historical v1/v2/v3
protocols, runners and evidence; they are reproducibility inputs, not deletion
targets. Keep the existing trainer, evaluator and allocation owner for the
prospective pilot. There is no additional local deletion target.

## Remaining acceptance

After delivery of that prerequisite, remaining acceptance consists of:

1. A prospective protocol/regime/runner identity, explicit source/world binding,
   fresh initialization and retained 140/270-row replay parity, frozen-version
   compatibility and capacity-boundary checks. A representation change also needs
   every shared observation path and learning input to agree before execution.
2. Fresh inventory and provider-bill reconciliation against the same cumulative
   cap, followed by an exact verified source freeze and bounded GPU allocation
   through the existing launcher owner, with deadlines and teardown retained.
3. Observed learner updates and the requested progress milestone, then admitted
   trained checkpoints and the complete prospective evaluation schedule. V3's
   missing games remain missing; a new run cannot fill historical result slots.
4. Reported learning gains, uncertainty, errors, total cost and Elemental-relative
   pace limits. Shipping evidence or green implementation tests alone do not
   complete the training outcome.

Verification established that main and PR #416's runner, regime and protocol
match the failed v3/256 execution. Manifest hashes and sizes for `summary.json`,
`capacity-replay.json`, `analysis.json` and `settlement.json` matched. Decision:
fix forward after the shared-effect repair; merge alone did not complete MTG-213.

Checks: `git diff --check` passed; manifest SHA256/size verification passed for all 120 added files; `uv run --extra dev pytest tests/infra/test_experiment_data_untracked.py -q` passed (1 test); prior 38-test result reused, no executable behavior changed. Gate owns prospective pilot checks after Task 5da5f4 delivers.
