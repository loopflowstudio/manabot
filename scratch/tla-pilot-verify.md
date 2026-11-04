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
(see below); its concrete layout and capacities are that owner's engineering
choices, with no prospective pilot binding yet. V1/v2/v3 source, protocols,
attempts and evidence stay frozen;
no truncation, deck exclusion, failed-game replacement or inherited budget applies.

## Completed implementation and execution

PR #416 merged as `3b122de9078c985f846a0f3c5a3a42ebca3b1182`; its tree
`3e696c396defb93716b3a428a1bef66b9abe71ab` equals pre-merge source `7b8bb3273`.
CI run 41752523278 and 38 focused corpus/pilot/discovery tests passed. Etude froze
that merged source for run `c216c6be7037`, cohort `mtg213-pilot-v3`. Subsequent
checkout changes retain the results and reconcile context; they are not its pin.
Verification on 2026-11-04 at 05:01 UTC confirmed that fetched main still
resolved to this merge and its runner, protocol, regime, implementation and
tests had no diff from this checkout. GitHub confirmed PR #416 merged.

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
artifact-published in the manifest; the 05:01 UTC verification recomputed all
120 SHA256 hashes and byte sizes successfully. Advertised `etude-result.json`
was absent. These results establish launch and failure progress, but collection
still fails before learning under the merged contract.
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
embeddings before attention. Task 5da5f4 owns this bounded prerequisite. The
latest retained dependency read (`lf task status 5da5f4 --json`) showed no PR,
an active "Simplify shared effects" Session and its code Graph at `compress`
(updated 2026-11-04 04:56:20 UTC). Recheck delivery before binding the next
pilot; do not duplicate that owner's implementation. This is an execution
dependency with accepted direction, not an unresolved product decision.
Generic v6's 512 default does not select the prospective contract, and
unchanged v3 retries cannot satisfy acceptance.

Lossless factoring preserves complete payloads and explicit memberships;
the retained projection roundtrips as nine payloads and 135 memberships.
Redundant-activation merging requires separate proof of identical future
behavior, including expiry, source dependencies and intervening effects.
The existing trainer, evaluator and allocation owner remain the execution path.

## Delete — do not maintain

Remove duplicate verification chronology from this note; keep dated evidence
beside the result or dependency it establishes. No pilot predecessor is a
deletion target: frozen v1/v2/v3 protocols, runners and evidence remain
reproducibility inputs. The representation owner controls its own deletion cut.

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

No prospective binding, GPU allocation or code change was made pending delivery.
The pre-consolidation note is retained locally at
`.runs/mtg213/context-before-compress/verify-before-consolidation-20261104.md`;
earlier committed context is at
`8b1259d7308bf37f252b413ebbfb7014301b225b:scratch/tla-pilot-verify.md`.

Check: `git diff --check` passed; prior 120-file manifest verification, artifact-tracking test and 38-test corpus/pilot/discovery result reused (documentation only); gate owns prospective contract checks after Task 5da5f4 delivers.
