# TLA pilot continuation — 2026-11-04 UTC

**Fix forward.** MTG-213 remains incomplete: v3/256 failed at 270 distinct facts
before its first GPU learner update. No trained endpoint exists and 256 scheduled
evaluation games remain missing. The approved repair belongs to Task 5da5f4 /
PR #419; consume its verified delivery before binding a fresh pilot. An unchanged
v3 retry cannot satisfy acceptance.

## Accepted outcome and constraints

Jack Heart's 2026-11-03 direction selects preliminary self-play and evaluation
on independently sampled trophy decks from the same frozen 134-entry population.
Acceptance requires GPU learner updates, an admitted trained checkpoint, complete
paired evaluation and an honest comparison of learning per decision/compute/wall
time with the evidenced Elemental setup. An honest negative result is possible;
raw throughput and initialization scores do not establish learning. Live fetching
and held-out-deck generalization remain deferred.

The cumulative $10 paid-compute cap includes setup, failures, retries and cleanup;
shipping enabling changes after checks is authorized. Preserve v1/v2/v3 source,
protocols, attempts and evidence. No truncation, deck exclusion, failed-game
replacement or inherited budget applies. Attribution and earlier engineering
choices remain in `scratch/questions.md`; results and accounting belong to
`experiments/tla-trophy-pilot.md`.

## Retained evidence

PR #416 merged as `3b122de9078c985f846a0f3c5a3a42ebca3b1182`; its tree
`3e696c396defb93716b3a428a1bef66b9abe71ab` equals pre-merge source `7b8bb3273`.
CI run 41752523278 and 38 focused corpus/pilot/discovery tests passed. Etude froze
that merged source for run `c216c6be7037`, cohort `mtg213-pilot-v3`. Subsequent
checkout changes retain results and context; they are not its source pin.
The 05:50–05:51 UTC verification found remote main still at that merge and no
checkout differences in `manabot/`, `managym/`, v3 runner, tests or dependencies.

The runtime is w17/card digest
`dcb9d47c674a657176c2a9d56e016288d1e175729aa09cd60f0dc333633330ed`,
with v6/256, unchanged learning settings, seeds and eight-pair/four-leg schedule.
`experiments/tla_pilot_v3.py:frozen` enforces this contract. PR #415's later
306-list admission audit includes all 134 original entries but does not expand
the pilot population.

The manifest retains all 120 artifacts under
`experiments/data/mtg213/gpu-v3-20261104/`; their SHA256 hashes and byte sizes
were reverified at 05:50 UTC. CUDA checks passed, but collection failed before
learning: 171 completed-but-unoptimized games, 86 active stream prefixes and a
270-row overflow reproduced in scalar/vector/Python encoders. Nine Drill
activations over fifteen permanents produce two keyword-removal facts each.
Diagnostic parity at 512 covers only this counterexample. Fresh initialization
completed 32 valid baseline games (3 wins / 29 losses, all replay-passing);
analysis retains all 288 expected games, `complete=false`, `success=null`.
Advertised `etude-result.json` was absent. These receipts reject completion
without another paid retry or repetition of the failed collection workload.

## Budget and execution ownership

Etude 0.2.13 via isolated `uv tool run` restored inventory decoding of
`input_undelivered`; the frozen worker retained 0.2.12. Provider access uses
nested Doppler `manabot/prd` (artifacts), then `etude/prd` (RunPod).
L4 `l4-762887e71be1` / pod `z84dbb8r0fw52v` cost $0.59/hour, with a $0.7375
maximum reservation and $1 cohort ceiling. Its existing owner settled under
the supply lock and confirmed deletion at 02:26:33 UTC after a network outage
delayed cleanup. Unrelated provider inventory was left untouched.

Prior provider bills total $0.07092813; the full v3 rental estimate is $0.09958535,
including idle cleanup. Cumulative billed-plus-estimated compute is $0.17051348;
the smaller per-run estimate ($0.03342127) omits some rental time. Retain all
three reservations ($2.2125) against $10 until final billing. These are dated
settlement observations, not fresh availability or final v3 billing. The report
links the receipts and documents the outage's evaluation fallback. Local scripts
and intermediate receipts remain in `.runs/mtg213/v3-continuation/`.

## Repair dependency

Comment 42a12145-a070-4afb-8735-4d7bf56116a9 relays Jack Heart's approval to
ship shared-effect fixed tensors: encode effects once and route them to affected
object embeddings before attention. Task 5da5f4 owns implementation and landing;
the repair choice is settled. Do not duplicate its work or substitute generic
v6's 512 default as the next pilot contract.

PR #419 merged at 2026-11-04T05:55:57Z as `fb5ad2c1372f9c967b550f0b8ae0f84883f75563`, with all executed CI checks passing. Its delivered implementation is integrated in this checkout.

The reviewed PR description requires explicit `policy_input_version=7`; fresh
defaults remain v6. Retained cases use 7/9 complete payloads and 70/135 memberships,
with bounded overflow still fatal. The prospective binding must include the
delivered serialized capacities. Lossless
factoring preserves payloads, memberships and multiplicity; redundant-activation
merging needs separate proof of identical future behavior, including expiry,
source dependencies and intervening effects. CPU learner/export checks do not
establish CUDA behavior. The PR reports 37,888 padded input bytes at v7 defaults
versus 12,288 for v6/256; factoring alone does not establish lower memory use or
faster learning. It also retains an existing keyword grant/removal ordering
defect. The repair preserves distinct records rather than proving that redundant
activations can be merged; that limitation remains relevant to interpretation
of pilot outcomes, without changing the frozen deck population.

## Remaining acceptance

1. After verified delivery, bind a prospective protocol/regime/runner identity
   with explicit source/world/configuration, fresh initialization, retained
   140/270-row replay parity, frozen-version compatibility and capacity-boundary
   checks. All shared observation paths and learning inputs must agree.
2. Move the analysis and discovery consumers with the new identity, then freeze
   the exact verified source. Reconcile inventory, provider bills and reservations
   against the same cap; allocate through the existing launcher owner with bounded
   deadlines and retained teardown. Pass the ordinary GPU numerical gate.
3. Observe learner updates and send the requested milestone with job/GPU/progress,
   rate and deadline evidence. Admit trained checkpoints and complete the new
   evaluation schedule; new games cannot fill historical v3 result slots.
4. Report learning gains, uncertainty, errors, total cost and Elemental-relative
   pace limits. Shipping code or green tests alone do not complete this outcome.

## Prospective migration scope

The analysis and pilot tests still import v3, and experiment discovery still
lists it. Adoption therefore includes coordinated changes to
`experiments/tla_pilot_analysis.py`, `tests/eval/test_tla_pilot.py` and
`tests/eval/test_experiment_files.py`, retaining schedule completeness and
missing-result accounting. Frozen protocols, regimes, runners, receipts and
compatibility tests remain historical inputs. The trainer, evaluator and rental
owner remain shared; Task 5da5f4 owns the representation implementation.

Earlier committed context is at
`c10fba26a790aa0286d7ea84b1e45edd2b4a9c06:scratch/tla-pilot-verify.md`.
The complete uncommitted verification/implementation notes are preserved at
`.runs/mtg213/context-before-compress/verify-before-compress-20261104-0551.md`.

Check: rebuilt native extension; 17 shared-transport tests (retained 140/270 parity, v1–v6 compatibility and all shared capacities), 28 pilot tests and exact discovery inventory pass. Initial identity errors were resolved by admitting new source paths to git tracking. Full affected suites remain with gate; paid execution retains the ordinary CUDA numerical gate.

## Delete — do not maintain

Replace the active `experiments/tla_pilot_v3.py` runner with v4 and move analysis/discovery/tests/docs together. V3 execution and analysis remain reproducible from `3b122de9078c985f846a0f3c5a3a42ebca3b1182`; preserve all v1/v2/v3 protocol/regime bytes and evidence. No representation implementation is replaced here.

V4 implementation now binds input v7 and all serialized shapes; population/recipe/schedule are preserved and predecessor rejection covers v1–v3. Fresh 05:59 UTC inventory confirms all three pilot rentals ended and no provider pods; final prior bills total $0.17054830933921039. Conservatively keep $2.2125 of prior reservations held; the next reservation must fit $1 and cumulative $10. The only unsettled session is the unrelated non-rented `mini` session.
