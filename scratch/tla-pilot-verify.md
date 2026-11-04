# TLA pilot continuation — 2026-11-04 08:19 UTC

Decision: **External capacity blocked** at the latest recorded read. No human
choice is missing. PR #420 delivered the card-capacity repair; preserve it and
continue the prepared Run when capacity permits. Do not replay publication or
refreeze its source.

## Delivered repair and validation

PR #420 merged at 08:15:17 UTC. Refreshed main is
`0a5f2ad0fb55030f2b8caa2d7e403e3a3876dd8a`; it and this checkout have the same
complete tracked tree, `631e68b4274c2952f7208e63664aa28f223295f4`.

```sh
uv run --extra dev --extra artifacts --extra evaluation pytest tests/eval/test_tla_pilot.py tests/verify/test_shared_effect_transport.py 'tests/model/test_shared_effects.py::test_shared_effect_policy_uses_ordinary_learner_and_export[128]' tests/eval/test_experiment_files.py::test_etude_finds_each_current_experiment_and_nothing_else -q
```

Result: **50 passed in 22.18 seconds**, reused because source is unchanged.
Coverage includes the retained 61-card prefix at capacities 61/128 and failure
at 60, native scalar/vector/Python parity, shared-effect repair of the 140/270
fact failures, frozen versions, v5 population/protocol binding, discovery,
incomplete-result accounting and a CPU update with checkpoint export/reload.
Full gate results remain in `experiments/tla-trophy-pilot.md`; this continuation
did not rerun the full Rust/Python gate or CUDA admission.

## Latest recorded inventory and capacity

Authenticated Etude 0.2.13 reads used nested Doppler `manabot/prd` then
`etude/prd`. At 08:19:03 UTC all four pilot rentals had ended, no provider pods
remained and no hardware rental was unsettled. The unsettled `mini` session has
no hardware and is outside this pilot. Provider bills for all four rentals total
**$0.213831719709560274**, superseding the earlier billed-plus-estimated amount.
Keep the conservative **$2.95 reservation accounting against the $10 cap**.

A6000 was rentable at 08:17:09 UTC, but subsequent reads found A6000 unavailable
at 08:19:07, L4 at 08:19:22, A4000 at 08:19:37 and A4500 at 08:19:38 UTC.
Recorded rates were $0.59/hour for A6000/L4 and $0.25/hour for A4000/A4500.
Availability is not a reservation. No quote, submission, rental, additional
spend or background command was started by this continuation.

Receipts: `.runs/mtg213/v5-continuation/inventory-20261104-0819.json` and
`.runs/mtg213/v5-continuation/capacity-check-20261104-0819.json`. The complete
pre-consolidation note is preserved at
`.runs/mtg213/context-before-compress/20261104-0820/tla-pilot-verify.md`.

## Remaining execution and acceptance

The Graph's execution continuation owns prepared v5 Run `e8c76bc2c06f`, frozen
source `97a2cfb963d7e3a3b2816f31c45dda72f34cdddd`. After new capacity evidence,
reconcile inventory, billing and a fresh quote; cap the next allocation at $1.
Use the existing launcher/deadline/teardown owner and the Run's CUDA gate.
Record hardware choice before scoring. Do not silently retarget to main.

Send Jack Heart's requested launch milestone only after observed GPU learner
updates. Retain an admitted trained checkpoint, complete prospective paired
evaluation, full costs and an honest Elemental-relative learning/cost report.
The v4 summary has zero updates and 238 completed-but-unoptimized games before
`agent_cards plus sideboard 61 > 60`. Its evaluation has only initialization:
32 valid games, 256 missing, complete=false, success=null. Merge and CPU smoke
do not satisfy the outstanding acceptance; prior identities and missing slots
remain frozen. Any new workflow defect delivers through a new PR from main;
merged #420 stays as history.

## Delete — do not maintain

No code deletion targets: the tracked tree is already delivered. Consolidate
superseded operational notes here; retain experimental contracts and evidence.
