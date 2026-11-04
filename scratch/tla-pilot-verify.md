# TLA pilot continuation — reconciled 2026-11-04 UTC

Accepted outcome: Jack Heart's 2026-11-03 direction selects preliminary TLA
self-play and paired evaluation on the same frozen trophy population, with
GPUs and enabling repairs authorized under a cumulative $10 cap. The corpus,
runners and capacity repairs are delivered. GPU learner updates, an admitted
trained checkpoint, complete evaluation and an Elemental-relative learning/cost
report remain outstanding. No product decision is missing; the latest retained
capacity read is negative. Detailed accepted choices remain in `questions.md`.

## Delivered repair and validation

PR #420 merged at 08:15:17 UTC as
`0a5f2ad0fb55030f2b8caa2d7e403e3a3876dd8a`. The implementation matches that
merged commit; this checkout also contains continuation notes. The active v5
regime retains 134 independently sampled entries, explicit input v7 and
`max_cards_per_player=128`. This is a prospective architecture with fresh
initialization; it does not resume v4's checkpoint.

The 09:02 UTC source inspection found the pilot runner, regime, protocol and
vector capacity binding unchanged on main at
`5dd11cec4100aa0e1f4a23c063dfd5778e2ba28f`. That main also contains later
TLA rules/content changes; the checks below do not validate its newer runtime.

Check: `uv run --extra dev --extra artifacts --extra evaluation pytest tests/eval/test_tla_pilot.py tests/verify/test_shared_effect_transport.py 'tests/model/test_shared_effects.py::test_shared_effect_policy_uses_ordinary_learner_and_export[128]' tests/eval/test_experiment_files.py::test_etude_finds_each_current_experiment_and_nothing_else -q` — 50 passed in 22.18s; retained result reused for unchanged implementation.

Coverage includes the retained 61-card prefix at capacities 61/128 and failure
at 60, native scalar/vector/Python parity, shared-effect repair of the 140/270
fact failures, frozen versions, v5 population/protocol binding, discovery,
incomplete-result accounting and a CPU update with checkpoint export/reload.
Full gate results remain in `experiments/tla-trophy-pilot.md`; this continuation
did not rerun the full Rust/Python gate or CUDA admission.

## Latest recorded inventory and capacity

Authenticated Etude 0.2.13 reads used nested Doppler `manabot/prd` then
`etude/prd`. At 09:03:38 UTC all four pilot rentals had ended, no provider pods
remained and no hardware rental was unsettled. The unsettled `mini` session has
no hardware and is outside this pilot. Provider bills for all four rentals total
**$0.213831719709560274**, superseding the earlier billed-plus-estimated amount.
The conservative **$2.95 reservation accounting remains against the $10 cap**.

A6000 was rentable at 08:17:09 UTC, but subsequent reads found all four GPUs
unavailable at 08:19 and again at 09:03:37 UTC.
Recorded rates were $0.59/hour for A6000/L4 and $0.25/hour for A4000/A4500.
Availability is not a reservation. No v5 submission or rental is recorded.

Receipts: `.runs/mtg213/v5-continuation/inventory-20261104-0903.json` and
`.runs/mtg213/v5-continuation/capacity-check-20261104-0903.json`. The complete
pre-consolidation notes and report are preserved in
`.runs/mtg213/context-before-compress/20261104-0903/`. Earlier committed
continuation context is at
`8bd685ee02a9dd31588911b66a2713f4aec037f6:scratch/tla-pilot-verify.md`.

## Remaining execution and acceptance

Delete — do not maintain: duplicated verification-pass narration. Keep one
current execution state here; frozen runs, contracts and evidence remain retained.
There are no code deletion targets.

Prepared v5 Run `e8c76bc2c06f` remains bound to frozen source
`97a2cfb963d7e3a3b2816f31c45dda72f34cdddd`. Execution depends on fresh inventory,
billing, availability and quote reconciliation; the next allocation ceiling is
$1 within the cumulative $10 cap. The existing launcher owns deadlines and
teardown, with CUDA admission before training and hardware choice recorded
before scoring. Publication of newer code does not retarget this Run.

Jack Heart's requested launch milestone requires observed GPU learner updates,
with run/GPU/progress/rate/deadline evidence. Acceptance also requires an admitted
trained checkpoint, complete prospective paired evaluation, full costs and an
honest Elemental-relative learning/cost report.
The v4 summary has zero updates and 238 completed-but-unoptimized games before
`agent_cards plus sideboard 61 > 60`. Its evaluation has only initialization:
32 valid games, 256 missing, complete=false, success=null. Merge and CPU smoke
do not satisfy the outstanding acceptance; prior identities and missing slots
remain frozen. Any new workflow defect requires its own repair and validation;
merged #420 remains delivered history. The separate Etude Vast.ai Task EXP-35
does not establish available hardware or transfer a budget to this pilot.
