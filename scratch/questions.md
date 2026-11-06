# Decisions and open questions — 5da5f4

## Current authority

- Latest, Jack Heart, 2026-11-05: "can we schedule soem evals in paralell with
  the additional training" selects recovery option A with parallel CPU evaluation,
  superseding the stops below for an amended cohort. Retain the three complete
  learners and failed attempt; restart pool/74212 from initialization on its
  original seed rather than reset optimizer/EMA halfway. Freeze the amendment
  before scoring. Original $25, seeds, horizon and criteria remain; interim
  results never guide training or selection. The plan owns scope and constraints.
  Fix launcher deletion before a new rental. PR84 and study merges are not approved.

- Jack Heart, 2026-11-05 after machine loss: "keep goign"; continue recovery
  investigation and useful work, preserving completed training and the original
  budget. He noted Vast may be interruptible and requested a separate Etude SDK
  design task for training robust to an interruptible compute layer. Captured as
  Etude inbox Task `ff3d48`, `loopflow://etude/task/ff3d48`; not launched.
  Audit result: the rental was on-demand, and our launcher deleted it after one
  non-running read. Completed learners restore; pool/74212 cannot continue
  exactly. Do not silently alter the study's trajectory or discard the completed
  learners.

- Jack Heart authorized actual GPU learning through the beginning of a sustained
  plateau within the **original cumulative $25**, including necessary Etude SDK
  work and Vast use. The completed CPU comparison is insufficient for architecture
  selection. His “start it when you can”, “keep going” and “thats ok lets try”
  authorized viable execution and bounded diagnosis, not a shorter scientific
  objective. Diagnostic/admission attempts remain capped at $0.50 each.
- His 2026-11-04 22:38 UTC direction requires investigating implementation before
  assigning host fault; his 23:15 continuation authorizes ordinary-learner
  profiling and measured repairs. No synthetic load or competing allocation.
- His “keep going” superseded the operator-created final-settlement-only gate.
  Completed quote caps are not permanent charges. The score-independent rule
  recorded before admission completed reserves max($3.50, twice the fresh sum of
  each ended rental's reported/lifetime maximum, including settled L4), plus $6
  ancillary and every unknown/live commitment in full. No invented settlement
  watermark, double counting or increased cap. The plan owns current accounting.
- Jack Heart selected public Etude Git main, pinned exactly in `uv.lock` for
  launcher and worker. Refresh before freezing a run, never during training.
  Historical source/lock/receipt bytes remain immutable; no SDK release is claimed.
- Results stop at demo/review. No study merge, model/default promotion, temporal
  memory implementation or MTG-213 spending. Earlier “ship it” approved PR419's
  shared-effect contract. Relation splitting, membership interning and semantic
  coalescing remain proposals. Coalescing needs identical future behavior across
  expiry, sources, affected sets, intervening effects and multiplicity. MTG-240
  owns the separate keyword-ordering defect; do not delete rules records.

## Current handoff and implementation decisions

The plan owns fresh scientific progress, accounting, evidence paths and remaining
acceptance. Run `e51f61b92931` at `af601877e` ended `lost / machine_lost` on
2026-11-05 at 13:09:07 UTC, with no final inventory or child exit code. Rental
`59194544` is absent and no Task rentals remain live. Three learners completed;
the fourth reached 83 updates in the last retained receipt. All 36 checkpoint
backups verify, but no curve/held-out endpoint survives. The plan's recovery
audit owns the deletion cause. The completed CUDA gates and matched throughput
qualified the launch, not a scientific result. New science needs the frozen
amendment and a launcher with the liveness repair.
Instance-only SSH attachment of the existing local public key enabled read-only
receipt/checkpoint collection; no source, recipe, deadline or provider account
changed. The diagnosis owns detailed measurement and delivery evidence.

The failed combined diagnostic lost completed traces when its rental disappeared.
End diagnosis before separate admission so Etude can deliver its inventory;
carry bounded parity/operator summaries in logs too. That boundary was verified
by `911423cce90b`. Missing outputs from earlier attempts remain unknown exposure
or cause, not zero work or proof of OOM. Preserve every failed attempt.

CPU profiles justified batching, but could not establish CUDA cost. Independent
float64 checks now bound the small float32 accumulation discrepancy without
changing gradients or learner semantics. Preserve token order, duplicate joins
and absent gradients for unused Adam parameters. First-step CUDA traces alone
do not justify another runtime repair or establish recurring throughput.

The 256-stream recipe changes optimization and therefore uses fresh identity
and paired seeds, preserving 65,536 games per learner and schedule progress by
exposure. Admission scores choose no treatment. Scientific horizon, evaluation,
plateau criterion and failed-attempt denominator remain fixed.

## Inherited boundaries and historical references

T7's published chooser and viewer authority survive the sync; control is not a
perspective switch. Preserve frozen widths and saved omissions. Synthetic
fixtures establish no runtime compatibility, WebSocket recovery or crash durability.
Full T7 decisions: `4463e57f5a92ebd3b60e2fb50c046b84ac2c0372:scratch/questions.md`.

MTG-213's v9/v10 choices remain pilot-specific: first-64 support for heuristic,
candidate and initialization anchor; permanent64 and opt-in per-game recovery
at .25 with 100/100 window/warmup. Native legality and Academy retain all actions;
omitted overflow/recovery policies still reject. This study retains 512 actions,
overflow rejection and its separate $25. Pilot failures and missing evaluations
are not representation-study results. Pilot direction/evidence:
`8e8d702f807d61675fc1bca46417eee69316daa0:scratch/questions.md`,
`ee85d6fc82ed631d7b460e68c73e7ac27f0458ff:scratch/questions.md`, and
`experiments/tla-trophy-pilot.md`. Changed runtime needs a prospective binding
even when recipe hashes agree.

Complete preceding notes: `2784e4283ce12011643b319968131ae5dbba9f6e:scratch/`.
Earlier SDK decisions: `075378dd87600428ad3efa60b7288b327c32cee7:scratch/questions.md`;
original design: `1b2ddcdfadcb09df76b2dda450152cbdfa60c8c2:scratch/`.
All removed detail is committed there.

Decision: one Etude Run per learner and per evaluation, now frozen in the amendment.
Assumption: evaluation on Vast hosts with an unused GPU counts as "CPU evaluation";
device is CPU and no training shares the machine.
Assumption: the first learner was dispatched beside the first measured endpoint,
not after all three retained evaluations began, to catch scarce 3090 supply.

Assumption: the `/verify` commit instruction was not applied to Etude PR #84,
because that skill checks merged work and this PR is an unmerged draft. The same
applies to commits on PR421's branch.

Assumption (compress, reversible through git): admission and CUDA profiling are
not required behavior once the cohort is complete; recovery units never used
them. A future amendment restores them from `d57c5c9e7` and freezes its own
identity. `runs()` now requires `--unit`. Retained job configs still carrying
`admission_limits` are historical and are never re-read by `StudyJob`.

## Main sync — MTG-213 context (2026-11-05 UTC)

MTG-213 decisions stay at `e113602a2dc1c866af513ff0953598db390db9f2:scratch/questions.md`
and in `experiments/tla-trophy-pilot.md`. That pilot and its separate $10 cap do
not replace this study's GPU run, $25 total or review boundary.

Decision (operator, 2026-11-05, before any pool/74212 score, reversible in
analysis): the restarted pool/74212 `initial_raw` differs from the retained one
only by host QR rounding in `orthogonal_` (≤1.04e-6 in 18 matrices; identical RNG
streams and other tensors), so it is admitted for scoring. Receipt:
`.runs/representation/recovery/init-comparison-typed_pool-74212.json`. The
paused evaluation `664f44738f35` stays an unscored attempt.

Etude defect exposed (2026-11-05 20:10 UTC, not fixed here): Vast answered two
creates with HTTP 400 (the $0.2022 Czech 3090 host, asks 591379xx–591380xx; a third on `recovery3090`). Etude records that as
"create reply unknown", and `VastMachines.find` raises while no labelled instance
exists, so sessions `1bd3d916419c`/`7da0af2dc646` never settle and their Run ids
stay blocked. Inventory showed no instance, so nothing was paid. These units were
recorded as never-started attempts (fresh unit identities follow). This belongs
to Etude (`ff3d48` / PR84 owners).

Assumption (realign, 2026-11-05): the `code` Graph does not call for Wave memory
maintenance, so `.lf/wave/` stays read-only here and the update remains proposed
in the plan. Replayed steers from earlier passes are history; none adds direction
beyond the completed recovery cohort.
