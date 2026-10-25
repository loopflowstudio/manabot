# MTG-183 live plan

Frozen protocol: `experiments/protocols/elemental-183.md` (the commit that adds it),
with its amendments (0.03 saturation margin, 2026-10-24 02:10 UTC; "continue on
Etude after w2", 2026-10-24 19:00 UTC). It stays the authority; this note tracks state.

## Decisions

- Jack Heart, 2026-10-23: launch one long seed within the $50 cap.
- Jack Heart, 2026-10-24 ~18:55 UTC: after the migration retired the cohort
  controller, continue the same seed past w2 on Etude with the remaining cap;
  one learner lineage, no new seed. Build continuation on #362; allocate nothing
  until #362 is on main; do not touch w2's pod, MTG-181's follower or S3 folders.

## Delete — do not maintain

- Cohort windows/CLI and the follower's S3/cohort admission: deleted with #362.
  Frozen job specs `experiments/protocols/elemental-183-job-*.json` stay as w1/w2 evidence.
- `experiments/elemental_long_follow.py` and its test: deleted (2026-10-25),
  replaced by `experiments/elemental_long_eval.py` (Etude eval). The pinned
  checkout `/Users/jack/.manabot/mtg183-source` still runs the old follower
  until cutover; its ledger is read-only input for w1/w2 checkpoints.
- `experiments/elemental_grid.py` is MTG-181's unmerged bytes, copied unchanged;
  whichever branch lands second keeps one copy.

## Exists (2026-10-25 02:15 UTC)

- Branch merged main (#362, heuristic v2 `639ab7770`) at `261c62027`, then #360
  (`arena/` split into `sim/` + `eval/`, etude-sdk 0.2.2) at `9871e3545` (pushed).
- Seed 18301: w1 ended at update 1,325, w2 at 2,534 (below). Old cohort
  controller unloaded.
- Old follower `gg.manabot.mtg183.follower` stopped (`launchctl bootout`,
  2026-10-25 00:15 UTC; plist kept). MTG-181's follower had exited and MTG-181
  moved to `elemental_181_new_heuristic`, which the follower's process-name yield
  does not match, so at 23:49 UTC it began `heuristic-000000-full` against
  heuristic v1 and stopped admitting exports mid-unit. Those partial v1 groups
  are not evidence. Ledger `/Users/jack/.manabot/mtg183-follow/follow/ledger.json`
  (copy `ledger-before-stop-20261025T0015Z.json` beside it): 48 exports admitted, 0…2,200+.
- `experiments/elemental_long_eval.py`: one Etude eval run (`mtg183-elemental-eval`
  / `mtg183-eval`), one match per grid row, tables delivered to
  `r2://manabot/evaluations/`; checkpoints from the old ledger (w1/w2) and w3's
  Etude run; `<directory>/state.json`. Report reads `statsdb` + state.
  R2 delivery and `statsdb` read-back verified with a throwaway prefix (deleted);
  one real game each of heuristic/subset/pair played through `play`.
- Protocol amendment "heuristic v2, Etude eval, R2" (2026-10-25 00:00 UTC).
- `fetch_etude_checkpoint` takes an `ArtifactRef` and reads S3 or R2.

## Etude launcher (2026-10-25 00:20 UTC)

`~/.etude/launcher.db` was schema 1 (2026-10-15) with 0 runs/rentals/sessions;
moved aside to `launcher.db.schema1-20261025`, and 0.2.1 created schema 10.
Registered: storage `r2` = `r2://manabot/jobs`; provider `runpod` (storage r2);
on-demand `l4` (NVIDIA L4, 6 vCPU, 48 GB, 50 GB disk, max 1, ≤ $0.60/h, w1/w2's
image digest `cb154fcc…`, startup 900 s, delivery 300 s). Keys: `RUNPOD_API_KEY`
from Doppler `etude/prd`, R2 (incl. `R2_API_TOKEN`) from `manabot/prd`.
`etude availability --rent l4`: $0.59/h, rentable (00:17 UTC; again under 0.2.2
at 01:26 UTC, run as `doppler run -p etude -c prd -- doppler run -p manabot -c prd
--preserve-env=RUNPOD_API_KEY -- uv run etude …`).

## Evaluator (running since 2026-10-25 00:22 UTC; re-pinned ~01:25 UTC)

launchd `gg.manabot.mtg183.eval` (plist in `~/Library/LaunchAgents`) runs
`elemental_long_eval` under `doppler run -p manabot -c prd` from the pinned
checkout `/Users/jack/.manabot/mtg183-eval-source` (`9871e3545`, etude-sdk 0.2.2,
`_managym` from this worktree's #360 build), directory
`/Users/jack/.manabot/mtg183-eval`, `--windows` the old ledger (re-read every
poll), `--storage r2://manabot/evaluations`, `--until 1793584946`. It yields
(silently; `state.json` refreshes each poll) while any experiment from
MTG-181's checkout runs, so its wall time follows MTG-181, not idle capacity.
The pin moved from `d9fc58df2` with 0 rows played because #360 changed
`INFERENCE_CODE` (agent identity). Later restarts (adding `--etude-source/--run`)
keep identities only if the pin's `INFERENCE_CODE` files are unchanged; rows
played under different pins are not comparable.

## w2 end and w3 spec (2026-10-25 02:06 UTC)

- w2 ended 01:47 UTC at update 2,534 (`paused`), $7.2358. One-shot `_admit`
  (old checkout, under `owner.lock`) took the ledger to 53 exports, 0…2,534
  (copy `ledger-before-tail-admit-20261025T0150Z.json`).
- w2's run.json/raw/EMA/Adam fetched to `~/.manabot/mtg183-w2-final/parent`,
  staged to `r2://manabot/jobs/mtg183/w2/`, each verified by sha256.
- Staging first bound update 1,325: #365 made `StageRecord.updates` explicit and
  `TrainingRun` read w1/w2's records as 0 updates (the worker's admission would
  also have refused the parent). Fixed in `418eb456c` (rule moved from
  `RetainedRun` to `RunRecord`; tagged `mtg183-w3-source`); local admission of
  the real w2 record then gave iteration 2,534.
- Etude caps a rental at 43,000 s, so one 57.7 h w3 cannot run (worker would
  refuse its deadline). Amendment "segments of one rental each" (02:06 UTC,
  its commit time; first stamped a future 02:30 UTC, corrected before allocation;
  tag `mtg183-w3-spec` = `4eaa59988` still carries the old stamp):
  a chain w3, w4, … sized from receipts. w3: 11.6 h, $7.11, target 3,500;
  quoted $6.99 at $0.59/h. Spec `experiments/protocols/elemental-183-w3.json`.
- Evaluator `state.json` (02:06 UTC poll) holds all 53 exports; still 0 rows
  (yielding to MTG-181's `elemental_181_new_heuristic`). The ledger's `error`
  lists ValueErrors for the old cohort's never-run w3–w7 windows: historical.
- Launch as `Run(job, seed=18301)` (the worker needs `ETUDE_SEED`); MTG-186 holds
  the shared `l4` on-demand's one slot, so `etude ondemand max l4 2` before launch.

## w3 (training since 2026-10-25 ~04:00 UTC)

- Branch landed as #367 (squash `9ba18ded5`); branch merged main back.
- Etude run `e7d47d8b220f` (launched 03:55 UTC) trains from update 2,534
  (Etude clock absolute: 2,539 at 04:02). Source `3317a8dec`
  (tag `mtg183-w3-source-2`), spec re-frozen before learning; amendment
  "launches that never trained" (03:51 UTC). Rental ends ~15:50 UTC.
- Four earlier launches never trained (amendment lists causes): `a04845f2548b`
  (w1/w2 image lacks python3.12, ~$0.15), `9fe2a28ddbf3` (capacity, $0),
  `3828021cbc80` (inputs staged under `jobs/`; Etude reads bucket-root keys;
  Etude courier never retries a failed input fetch; pod deleted via RunPod
  API, ~$0.44), `cf7dec5040bc` (gate: no `pytest` without `--extra dev`, ~$0.01).
  Fixes: `l4` boots Etude's stock image `4d1721e6…`; inputs restaged at
  `r2://manabot/mtg183/w2/` (the `jobs/mtg183/w2/` copy is redundant; for the
  next segment stage with storage `r2://manabot`, prefix `mtg183/w<N>`); MTG-186's
  worker fixes ported verbatim (Rust setup, `--extra dev`, source bundle, ETUDE_
  env stripped from the gate suite).
- launchd: `gg.manabot.mtg183.controller` (`etude controller run`, KeepAlive);
  `gg.manabot.mtg183.w3-launch` (`~/.manabot/mtg183-controller/launch_w3.py`,
  `runs.json`) exited 0 once w3 stepped; it is w3-specific.
- Evaluator on the Mac mini (amendment 04:34 UTC): Etude run `ac6c45760d88`
  (pool `mini`, owned, $0; spec `experiments/protocols/elemental-183-eval-job.json`,
  revision `39b02876e`), started ~04:45 UTC, state in mini `~/.manabot/mtg183-eval`.
  R2 keys from a read-only Doppler service token `mtg183-eval-mini` (manabot/prd)
  in mini `~/.config/manabot/doppler-prd.token` (keychain unreachable over SSH;
  `--config-dir ~/.config/manabot/doppler`). w1/w2 checkpoints copied from the
  follower cache to R2 at their S3 keys (`jobs/<job>/runtime/artifacts/sha256/…`,
  53 read back and verified), listed in `elemental-183-windows.json`. Segment
  ends come from the tracking bucket's `state.json`, not Etude status. Admitted
  54 exports (…2,550). MTG-181's evaluation also runs on the mini; no yield.
  `failures/*.jsonl` in arena evidence are pre-game crash-safe logs, not failures.
- First mini match delivered 06:48 UTC (`r2://manabot/evaluations/mtg183-eval/`):
  500 valid games, u000000 0.112 vs heuristic v2 on its row; a unit (200 groups)
  takes ~2 h on the shared mini; admission polls only between units.
- Laptop evaluator stopped 04:44 UTC with 0 matches played, LaunchAgent removed
  06:50 UTC (plist kept at `~/.manabot/mtg183-eval/…removed-20261025`); the segment reader
  had read a local path named `r2:/…` (fixed `b668e92d2`), so w3 exports were
  first admitted at 04:40 UTC.
- Spend: w1 $7.2252 + w2 $7.2358 + non-training launches ≈ $0.60 + w3 ≤ $6.99.

## w3 end and w4 spec (2026-10-25 ~21:40 UTC)

- w3 stopped 14:54 UTC on its regime watchdog at 3,264 (`interrupted`, $6.31),
  no final exports (brief: `scratch/w4-restart.md`). Jack Heart: w4 continues
  w3's raw monitoring export at 3,250, EMA := raw, fresh Adam.
- Code `5d676d6df` (local tag `mtg183-w4-source`): raw-only
  `LearningStateImport.monitoring_update` / `LearnerContinuation.from_monitoring`
  / `stage --monitoring-update`; `execute_regime(stop_reserve_seconds)` ends a
  segment `paused` with raw/EMA/Adam before its watchdog; `TrainingJob` always
  uses it (default 120 s or a tenth of the watchdog) and reports success at the
  update reached. Real w3 bytes admitted locally (origin 3,250, resets recorded).
- Staged and read back: `r2://manabot/mtg183/w3/{run.json,monitor-00000014.pt}`
  (local copies in `~/.manabot/mtg183-w3-final/`).
- Amendment 21:39 UTC + `experiments/protocols/elemental-183-w4.json`: 46.0 h,
  max $27.32, target 5,950, quote $0.59/h at 21:29 UTC.
- Check: `tests/training/test_{etude_job,learning_state,pretrained}.py`,
  `tests/eval/test_elemental_long.py` → 43 passed (21:30 UTC). Gate owns the rest.

## Remains, in order

1. Launch w4 (operator session; not launched here): push branch + tag
   `mtg183-w4-source` (Etude bundles the revision), move the launchd controller
   to a 0.2.6 checkout, `etude ondemand max l4 2` if MTG-186 holds the slot,
   `Run` from `elemental-183-w4.json` with seed 18301.
2. Then relaunch the eval job (`mtg183-eval-cards`, currently `493438ce31d1`,
   revision `fb71050cf`) with `--run e7d47d8b220f --run <w4>` and no
   `--continues`. Relaunched never-trained w4 attempts replace its `--run`.
3. S3 cleanup below (w1/w2 checkpoint bytes already copied to R2 at their target keys).
4. Results report: mark the 3,250 discontinuity on the update axis (amendment
   21:39 UTC). Publish evidence with `manabot.data publish`, manifest, protocol
   Results pointer.
5. Publish this Task's follow-up commits as its next PR.

## Proposed Wave memory (not written; standalone realign leaves `.lf/wave/` read-only)

- One RunPod job was capped below twelve hours, so the old cohort chained
  `last-complete` continuations; on Etude, learner state continues through
  `TrainingJob.continuation` (raw/EMA/Adam + parent run record, refusing another
  parent update or seed; collector/RNG streams restart).
- Run long-lived followers from a detached pinned checkout so the Task branch can move.
- Yielding by another Task's process name fails silently when that Task
  switches scripts; MTG-183's follower then played a stale anchor and stopped
  admitting exports. Admission and play should not share one blocking loop.
- The report's synthetic test showed a strict "gain not significant" saturation
  rule never fires at 5,000 paired games; use a practical margin.
- Local evaluation that yields to another Task's follower can stall entirely;
  declare its expected wall time against the real queue, not idle capacity.

### S3 cleanup (Jack Heart via the Etude storage-cost session, 2026-10-24)

When w2 and both old-system followers have stopped: copy keepers (checkpoints
.pt, final run/run.json, latest manifest, small control files, Etude metrics)
from `s3://etudefantasia/manabot/jobs/mtg183-*` and
`s3://etudefantasia/manabot/cohorts/mtg183-elemental-long/` to R2 `manabot` at
the same relative path minus `manabot/`, verify by size/sha256, then delete
every version and delete marker of those folders. Report copies and bytes
deleted. w3 must stage w2's learner state first. Never start old-system runs.

### Direction (Jack Heart, 2026-10-24 ~22:30 UTC): R2 and new Etude code wherever possible

"Let's move over to R2 and the new Etude code as much as possible": w3 on Etude
storing to R2 (SDK 0.2.1), evaluation as Etude eval with tables in R2 (done:
`elemental_long_eval`, explicit `Match`es; the frozen schedules and the rating
fit unchanged), retire the bespoke follower (deleted), w1/w2 artifacts to R2
after w2 ends, report data via `manabot.data publish`.
**Anchor decided (Jack Heart, 2026-10-24): heuristic v2** (`639ab7770`); recorded
in the protocol amendment. v1 scores (MTG-181, ladder) are not comparable.

Check (2026-10-25 01:25 UTC, at `9871e3545`, load average ≈ 200 from MTG-181):
`tests/eval/test_elemental_long{,_report,_eval}.py`, `test_elemental_grid.py`,
`tests/training/test_etude_job.py` → 30 passed, 1 failed
(`test_worker_runs_real_learner_and_reports_admitted_policy`: its 30 s subprocess
timeout expired under that load; rerun when the machine is idle, gate owns it).

### Direction (Jack Heart, 2026-10-25 ~04:30 UTC): evaluation off the laptop, onto the Mac mini via Etude

"Can we stop using my laptop for evals" → chose the Mac mini via Etude ($0).
Register `mini-heart.local` (M1, 8 cores, 16 GB, always on) as an owned Etude
machine in a pool, run the MTG-183 evaluation there (Etude evaluation job or the
evaluator under Etude on that host; tables still in `r2://manabot/evaluations`,
same frozen schedules, agent identities and v2 anchor), and remove
`gg.manabot.mtg183.eval` from this laptop once the mini has taken over. The
laptop-only yield to MTG-181 then no longer applies. No paid CPU rental.

### Direction (Jack Heart, 2026-10-25): remove Etude's 12 h rental cap

"Etude caps a single rental at about 12 hours — why would we do that? Remove."
Filed as Etude task ee6e73 (code workflow launched). `OnDemand.max_rental_seconds`
= 43,000 is stale (grants: S3 36 h, R2 7 d). Once released and pinned, w4+ can
run as one rental to the remaining cap instead of ~11.6 h segments (protocol
amendment before that allocation).

## Direction (Jack Heart, 2026-10-25 ~14:20 UTC): card stats going forward

- Amendment `fb71050cf`: every eval game also delivers card tables
  (`trace_card_tables`, participants `model`/`heuristic` and `newer`/`older`),
  evidence under the run folder's `artifacts/`. Job spec `3d7bb2a54`.
- Current code's match keys differ from the 0.2.2 run's (0 of 110 overlap), so
  it is a new run `mtg183-eval-cards`, state `~/.manabot/mtg183-eval-cards` on
  the mini, replaying from the start (~8 h to catch up to where `mtg183-eval`
  was). `ac6c45760d88` cancelled ~14:35 UTC; `mtg183-eval` kept as superseded,
  a reproduction check for the replayed units.
- Smoke (2 grid + 2 pair games on real R2 checkpoints): card tables present,
  ~44 KB evidence/game.
- Launched with the pinned 0.2.2 client (`~/.manabot/mtg183-eval-source/.venv`)
  so the controller DB w3 depends on is untouched. The w4 relaunch keeps
  revision fb71050cf, adding `--run <w4>`.
