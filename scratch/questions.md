# c5fba4 open questions and assumptions — 2026-11-06

Material assumptions made headless, reversible until launch. The first implement
step's full list, including choices the design review superseded (w256/15×, the host
screen, equal compute at the endpoint only), is at `739f95b8d:scratch/questions.md`.
Inherited MTG-243 and 07c490 notes are at `8e5d152ab` and `e8da7909d` and confer
nothing here.

Still live from the first implement step and concept reviews 2–8:

- **Input v9 with 512 facts**: v1–v7 reject layer payloads TLA cards produce; v9
  carries naming identity. **Same learning rate (1e-4 max) for both sizes**, so a
  rate change does not confound scale.
- **16 pairs (64 games) per comparison** for ±~0.09; **cpu32 passes** (1.75 h inside
  USD2); **600 s per game**, sized on this Mac (a real fresh w256 match took about
  0.05 s a decision; the rehearsal's rented reading decides).
- **Calibration: five updates in a 3,600 s wall**, fresh policy, so late games may
  cost more. **Launch gate 1,500 s/update** for the large arm, a rough line; if the
  large arm costs little more than w64, a larger size inside 10–100× is a new version.
- **Failed legs bound a reading** (`worst`/`best`), not void it; first-64 cutoffs are
  reported, not an issue. **Exports every 25 updates**, milestones 75 and 150.
- **Checkpoint tag names a size** (`TrainingRun.job_name` would need an Etude release);
  **SIGTERM ends every `TrainingJob` `interrupted`**; **launcher status closes a lost
  run in manabot** and **`C5FBA4_PASS_ATTEMPT` renames a relaunched pass** (Etude
  `observe`/`advance` are the better owners).
- **Rehearsal in calibration, gated at 300 s**, all 128 games stored and valid;
  `startup_seconds` 900 on every rented on-demand. The USD evaluation cap is an
  operator check.

## Design revision and implement passes (2026-11-06, headless)

- **Four producers for both arms**; the accepted one-producer w64 calibration pod
  measures what they buy and gates nothing. Calibration launches with `--parallel 3`;
  ceilings USD0.55 / 0.90 / 0.55 (w64, w160l8, w64 one producer), USD2 total; launched at `--parallel 2`, since `l4` allows two machines.
- **Horizon S = min(10,000, ⌊194,400 / s⌋)**, s = calibration's median s/update over
  updates 2–5, no slowdown allowance. Provisional 1,296 (w64, in its regime) and 129
  (w160l8, in the protocol's `large.expected_updates`) until calibration.
- **Update cap 10,000** so the wall binds; milestones extend to 6400.
- **Equal-hours pairs: w64's last checkpoint at or before w160l8's hours**, from Etude
  `Checkpoint.wall_ms` through one source (`checkpoint_hours`) for `eval` and analysis.
  The Etude PR is filed by the operator conversation (steer a6c38666). An equal-hours
  endpoint pair after w64 ended earlier is unreached. Scale stays the endpoint
  head-to-head when both finish at the wall.
- **Evaluation USD14, reserve USD10.**
- **Defaults live in recipe builders, not Pydantic field defaults**; `sampled_self_play`
  uses game-error threshold .25 (v10's).
- **Compute parameters after consolidation exclude embedding lookup tables**; report
  both counts if shared embeddings blur the line. Outside 10–100× returns the size to
  review. **Policy input stays v9** unless 30df0c's merge moves it.
- **The 25% rolling game-error stop stays** beside `recover_capacity`; no calibration
  gate on overflow counts.
- **Fact-embedding check: 100 updates, 9,000 s wall, USD2/3 h per arm**, three-rule
  criterion (protocol item 5); value loss is the learning curve.
- **A run lost below update 25 is "complete" in scheduling only**: every slot is played
  or unreached; `complete` stays False because "A run did not finish" (direction 26).

# Inherited from b015d5 (PR #435, merged)

Its notes are at `54b109aff:scratch/questions.md`. One live item:
`test_resource_limit_receipt_replays_its_retained_encoding_boundary` fails on a
schema 7 fixture frame against native version 8 (`binding_mismatch`); it is in no CI
selection and needs its own bug Task.

## Gate (2026-11-07, headless)

- **The fact-embedding check runs from the merged #439 revision with the regime's
  provisional horizon**, not a second horizons merge. Concept review 15 said it needs the
  horizons; it does not. Inside the check's 84 s/update line S ≥ 2,314, so the rate stays
  clamped for all 100 updates and τ moves equally in both arms. Reversible: land the horizons
  first if the check should carry the exact freeze regime.

## Fact-contribution reading (implement, 2026-11-07, headless)

Reversible choices for Jack Heart's review:
- Positions come from seeded random-action play on the frozen pairs, not the
  heuristic or a checkpoint, so they are identical for every checkpoint and need
  no policy. Random play may under-represent late-game fact kinds; per-kind counts
  are reported so a thin kind is visible.
- The denominator is the object's embedding entering the fact encoder (after
  permanent links, before facts) for both fact and shared-effect ratios, as the
  direction says; shared effects are measured on objects named by a membership row.
- It runs on CPU where the analysis runs (no rental): about 1 s per w64 checkpoint
  on 495 positions here, so a full frontier read is minutes, not paid compute.
