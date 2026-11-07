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
  ceilings USD0.45 / 1.10 / 0.45 (w64, w160l8, w64 one producer), USD2 total.
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

# Task b015d5 — training seconds per update (PR #435, merged `45cc961a2`)

History of the measurement and concept reviews: `git show 645b9be21:scratch/questions.md`.
Live items:
- Inherited, unrepaired: `test_resource_limit_receipt_replays_its_retained_encoding_boundary`
  fails because `tests/fixtures/tla-pilot-permanent-capacity.jsonl` retains a schema 7
  frame while native emits `SEMANTIC_DECISION_VERSION` 8 (`binding_mismatch`). The test
  is in no CI selection. It needs its own bug Task; the fixture is retained evidence.
- `HostFacts.cpu_model` reads `/proc/cpuinfo` (null off Linux); the GPU stays in run
  identities, not `TrainingRun.host`.
- Proposed infrastructure Wave memory (not written; `.lf/wave/` is read-only here):
  rented-host CPU class moves update time 1.6-1.8x at a fixed GPU label, so compare
  `host.cpu_model` and native stepping before attributing a slowdown; a packed
  ~50,000-row column makes per-time-step device recursions launch-bound.

## experiment-design b015d5 (2026-11-06): training throughput

Design: `experiments/protocols/training-throughput-b015d5-v1.md` (draft). Reversible
assumptions, for Jack Heart's review:
- Control = 07c490's `elemental_variants.elemental_recipe` (world relabel + capacity
  recovery), so the design waits for PR #436 instead of declaring a second copy of the
  same variant. If #436 stalls, this file can declare the identical two changes under
  its own ID.
- Reuses 07c490's 50 `ELEMENTAL_PAIRS` evaluation deals, so scores read against its
  baseline curve. They stay evaluation-only here; that protocol called them reserved, so
  sharing them is a choice for review.
- Packing is judged without strength runs, because processes share only hardware.
  Spending strength runs on it would buy nothing a digest cannot show.
- Proposed reservation $24.40; the remaining $25.60 of the cap is held back pending an
  amendment.
- An A4000 probe is included as the larger-CPU, cheaper-GPU shape. Etude's `a4000`
  template lists several rented machines that may be ghost records; check
  `etude machines` before launch.

## implement b015d5 (2026-11-06): step-1 probe software

Built: `scripts/bench_training_update.py` `--json` (whole receipt), `--recipe` as a file
or `module:function(regime_id, updates)` declaration, `--cuda-counts`;
`experiments/training_throughput.py` (phase driver, 1 Hz samplers, `/dev/shm` journal,
module-run cleanup, `runs()`); `tests/training/test_training_throughput.py`.
Reversible choices:
- PR #436 is still open, so `elemental_variants` is not on main. The probe names the
  control as the declaration string `experiments.elemental_variants:elemental_recipe`;
  it resolves at any commit containing #436 (main since `ebed0963e`). The CPU test
  keeps the frozen 183 file: it checks driver mechanics (receipts, samples, memory
  skip, cleanup), which the recipe does not change; the control's own 2-update CPU
  bench is recorded in the design status. Launch pins a commit that has both.
  Checked: the declaration loads from #436's file (w17, 256 streams, both seats,
  `recover_capacity` true).
- Children drop every `ETUDE_*` variable and track locally; only the driver is the
  launched run. Each child's `TrainingRun` receipt is the evidence.
- A phase is skipped when the previous completed phase's largest child peak x its child
  count exceeds `memory_fraction` (.85) of machine memory. Peak is Linux VmHWM, else the
  largest sampled RSS.
- Any failed phase fails the run after all phases have run; evidence is kept.
- Quote 2026-11-06 at `8d178f40e`: `--rent l4 --machines 2` plans $2.06 (reservation
  $2.10); `--rent a4000` refused, no stock at quote time. Re-quote before launch.

