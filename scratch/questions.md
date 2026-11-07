Inherited MTG-243 naming notes (another branch's handoff) were removed; they are at
`c6746f3ff:scratch/questions.md` and confer nothing on b015d5. Main's 07c490 notes (arrived with PR #436) are at
`ebed0963e:scratch/questions.md` and likewise confer nothing here.

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

