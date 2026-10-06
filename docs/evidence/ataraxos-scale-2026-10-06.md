# Ataraxos-scale laptop feasibility — 2026-10-06

Jack Heart requested ordinary larger models and bounded laptop measurements.
The 384/8/feedforward-1536 model trained through TrainingRegime and reloaded
through the ordinary world-bound loader. It has **16,815,746 parameters** with
the selected semantic input and scalar value token. This is size parity, not
reproduction of the Stratego architecture. No strength or default-model claim.

The [compact evidence](ataraxos-scale-2026-10-06.json) binds original reports,
resolved recipes, architecture/world/runtime identities, checkpoint hashes,
per-window timings, costs and verification recovery. Full exports, logs and
source-file hash manifest remain in this checkout's `.runs/etu115-scale`;
preserve that directory. Frozen pre-change compatibility data is separately
retained in `tests/model/fixtures/capacity_base.json`.

## Measured rates

Apple M4 Max, 128 GiB unified memory; macOS and Torch identities are in the JSON.
Float32, batch 4, one CPU thread, four attention heads, scalar value-token,
history off and authored Allies/Lessons semantic input. Every model presented
**203 slots** to attention; the same saved native input batch was reused.
Rates are samples/s, median [min, max] across three two-second windows.

| Rung | Parameters | CPU forward | CPU diagnostic update | MPS forward | MPS diagnostic update |
| --- | ---: | ---: | ---: | ---: | ---: |
| w64-d1 | 138,498 | 234.6 [231.3, 404.2] | 77.2 [68.6, 80.4] | 137.0 [109.9, 146.0] | 22.0 [21.3, 23.4] |
| w64-d2 | 188,482 | 556.9 [540.4, 574.3] | 93.3 [77.7, 100.5] | 138.8 [135.6, 151.2] | 14.4 [10.6, 14.6] |
| w128-d2 | 712,834 | 271.7 [250.4, 277.4] | 45.2 [41.2, 47.0] | 139.4 [135.5, 159.9] | 21.2 [20.9, 21.3] |
| w384-d8 | 16,815,746 | 53.2 [48.7, 69.2] | 11.3 [11.0, 11.5] | 117.1 [101.9, 121.6] | 15.8 [14.6, 16.5] |

The diagnostic update is forward + uniform legal-action cross entropy + squared
scalar value + backward + Adam. These rates are neither RL learner throughput
nor complete-game progress. First call and construction timings are retained
separately. The second `first_forward_seconds` entry in original probe JSON
includes the first diagnostic optimizer step; the first entry is inference only.

Host one-minute load fell from 49.60 to 18.83 during the attempt. The 64/1 CPU
forward rate being lower than 64/2 is a visible contention confound. The first
fixture briefly overlapped focused tests; this was not an uncontended scaling
experiment. MPS fallback was disabled; all eight device probes completed.
No giant-batch fit or portable hardware ranking follows.

## Memory

MiB, with endpoint samples rather than exact peaks. Gradient + two Adam moments
are static estimates; activations and allocator overhead are additional. MPS
allocation and driver memory share system memory and must not be added to RSS
as independent physical totals. Buffers and input bytes are retained separately.

| Rung | Parameters | Gradient + Adam estimate | CPU RSS sample | MPS RSS sample | MPS allocated / driver |
| --- | ---: | ---: | ---: | ---: | ---: |
| w64-d1 | 0.53 | 1.58 | 370.3 | 546.9 | 17.1 / 87.9 |
| w64-d2 | 0.72 | 2.16 | 370.1 | 541.0 | 24.5 / 95.9 |
| w128-d2 | 2.72 | 8.16 | 430.0 | 553.0 | 29.7 / 1128.0 |
| w384-d8 | 64.15 | 192.44 | 945.5 | 625.7 | 352.7 / 1472.1 |

Default attention input is two players + 120 card slots + 80 permanent slots,
plus one value token. Actions and events are not appended attention rows. Naive
padding removal changes historical critic pooling and breaks focus/ownership
indexes. Safe packing would need remapping and preservation of the original
consumer layout or mathematically equivalent pooling contributions. No packing,
capacity or world ABI change was made.

## Training, failure and recovery

Each ordinary CPU fixture used one Ataraxos update, eight learner transitions,
two streams (the executor minimum), and zero advantage floor. All four
TrainingRuns completed and exported raw/EMA/optimizer artifacts. The fixture
verifier then failed four times because it compared a loaded evaluation-mode
model against a training-mode model: maximum policy differences ranged from
9.3e-10 to 5.6e-9. Those original failures and logs remain retained.

The verifier now uses evaluation mode for both sides. A separate read-only
verification recovery used the existing exports, checked positive optimizer
exposures and changed weights against each initialization seed, and proved
bit-exact policy/value equality after strict reload. No training or timing probe
was repeated. The original attempt took 186.51 seconds; recovery took 1.58
seconds. Elapsed wall time through its receipt, including intervening repair,
was 208.09 seconds, within the original 900-second cap. Full diagnostics and
individual stage costs are in the compact evidence.

This establishes bounded ordinary recipe training/export/reload and standalone
CPU/MPS model execution. It does not establish complete-game strength, scientific
capacity preference, chapter acceptance, or GPU stage execution support.
