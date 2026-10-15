# GPU validation — evidence and budget

Jack Heart authorized $50 total. All seven rentals are provider-confirmed deleted.
Bills/whole-rental estimates total **$2.191590**; conservative charges $2.82 plus
$5 ancillary reserve leave **$42.18 unallocated**, with no pending reservation.
`gpu-probes/serving-007/budget.json` freezes this receipt; `.runs/gpu-validation/budget.json`
is the live ledger.

## Complete serving boundary: prospective validation passed

Run`77223d8a4050`, source`4d220b3af1b405d06a5e33087af575b2cb2e205f`, protocol
`serving-confirmation-protocol.txt`. One H10080GB HBM3, driver570.195.03, at$3.99/hour.
All six model calibrations and nine serving cases completed; all37 manifest entries
verified. Archive SHA`c5316eef29dbfcf1c113e1fd307dcbd953f99f156fec5e4c141f2d5b2f47a8fe`.

The frozen model includes preparation, H2D, D2H and response-validation costs as
nonnegative affine functions of padded rows and request count. Coefficients were
fit once to381 actor8/repeat0 batches from run006; forward points alone were replaced
with this new rental's six measured sizes. No overhead refitting occurred.

New actor counts4/16/128, maximum batches64/256/1024, three repeats each:
**all nine medians passed the preregistered20% timing criterion**. Actual/predicted
median duration ranged0.979–1.090×; all27 individual samples were0.950–1.129×.
All best-batch predictions passed.
The original model-only-plus-links control underestimated duration1.296–2.047×.

|Actors|Measured median seconds at maxbatch64 /256 /1024|Predicted best|Measured best|
|---|---|---|---|
|4|0.710 /0.474 /0.473|256 or1024|1024|
|16|2.138 /0.944 /1.087|256|256|
|128|16.277 /4.883 /3.589|1024|1024|

Every request/input identity and all cross-repeat output hashes passed. Peak GPU
allocation was1,677,555,200bytes. Declared repeat1 at128actors/batch1024 spent1.636s
preparing tensors versus1.284s in forward execution: preparation remains a concrete
optimization target. Its forward service was1.038× calibrated; preparation1.192×
the fixed fit. Full per-phase residuals, batch histograms and timer/delivery costs
are retained. No fastest-repeat selection or numerical tolerance was added.

This is real one-worker serving with virtual host delays, not actual simulation,
IPC/network, learning or multiple physical GPUs. Fixed real observation-pool rows
are chosen by deterministic indices, not the original game's exact input sequence.
Request counts above8 and occasional one-row batches extrapolate beyond fit support;
counts are reported per case. This successful cross-host/driver check supports this
specific boundary, not arbitrary hardware, trained policies or distributed rates.

`gpu-probes/serving-007/service-model.json` retains unchanged overhead coefficients
plus current forward points; `capacity-report.json` retains model-only service.
All source hashes, predictions, raw JSON/logs, budget and cleanup are portable there.
Full archive/input tensors remain in`.runs/gpu-validation/result-007` and Etude output.
The queue's optional `batch_service` replaces both model and link costs, avoiding
double charging; the default path remains exact. CLI integration into high-scale
and learner sweeps is still pending, so their old rates remain unvalidated.

Pod`xxf0u7g7fqgnq0` was confirmed absent at2026-10-15T13:09:42.447945Z.
Whole rental estimate$1.158104 exceeds the job-status estimate$0.390643: the attempt
finished12:58:08.504170Z, deletion was13:09:40.950585Z, a692.446s billable post-job
interval. Its cause is unestablished; `lifecycle-summary.json` preserves the evidence.
Use whole-rental costs for budget reservations, including idle/setup time.

## Why the model changed

Run006 `8c51916229c9` used the original model plus assumed12GB/s and20us/direction.
All nine timing predictions failed: measured/model medians1.349–1.900×, although
best-batch rankings held. Forward costs were0.983–1.071× calibrated for actual
batches, while transfers were1.146–3.031× assumed. CPU preparation was missing.
One output hash differed in one64actor/batch256 repetition; no tolerance was added.
All37 manifest entries and provider deletion passed; full evidence is in`serving-006`.

The fit protocol is`serving-calibration-protocol.txt`; implementation/report are
`serving-cost-calibration.py` and`accelerator-probes/serving-cost-calibration.json`.
Its excluded actor1/64 medians passed20%, but one barely passed and an individual
excluded repetition failed. Aggregate durations had already been inspected: this
retrospective partition was not blind validation. Run007 supplied the prospective
new-count check. A nine-case regression proves the default queue output unchanged;
closed-form complete-service/no-double-charge and invalid-cost checks passed.

## Earlier findings and remaining scope

Runs001/002 retain bootstrap and source-admission failures. Run003 catalog reuse
improved64/256stream collection medians1.490×/1.252× with exact trajectories/raw/EMA;
small Adam differences remained. Run004 reproduced Adam differences without caching,
retained a26%512stream regression and two4096stream collector deadlines.

Run005 measured exact-input H100/L4 small-model inference ratios0.878/0.859/0.884/
1.185/2.317/4.999 at batches1/4/16/64/256/1024. All ten ordinary three-update H100
cells completed, including4096streams:98,304rows,63games,16.040s collection,5.003s
learning,22.481s whole run,9.786GB peak allocation. Raw/EMA matched;4096 Adam had
three nonexact elements(max5.68e-14). Larger inference gain did not imply larger
learning gain. Complete earlier controls, failures and curves remain in`h100-005`
and`l4-002/003/004`, with expanded prior notes preserved by hash.

Original local probes remain238.718/240seconds. The20second calibration slice now
uses11.868seconds; conservative combined measurement use276.252/300. This turn's
fit/regression and frozen-loader/fake-backend checks used2.316/1.851seconds with
342/304MB sampled RSS; no local ML was run. Ruff and diff checks passed.
Actual distributed execution, evolving policy traffic, target CPU contention,
multi-GPU optimization/all-reduce and strength per compute remain unmeasured.
