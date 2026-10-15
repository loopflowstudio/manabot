# GPU validation — evidence and budget

Jack Heart authorized $50 total. All eight rentals are provider-confirmed deleted.
Bills/whole-rental estimates total **$2.505036**; conservative charges $3.32 plus
$5 ancillary reserve leave **$41.68 unallocated**, with no pending reservation.
`gpu-probes/staging-008/budget.json` freezes this receipt; `.runs/gpu-validation/budget.json`
is the live ledger. Final invoices are not all settled.

## Reusable serving buffers: paired validation passed

Run`3375fdd99515`, frozen source`317cbdbd352394f60c46274616510bcc179de40c`,
H10080GB HBM3, driver580.126.09. All four cases completed. Median serving gains:

|Actors|Maximum batch|Reference seconds|Staged seconds|Speedup|
|---|---:|---:|---:|---:|
|16|256|0.873|0.727|1.200×|
|16|1024|1.024|0.798|1.284×|
|128|256|4.797|3.886|1.234×|
|128|1024|3.095|2.177|1.421×|

Both128actor cases passed the preregistered1.15× target; neither16actor case
regressed. ABBA/BAAB order, two samples per variant/case, all retained. At128/1024,
preparation fell1.110→0.562s, H2D0.493→0.127s, forward1.231→1.228s. This jointly
changes CPU selection, pinned host/device buffer reuse and copying; individual
component effects are not isolated. Histograms were identical within every case.

All44fixed groups/176service calls passed exact input/output parity and storage
ownership gates. Separate artifact audit checked67,968live deliveries: all expected
requests, exact inputs/outputs, zero hash differences, matching measured source.
All23manifest files and25authoritative outputs verified. Failed standard download
was recovered with authenticated range reads, without rerental. Raw evidence and
receipts are in`gpu-probes/staging-008`; all samples, memory and detailed boundaries
are in`staging-validation-results.txt`. Persistent pinned/device storage is9.046MB
each atbatch256 and36.184MB each at1024; GPU peaks include both backends coexisting.

This is one-worker serving with virtual host delays, not real games/learning or
multiple physical GPUs. Ordinary collection must preserve action_focus float32,
independent pending observations and sampling/journal contracts before any training
claim. The laptop model still uses run007 reference costs; do not apply one pooled
speed multiplier. A staged cost model needs independent calibration/confirmation.

Deletion was confirmed14:44:17.821584Z; post-job billing interval74.034435s. Whole
rental estimate$0.346252 exceeds job estimate$0.264197. Process-scoped caffeinate
was used; neither the earlier delay's cause nor a causal remedy is established.

## Complete serving model: prospective validation passed

Run007`77223d8a4050`, source`4d220b3af1b405d06a5e33087af575b2cb2e205f`,
H100/driver570.195.03. Frozen preparation/H2D/D2H/validation coefficients were fit
once to381 actor8/repeat0 batches from run006; only forward points were recalibrated.
New actors4/16/128 × maxbatches64/256/1024, three repeats: all nine median timing
gates passed, actual/predicted0.979–1.090×; all27 samples0.950–1.129×. All best-batch
predictions passed, including16actors favoring256 over1024. All output hashes matched.
All37manifest entries verified. `gpu-probes/serving-007` holds raw evidence,
`service-model.json` and matching model-only `capacity-report.json`.

Both laptop queue and learner CLIs select this boundary with `--serving-costs`,
checking exact hardware/recipe/input evidence and reporting extrapolation. It
replaces model+link costs without double charging; the old default remains exact.
See `serving-integration-results.txt` and `learner-flow.md` for scale comparisons.
Multi-server rates assume independent CPU/memory/link capacity per server.

Run007's692.446s post-job billing interval has unestablished cause. Its whole-rental
estimate$1.158104 at collection time exceeded job estimate$0.390643; the current
settled bill is$1.134274. Both historical receipt and updated ledger are retained.

## Negative and earlier evidence

Run006's original model+assumed12GB/s/20us links failed all nine timing gates:
measured/model medians1.349–1.900×. Forward was0.983–1.071× calibrated, transfers
1.146–3.031× assumed; CPU preparation was missing. One live output hash differed.
The retrospective full-cost fit passed excluded medians, one barely; an individual
sample failed. Aggregate results were already seen, so this was not blind validation.
Run007 supplied the prospective check. All evidence remains in`serving-006/007`.

Runs001/002 retain bootstrap/source-admission failures. Run003 catalog reuse
improved64/256stream collection1.490×/1.252× with exact trajectories/raw/EMA and tiny
Adam differences. Run004 reproduced Adam differences without caching, retained a
26%512stream regression and two4096stream collector deadlines.
Run005's exact-input H100/L4 small-model inference ratios were0.878/0.859/0.884/
1.185/2.317/4.999 at1/4/16/64/256/1024. All ten ordinary three-update H100 cells
completed, including4096streams:98,304rows,63games,16.040s collection,5.003s learning,
22.481s whole,9.786GB peak. Raw/EMA matched;4096Adam had three nonexact elements,
max5.68e-14. Larger inference gains did not imply larger learning gains. Full prior
comparisons remain in`h100-005`, `l4-002/003/004` and hash-preserved previous notes.

Original local probes remain238.718/240seconds. Calibration slice now19.589804/20s;
conservative combined measurement283.973775/300s. Staging preflight1.645465s,305MB,
no local ML. Prior integration passed9exact prospective predictions,27unchanged L4
outcomes,108paired cohort and6high-scale cases. Physical distributed execution,
evolving policies, target CPU contention/all-reduce and strength remain unmeasured.
