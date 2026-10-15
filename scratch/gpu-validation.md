# GPU validation — evidence and budget

Jack Heart authorized $50 total. All ten rentals are provider-confirmed deleted.
Bills/whole-rental estimates total **$3.443180**; conservative charges $4.58 plus
$5 ancillary reserve leave **$40.42 unallocated**, with no pending reservation.
`gpu-probes/phase-010/budget.json` freezes this receipt; `.runs/gpu-validation/budget.json`
is the live ledger. Final invoices are not all settled.

## Large-collection stalls: assembly boundary identified

Run010`4370fade4b40`, source`88e91fc149d6af2c46dc3834113313f0ad5ca257`,
H100/driver580.126.09, retained4096streams x8steps and both seeds. Externally enforced
25s collections:10349/reference was killed inside `_build_batch`, active at least
14.532s; three cells completed. Completed assembly times ranged .863–7.194s;
slow10349 staged assembly used6.551/2.681s system CPU versus .649/.635s user CPU.
Container memory-pressure counters rose, but their scope does not isolate a cause.
This localizes a new stall; it does not explain run009's untraced64.393s sample.

Six retained samples match within seed; all four10350 samples match across variants
and prior009. No completed10349 reference identity exists. All41manifest files and
43outputs verified. Whole rental$.521702; pod deletion confirmed. Full outcomes,
raw traces, limits and next discriminating checks: `collector-phase-results.txt`
and `gpu-probes/phase-010`. Synchronized, process-isolated attribution supplies no
speedup ratio. Native stepping is41–45% of completed10350 collection, so the earlier
CPU-only1.25% share cannot rule out GPU-host parallelism. Assembly variability and
real multiworker/shared-memory scaling remain unresolved; calibration stays run007.

Run009`d17e9ea3ee14`, source`677710a879e6e9b7c670d268e676a28658c8af8b`, tested
ordinary64/512/4096stream x8step collection, two seeds, ABBA/BAAB. Four64/512 cells
had exact trajectories but mixed gains:64stream1.078×/.951×,512stream1.438×/1.034×.
One64case regressed5.13%; one512case missed10%; the large512gain has a slow first
reference. Both4096cells timed out90s; one staged64.393s/reference6.625s pair was
exact but incomplete and exceeded25s. All six fixed gates and18sample identities
passed;28manifest/30outputs verified. `collector-staging-results.txt` retains every
sample and proof limit. Ordinary collection performance gates remain unmet.

## Reusable serving buffers: narrower validation passed

Run008`3375fdd99515`, source`317cbdbd352394f60c46274616510bcc179de40c`, same GPU/driver.
At16actors, batch256/1024 gains1.200×/1.284×; at128actors1.234×/1.421×. All targets
passed. At128/1024, preparation1.110→.562s, H2D.493→.127s, forward1.231→1.228s.
Combined selection/buffer/copy intervention; component effects were not isolated.
All44fixed groups/176calls and67,968live deliveries matched exactly, with identical
within-case histograms. All23manifest files/25outputs verified. Allocation and memory
are explicit in `staging-validation-results.txt`; evidence is in`gpu-probes/staging-008`.
This one-worker/virtual-host result does not establish general collection or training
gains. Run009 supplies contrary transfer evidence. Laptop costs remain run007 reference;
no universal staging multiplier. Staged calibration and learning remain unverified.

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

Local limits and integration checks remain in accelerator-benchmark.md. Physical
distributed execution, evolving policies, shared-host interference/all-reduce and
strength remain unmeasured.
