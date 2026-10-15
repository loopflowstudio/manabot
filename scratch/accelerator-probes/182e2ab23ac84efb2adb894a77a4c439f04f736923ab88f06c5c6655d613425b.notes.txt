# GPU validation — evidence and budget

Jack Heart authorized $50 total. All nine rentals are provider-confirmed deleted.
Bills/whole-rental estimates total **$2.921478**; conservative charges $3.91 plus
$5 ancillary reserve leave **$41.09 unallocated**, with no pending reservation.
`gpu-probes/collector-009/budget.json` freezes this receipt; `.runs/gpu-validation/budget.json`
is the live ledger. Final invoices are not all settled.

## Ordinary collector staging: performance gates did not pass

Run009`d17e9ea3ee14`, source`677710a879e6e9b7c670d268e676a28658c8af8b`,
H100/driver580.126.09. Real64/512/4096stream ×8step collection, two seeds, ABBA/BAAB.
All four64/512stream cells completed with exact trajectories. Median gains at64:
1.078×/.951×; at512:1.438×/1.034×. One64case regressed5.13%; one512case missed10%.
The large512gain depends on a slow first reference sample; all samples are retained.
Both4096cells timed out at90seconds. One retained staged64.393s/reference6.625s pair
had exact32768-row trajectories, but no complete median; the25second cooperative
collection allowance was exceeded. The deadline checks microsteps, not final rollout
assembly or the full timed window. The phase causing the delay remains unknown.

All six fixed field/dtype/storage gates passed;18retained samples matched complete
rollouts, final buffers, journals, reset seeds and sampling RNG. Nine staged storage
checks passed. All28manifest files/30outputs verified. `collector-staging-results.txt`
records every sample, failures, competing explanations and the next phase diagnostic.
`gpu-probes/collector-009` holds evidence. Next: externally bounded4096stream phase
attribution before training integration. No cap increase/retry or production edit.

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

Original local probes remain238.718/240seconds. Calibration slice now19.589804/20s;
conservative combined measurement283.973775/300s. Staging preflight1.645465s,305MB,
no local ML. Prior integration passed9exact prospective predictions,27unchanged L4
outcomes,108paired cohort and6high-scale cases. Physical distributed execution,
evolving policies, target CPU contention/all-reduce and strength remain unmeasured.
