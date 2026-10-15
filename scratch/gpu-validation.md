# GPU validation — evidence and budget

Jack Heart authorized $50 total. All eleven rentals are provider-confirmed deleted.
Bills/whole-rental estimates total **$3.707791**; conservative charges $5.11 plus
$5 ancillary reserve leave **$39.89 unallocated**, with no pending reservation.
`gpu-probes/assembly-011/budget.json` freezes this receipt; `.runs/gpu-validation/budget.json`
is the live ledger. Final invoices are not all settled.

## Collection: CPU costs and a rejected memory-policy change

Run011`ed9273017de6` compared default/process-disabled huge pages,4096streams x8steps,
two seeds, two fresh collections per cell. All four cells completed; disabling THP
made median collection4.00%/9.48% and assembly28.77%/40.87% slower. Both diagnostic
advance gates missed. Allocation calls took5–8ms; row copy .704–1.060s dominated
assembly. Sampled container pressure stayed flat during rows; the earlier stall did
not recur. Eight samples match across policies and prior same-seed identities.
All67manifest/69outputs verified. Keep default policy and run007 calibration.
`collector-assembly-results.txt` records policy readback, counters and proof limits;
`gpu-probes/assembly-011` retains raw evidence. The failed monitor was resumed for
the same run; pod deletion confirmed. Next: matched actual1/2/4-worker replay scaling.

Run010`4370fade4b40` caught10349/reference at25s inside `_build_batch`, active at
least14.532s. Three cells completed; assembly .863–7.194s, slow10349 calls dominated
by system CPU. Container pressure rose but did not isolate a cause. Six retained
samples match within seed; four10350 match across variants and prior009. The10349
reference identity is missing. Native stepping was41–45% of completed10350 collection,
contradicting transfer of the CPU-only1.25% share. `collector-phase-results.txt` and
`gpu-probes/phase-010` retain all41manifest/43outputs. Synchronized/process-isolated
attribution supplies no production speed ratio; the kernel-level cause stays open.

Run009`d17e9ea3ee14` ordinary64/512/4096stream x8step ABBA/BAAB collection failed its
performance gates.64stream gains1.078×/.951×;512stream1.438×/1.034×. One64case regressed
5.13%; the large512gain has a slow first reference. Both4096cells timed out90s; one
staged64.393s/reference6.625s pair was exact but incomplete and exceeded25s. All six
fixed gates/18sample identities passed;28manifest/30outputs verified. Complete
samples and proof limits remain in`collector-staging-results.txt`. Later faster
runs do not erase these failures or identify the untraced64.393s sample's cause.

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
