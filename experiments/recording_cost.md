# Recording and statistics cost (Task 80db9a)

Exploratory measurement of CUDA collection decisions per second and update time
for off, native statistics with its persisted sink, 1%/10%/100% game recording,
escalated durable journals and the standard operation marker. One host, one seed
(80161), three rotating repeats, four updates per cell, fresh process per cell.
Ratios compare each arm to off in its repeat; report median and full range.
First-update costs remain included; this is a short-run operational comparison.

The Elemental recipe keeps its 256 streams and 64×2 model. Its frozen input v9
is replaced prospectively by current input v11 in every arm; historical recipe
bytes remain unchanged. Device is CUDA, with no CPU fallback. Journals include
escalated profiler scopes and use recording fraction zero (diagnostic cursors,
without per-decision state digests). Statistics include writes inside collection.
The standard marker arm includes watchdog and signal handling. Native game and
training workloads are unchanged between arms. Check receipts for equal counts.

Budget: one job, at most three execution hours, $5 job cap plus startup within
the launcher's $25 experiment commitment limit. No other rental or campaign.

Source: `1a3777d18987a8a7bc8fa3d5e2b797e2cbcd9ade`. L4 availability was
negative; A40 availability was positive but admission refused capacity. The
A6000 quote admitted a $0.69/hour offer and a $2.2425 planned window (including
startup), under the $5 job cap. A quote is not a bill or a rental confirmation.

Launch `edf5992304ee`, requested run `recording_cost/6bffcb5ff678`, failed
admission at 2026-11-16 22:54:46 UTC: `ProviderUnavailable: POST pods: status 400`.
Etude reported "job command not started" and "no run recorded". The initial
create result was unknown for `mtg338-a6000-2e3aec59e889`; no acquisition was
retried. A complete RunPod inventory read at 22:55:46 UTC confirmed no pod named
`etude-2e3aec59e889`. `etude terminate` then reconciled the request; the launcher
reports that machine unavailable with $0.00. No worker or training process ran.

The CLI's controller scanned retained storage before admission; the submitted
attempt used the public launch API with one job capped at $5 and an explicit
aggregate check against $25. Source preparation completed before the failed
provider create. Credentials were injected from Doppler, without changing auth.

**Measurement blocked:** a successful Etude/RunPod rental admission is missing.
No CUDA observations, ratios, spread or evidence-based default recommendation
exist. Do not substitute CPU timings or the rental quote. Resume this same
exploratory question on an admitted host; retain this failed attempt.

The all-off arm includes existing capture frame reads; this comparison measures
incremental option cost and does not separately attribute those baseline reads.
