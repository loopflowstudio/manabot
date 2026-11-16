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

2026-11-16 follow-up: `uv run etude logs --run recording_cost/6bffcb5ff678`
and `uv run etude logs --machine mtg338-a6000-2e3aec59e889` both retain only
`ProviderUnavailable: POST pods: status 400`, with no response body. Inspection
of the pinned launcher's `etude.launch.providers.http.send_to_provider` explains
why: its `HTTPError` handler returns `HTTPReply(status=error.code, body=b"")`;
`PodHTTP._ask` then raises a status-only error. The provider's refusal reason
cannot be recovered from those logs. That discard is an Etude limit, to be
fixed in Etude. Earlier rentals saw RunPod's create answer 400 when listed
stock was gone at creation, so a stock refusal is the likeliest reading of
this attempt; it is not a recovered reason. Retry proceeds under a fresh run
id: at most three further creates across RunPod A6000/A40/L4, one per verified
refusal (no pod listed, session settled), stopping on a non-400 refusal or
unknown state.

A read-only availability check at 23:04:26 UTC again reported the A6000 shape
rentable at $0.69/hour, with 0 of 1 in use. It does not establish successful
admission or explain the 400. Provider and storage credentials were injected
from Doppler `etude/prd` and `manabot/prd` respectively; no auth was changed.
The frozen source, $5 job cap and prior $0.00 settlement remain unchanged.

The all-off arm includes existing capture frame reads; this comparison measures
incremental option cost and does not separately attribute those baseline reads.
