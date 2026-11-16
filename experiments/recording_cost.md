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

Pending: launcher run identity, measurements, spread, recommendations, launcher
cost and provider-confirmed termination. No overhead claim yet. The all-off
arm includes existing capture frame reads; this comparison measures incremental
option cost and does not separately attribute those baseline reads.
