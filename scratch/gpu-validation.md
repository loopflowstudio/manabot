# GPU validation — current evidence and allocation

Jack Heart authorized **$50 total** GPU spend for this goal, superseding the
original no-paid boundary. No other experiments, Tasks or publication are selected.
Full previous protocol/history and findings remain in committed versions and
hash-named `.protocol.txt`, `.history.txt` and `.notes.txt` under `gpu-probes/`.

## Earlier measurements and catalog direction

First allocation failed bootstrap before measurements. Second run `a7df3b8d1a77`
completed eight model cells, nine collection cells and a profile; two learning
cells failed source admission because archives lack `.git`. Later probes use the
existing verified source-bundle contract. All failures and costs remain retained.

Second-rental inference/collection curves are in `gpu-probes/l4-002` and the
main benchmark note. Batching helped the small model but not width384/depth8;
all nine exact collection comparisons passed. Whole collection stayed below50%
of best model-only throughput. All53manifest files verified; large tensors/trace
remain in `.runs/gpu-validation/result-002`. These are short untrained workloads.

Third run `a7f16bef6f6c`, frozen source `a5b73db1007ae247fc18d59e08722b1e4bfbc75f`,
tested catalog reuse only within one frozen-weight no-grad collection, restored
before learning. Three seeds10831–10833,64/256streams,4,096learner transitions,
four timed samples in ABBA/BAAB order after warmup. First cache computation stays
inside timing. The prototype rejects gradients and weight-version changes.

Median collection speedups were **1.490× at64**, **1.252× at256**, meeting both
predictions. All24trajectories matched exactly, including journals/reset seeds;
extra peak GPU allocation8,192bytes. The64stream seed10832 reference outlier
4.541s versus2.285s is retained; its2.181× pair does not determine the median.
Four ordinary three-update training runs completed at16/64streams:1,536learner
rows,384exposures each,96/24optimizer steps,11/zero finished games. All had48changed
state tensors and zero nonfinite/skipped steps. Initial/final raw and EMA matched;
16stream Adam was exact,64stream Adam differed in3moment tensors(max3.64e-12).
No tolerance was preregistered. Whole-run times8.825→5.847s and3.143→2.916s are
single ordered pairs with setup/host variation, not robust training-speed claims.
All59manifest files verified. Small records: `gpu-probes/l4-003`; large tensor
archive: `.runs/gpu-validation/result-003`. The fourth controls below address the
optimizer discrepancy. Production invalidation/API and mature-policy safety remain
unselected; no playing-strength claim follows.

All rentals used one L4 at$0.59/hour, pool ceiling$0.80, job30minutes/$1.50,
cohort$3,maxoneattempt. Driver caps: setup900s, measurements600s, child90s.
Etude owns source transport/self-deletion; credentials stay in-process. GPU drivers
differ across rentals, so use within-rental comparisons. Exact IDs, failed logs,
source/output checksums, billing and cleanup receipts remain with each result.

## Learner shapes and repeatability (fourth allocation)

Preregistered in `gpu-scale-protocol.txt`, frozen source
`d056d09218d3be3c234b40c9eafa70ad736f5c81`; run `1f97b066ed87`,
session `20e2d6c4c331`. Source archive SHA
`cfa68303d46ef31aac7668353b5840ef2aa5b6d0dd839bbfe9a96be75719bde4`.
A source readback timeout happened before rental; complete ordered range readback
with If-Match verified the unchanged archive in9.137s. No rental retry occurred.
L4 driver580.126.20 differs from earlier rentals; use within-rental comparisons.

Six three-update cells completed; both4096stream cells were interrupted by the
55second training deadline during their third collection. Each had already
completed two actual updates:65,536learner rows,16,384exposures,16optimizer steps,
21finished games. This is not an OOM result or a completed three-update run.
The partial timings remain diagnostic evidence; complete-loop capacity and
post-run GPU peak allocation are unavailable for this shape.

| Reference shape | Complete learning rows/s | Learning seconds / iteration | GPU peak allocated |
|---|---:|---:|---:|
|8×64, A|2,273|0.2252|166,784,000bytes|
|8×64, B|2,169|0.2361|166,784,000bytes|
|8×512|11,099|0.3690|1,135,156,736bytes|

The512stream point is4.883× reference-A rows/s, exceeding the predicted2×.
These include cold first updates and EMA. Synchronized method times separately
preserve all three samples; update_iteration excludes EMA. All completed cells
had24optimizer steps,25% retained exposures,48changed raw state tensors and zero
reported nonfinite/skipped steps. Shape changes filtering populations; equal
throughput units do not establish equal learning quality. There were zero games
at64streams and four at512, limiting representative-traffic claims.

Reference-A/reference-B Adam differs in3moment tensors/17elements, max3.64e-12.
Thus caching is **not necessary** for this non-exactness; its kernel-level cause
remains unidentified. Cached-A/B Adam is exact. Reference/cache differs in3moment
tensors at64(max3.64e-12) and2 at512(max1.14e-13). Initial/final raw and EMA states
are exact in every completed comparison. No tolerance was added and no universal
optimizer-equivalence claim follows. Interrupted final states remain unavailable.

At64streams, collection reference/cache ratios were1.300× and1.332×.
At512, cached collection was5.915s versus4.694s reference:26.0% slower in this
single ordered pair, with identical final raw/EMA. This prevents extrapolating
the earlier collection gain to larger shapes. Repeat with alternating order to
separate a real shape effect from host/time variation; neither is established.

All91manifest entries verified; output archive SHA
`a17b8f2dcb250ba4e58b1ff559fc89dabcdca65c91ef849cdfe1646c3be4ff86`.
Small raw records, source snapshots, summary, exact tensor comparison and explicit
successful-shape calibration are in`gpu-probes/l4-004`; full tensors/SQLite/archive
remain in`.runs/gpu-validation/result-004` and Etude output. The comparator's five
mechanical checks passed in0.703s; legacy probe use is233.608/240seconds.

Pod`g4893kl7qnbnhf` was confirmed absent at03:51:37UTC on2026-10-15; all four
rentals ended. First three provider bills total$0.188858; fourth estimate$0.067083,
combined$0.255941. Conservative charges$0.45 plus$5ancillary reserve leave
**$44.55 unallocated** of the original$50. No reservation remains. The authoritative
ledger and dated provider cost/cleanup receipts accompany the fourth results.

The laptop model now has measured distinct learner shapes rather than an assumed
linear per-row cost. Next is explicit global-cohort admission with measured service
and retained memory, coupled to inference sharing. Mature-policy traffic, high-end
GPU service and multi-GPU synchronization remain missing; no system-speed or
strength conclusion follows from these short L4 probes.
