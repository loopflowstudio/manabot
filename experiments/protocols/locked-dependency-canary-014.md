# Canary for the exact locked GPU dependency bundle

Frozen before GPU admission, 2026-10-17. Canary010 passed the source-identical
prebuilt Etude client installation in 10.170 seconds, then exhausted its separate
1,200-second project setup limit while installing CUDA dependencies. CUDA
admission and training never began. Its failure and provider-confirmed cleanup
remain retained; this protocol does not reinterpret it as a success.

Preinstall the exact 94 Linux x86_64 CPython 3.12 wheels selected by the existing
`uv.lock`, then execute the original GPU setup script. Named requirements, wheel
bytes and hashes are unchanged. Both the original canary lock and the grid lock
admit exactly this dependency selection. Local offline installation and frozen
reconciliation passed for both locks; this is not Linux CUDA validation.

The immutable archive is 4,221,296,640 bytes, SHA256
`a53386019a7f1bf557a1752b7492ca825114c526f4c16e1f904d78e0d10aa276`.
Its manifest SHA256 is
`b347ce79197346fbb6307b48e8c5ec04586bb46ab5eea481f7437cfbcb2c02b5`.
Admission requires complete remote readback matching that archive hash, the
helper's successful verification of all 94 real wheels against the original
source lock, passing failure-barrier tests, and a committed launcher revision.
Retain these receipts in `.runs/grid-007/linux-wheelhouse/`.

Embed the committed `locked_gpu_setup.py` helper and `client_install_canary.py`
wrapper in the immutable job command, retaining their hashes and command hash.
Write the helper outside the frozen source checkout. Supply a temporary signed
download URL through the job configuration; do not publish it in logs or commits.
Four range readers download the bundle with progress reporting. Verify archive
length/hash, manifest hash, allowed source lock, each locked wheel, exact archive
membership and requirements before any installer runs. Install wheels offline
with hashes and no resolver dependency expansion. Remove temporary bundle files,
then execute the ordinary setup, native build and final GPU health check.

Use the original scientific source
`4d65d477d8e7bc47a327464feea7ace08aae07da`, seed13599, small-model two-update
canary and existing prepared source projection. Preserve source-identical Etude
client installation, its 120-second total limit, the independent guardian, and
the five-minute pre-start cutoff. Initial GPU health gets 15 seconds; the helper
and original setup together get 1,200 seconds, CUDA admission 60 seconds, and
training 180 seconds. Preserve 240 seconds for retention inside a 30-minute job
and 36-minute rental limit. No deadline or learner target may be extended after
admission.

Admit one RTX5090, one attempt, no automatic retries, quote at most $1.25/hour,
with a $1 reservation. Fresh ledger reconciliation must include all historical
failures, current commitments and the $5 ancillary reserve within the existing
$50 total authorization. Do not start a scientific grid concurrently with this
canary. Any failure stops admission and requires reading its evidence before a
new recovery decision.

Success requires retained offline-install identity, normal completed setup,
passed CUDA admission, both training updates, complete artifact manifests,
source/checkpoint identity, finite checkpoint parameters, client startup phases
and provider-confirmed deletion. It proves this attempt's setup and retention
path; it does not resolve the historical CUDA/NVML cause or finite-weight anomaly.

This successful receipt may replace the failed canary010 prerequisite in
protocol011. That amendment preserves all remaining grid cells, sources,
transition targets, deadlines and grouped reservations. Embed the same verified
helper in each group and keep it inside the existing 1,200-second setup limit.
The ancillary reserve is now $5. The unlaunched GPU evaluation probe009 was
superseded by verified local route012 and receives no new reservation. Freeze any
longer learning comparison separately, using screen013 as explicitly prior
evidence and new independent training seeds and evaluation deals.
