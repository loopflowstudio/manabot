# Use local CPU scoring for the next learning comparison

2026-10-17. This supersedes the unlaunched GPU parallel-evaluation probe009;
it does not change or relabel completed GPU-only calibration008. No paid probe009
job was admitted. Its original protocol and pending history remain retained.

A separately frozen local timing probe used calibration628c404a1176's retained
384×8 initialization, trainingseed13598, and the same eight-leg random-anchor deal
block1359800000. Ordinary runtime compatibility, artifact hashes and replay passed
without overrides. It completed in54.909command seconds,53.590coordinator seconds,
with a14.299second slowest game. Its GPU cost was zero. The previous rental's
random panel took285.533coordinator seconds; these are different hosts, not a
controlled architecture speedup.

A second frozen local follow-up retained that serial random panel, added the
eight-leg scripted-greedy panel, then repeated both twice in four independent
one-thread processes. Serial16games took95.856command seconds; parallel32games
took67.693seconds. Equal-work speedup was2.8321×. Every canonical decision, result,
identity and replay matched its same-host serial counterpart; only timing fields
and timing-derived hashes were excluded. All45 follow-up manifest entries verify.
The original random-panel evidence is retained separately and was reverified.

Evidence roots are `.runs/grid-007/local-evaluation-probe/` and
`.runs/grid-007/local-evaluation-parallel/`, including their frozen protocols,
command records, complete traces, comparison and verification receipts. The host
has16logical CPUs and128GiB memory. At most four evaluation processes ran, each
using one inference thread. Individual games retained120second/10000command
limits; commands retained600second limits and the follow-up had a15-minute work
window plus240seconds for retention. No model was trained locally. Initialization
scores did not select a model, stream count or scientific seed.

The next learning allocation may therefore use GPU rentals for training and this
local CPU host for scoring, after declaring its complete scientific panel. All
models, training seeds, initialization controls and checkpoints in that comparison
must use the same declared local evaluation path, compatibility checks and game
limits. Preserve the GPU producer's identities separately from the evaluator's
runtime fingerprint. Do not mix earlier Linux scores into the new local panel or
claim cross-platform numerical identity. New-game/learned-policy costs may be
higher; retain bounded commands and failures. No scientific learning allocation
is admitted by this operational decision alone.
