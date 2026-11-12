# MTG-281 decisions

Jack Heart, Task directive dated 2026-11-11: "do 1 and 2. You have $15".
Retained analysis precedes a bounded matched before/after CUDA measurement.
The entire rental allowance, including pilots, setup and teardown, is USD15.

The inherited MTG-271 notes are unrelated; preserved at
`1659b0e290f55daa632ec8f869adff4a3059d79f:scratch/questions.md`.

Assumption: historical cumulative index growth can only be reconstructed
from surviving immutable blocks, not represented as a measured historical
file size or exact fold time. Retain this distinction in plots and conclusions.

The 500-update horizon tests early growth, not the later 850–1700 regime. A
600-update pair projects $14.46 before setup/teardown at the historical $0.59/h;
650 projects $15.76. The bounded plan does not assume a repair speedup to fit.
At preparation, no fresh provider quote was available and paid execution was
unlaunched. The execution continuation in collection-growth.md supersedes that status.

Reversible CLI choice: capture is the script's only operation; drop its redundant
subcommand. Reproduction requires a new `--out` directory, keeping the published
input directory read-only. Updated examples name the new destination explicitly.

Prospective engineering choices before any paired execution: analyze updates
51–500 with endpoint windows 51–100 / 451–500; practical-flatness bounds are
10% timing rise and 1 GB RSS rise, with >=50% reduction in every primary for a
partial verdict. All baseline primaries must grow; missing or materially
mismatched evidence withholds the overall verdict. These are descriptive
single-seed tolerances, not uncertainty bounds. Retain the frozen algorithm.

The 42,900-second requested rental leaves 300 seconds for best-effort teardown
inside 12 hours. Zero extra setup on the dedicated supply prevents the SDK
from adding time outside that envelope. Arms launch separately, not as a cohort.

Compression assumption: preserve published preparation/analysis snapshots as
evidence; simplify only maintained scripts and verify equivalent calculations
and identical job declarations without regenerating those snapshots.

Sync resolution (2026-11-11): retain MTG-281's measurement decisions as this
branch's active context. Main's unrelated MTG-272 hand-snapshot direction and
validation remain at
`37ffe26657f394b10f69256555ff45b3c58e82e4:scratch/questions.md`.
Main's implementation and conventions merge unchanged; the pinned experiment
sources and published preparation remain frozen.

Fixture choice (2026-11-11): declaration/budget tests inject pinned source
digests and the frozen world identity, and validate the recipe offline. They
never run games or training; temporary outputs are synthetic. Separate
unpatched runtime rejection prevents this fixture from serving as admission
evidence for the current checkout. Production guards and published bytes stay
unchanged.

Execution accounting (2026-11-12): RunPod's hourly amounts are still changing
within rows whose end times are in the past. At 02:08 UTC its two rows totaled
about $0.3683, while the baseline rental's elapsed lifetime at $0.59/h implied
$0.9567. Etude had already advanced its billed-through marker and displayed
less. Treat these as provisional billing; retain the full $7.20 live reservation
and reconcile with the larger whole-lifetime estimate, not the falling status
amount. This is no scientific-runtime change and no billing-finality claim.

Open (2026-11-12): the remaining ~$11.22 of the $15 cap fits one frozen-envelope
arm ($7.80), not a pair ($15.00 reserved, ~$11.86 expected). Choose (a) the after
arm alone, which stays inconclusive under the frozen reader, (b) a cap of about
$23 for a baseline retry plus the after arm, or (c) closing as inconclusive.
See collection-growth.md. No paid launch until one is chosen.
