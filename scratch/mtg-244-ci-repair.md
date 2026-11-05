# MTG-244 CI repair — 2026-11-04

PR #427's first publication (`6165e0fe48e7c4d500e724b030b6dc11fce749f6`)
failed the accepted risk outcome. Jack Heart directed repair before review
readiness, without baseline adoption, fewer choices, merge or training launch.
Sync completed at `1930146b08a1f1481af6e3b46a20f87d4a26a8fa`; this repair starts
there. The original operator note is preserved as git blob
`a40377be3672235b07c3932fa3545fa878280149`.

## Retained CI evidence

Run `41933042842`, Engine Budget job `126934566262`: same-runner reference
`ea316ff14bc6376dcc302b65d7c9c9330cd512e9` versus candidate:

| Measure | Reference | First candidate |
| --- | ---: | ---: |
| Search total | 168.192 s | 453.756 s |
| Worst reply | 4.909 s | 15.349 s |
| Steps/s | 12,817.0 | 5,654.8 |

All checked decisions/counts/digests match. This is an unaccepted regression;
`managym/perf/baseline.json` remains unchanged. Artifact:
`.cache/mtg-244-first-ci-engine-budget`; complete logs:
`.cache/mtg-244-first-ci.log`.

Layer Risk run `41933042919` failed before timing: its full branch checkout
lacked the squash-retained reference commit, and no artifact was uploaded.
Long integration failed its tiny comparison with collector deadlines on both
attempts. Academy UI had six failures. Final reinspection: main Integration
and Rust Tests were **cancelled**, not passed; other completed jobs passed.
There is no evidence of another failing assertion in those cancelled jobs.

## Implemented repairs

- Discover layer nodes initially and after static grants, retaining occurrence
  provenance and per-application reevaluation for dependencies. Apply proven
  independent instructions in timestamp order as one batch. Debug still checks
  every pair in that batch. Retain affected sets from resolved instructions only
  when an effect continues into another layer, avoiding a second predicate scan.
- Tap/untap advances its own revision. Discover tap reads through scopes,
  nested conditions and value predicates, including granted statics. Stun-counter
  removal also invalidates characteristics. Fixed finalized activations survive
  independent pool/tap writes; dynamic mana outputs and reductions refresh.
  Trigger event kinds depend on the raw board, with live LKI used for matching.
  Debug recomputation checks the caches and event index; clones retain isolation.
- Derived lookup tables use native permanent slots with holes and immutable Arc
  values. Large private operation expressions are boxed: the measured native
  `StaticPtBuff` is 840 bytes versus 48 for a resolved `Modification<LayerValue>`.
  Its inline storage previously inflated every per-target instruction. Rules
  state, policy schemas, Commands and source programs are unchanged by this cut.
- Whole-board trigger, mana, SBA and observation scans share one immutable board
  read in caller order. Every raw-board cache read retains its full recomputation;
  each finalized entry still compares against independent finalization. Singular
  reads retain their cache path. This removes nested whole-board validation per
  permanent without weakening the debug oracle or extending deadlines.
- `prepare` verifies/fetches both exact historical commits before extraction.
  Provenance is written before acquisition, binary hashes before execution, and
  the workflow uploads the explicitly selected hidden `.cache` artifacts even
  after failure. A shallow local Git fixture verifies fetching an absent pinned
  commit and refusing an unavailable one; the real prepare command also succeeds.
- Rebuilt the native release wheel and installed its extension atomically.
  `scripts/refresh_academy_fixtures.py` replayed the current w17 demo/navigation
  prefixes. Configurations, prefix offer IDs/labels, complete offers and decision
  kinds/revisions are identical. All five retained projections are identical
  after resolving shifted registry IDs by definition name. Historical corpora
  remain untouched; these are current-world runtime fixtures.

## Measurements and limits

The original 512-game probe at repair base counted 368,743 boards, 178,899
context clones and 198,672 comparisons, with zero dependency edges. An
intermediate batching/tap repair reduced these to 203,794 / 40,622 / 45,050,
retaining 101,650 steps, zero caps and digest 15843333459759757527. Subsequent
repairs further reduce finalization work and instruction storage. These are
exploratory attribution results, not final-source timing acceptance.

Contrary local timing remains: an intermediate complete Search-64 comparison
reported 127.581 s / 4.020 s worst against 77.341 s / 2.168 s reference.
All three full score digests, action counts, simulation counts and cap counts
match. One step pair was within 6%; a later pair was about 62% slower. The
shared M4 Max host's load rose from 27 to 124. The subsequent timing run was
stopped and reaped; its partial receipts are not valid comparisons. No local
result supersedes the first CI regression. Short `sample` profiles attributed
remaining work to invalidation, trigger-index rebuilding and layer instruction
allocation, rather than game cloning; they are not latency measurements.

Second publication, `beaacaa198acddb4162e5081cd570b1ec133ba05`, now has
completed failure evidence from CI `41952171449`: Engine Budget is 230.730 s
search / 7.045 s worst / 9,904.0 steps/s against 166.656 s / 4.883 s /
12,938.9 steps/s. Full score/action/simulation and step digests match. This is
still unaccepted. Artifact: `.cache/mtg-244-second-ci-engine-budget`.
Long integration again fails the tiny comparison, now the training resource wall
deadline on both attempts; 26 other long cases pass. Exact log:
`.cache/mtg-244-second-long.log`. Academy UI, Academy integration, clean-machine
Play, Python Unit and both conformance jobs pass. Rust/main integration were still running at inspection. Layer Risk `41952171271`
subsequently succeeded: every full workload/payment/policy comparison matches,
but it measures the still-regressed predecessor. Source-plan runtime `614a38fc8`
is pushed in measurement checkpoint `a7b0ddaff`: CI `41955382680` passes long
integration, but step time remains 10.54% over reference. Final lazy-SBA head
`68ef4a63d` now passes Engine Budget in CI `41956687926`: search
164.461→132.934 s, worst 4.818→3.860 s, steps 7.772806→7.104065 s,
with full scores, actions, counts and digests matching. Its long integration
and all Academy/Python/conformance jobs pass; Rust/main integration were
cancelled and the aggregate failed. Layer Risk `41956688122` remains incomplete;
prior risk run `41955382652` now passes all comparisons (risk note retains the
receipt). Published head is now `7851e6765`, including
the later simplification and main's pilot-evidence sync; its CI `41957935041`
and Layer Risk `41957934391` are running. The prior head's pass is not
current-head acceptance. The source-plan redesign and its measurements are recorded in
[the risk note](mtg-244-risk-validation.md); local passes cannot replace CI.
Pre-reconciliation note: blob `e0079911b6f52fdfde8bfd8967d0cf41c0dc537a`.

The retained debug failure spent 29.35 of 30.24 collection seconds in native
stepping/encoding, versus 0.70 in policy inference. After immutable batch reads,
the exact tiny-comparison test passes against a `CARGO_PROFILE_DEV_OPT_LEVEL=1`
wheel (97.84 s): all four runs complete both games and one update, monitoring
and notebook preservation pass. Collection takes 20.43 / 17.89 / 2.87 / 3.30 s.
These diagnose the workflow repair, not isolated release-speed acceptance.
Intermediate failures remain in `.cache/mtg-244-ci-training-*.log`; one intermediate
monitor rejected a changed source hash because edits overlapped that run. The
final passing run used stable sources throughout.

Check: 22 focused debug layer cases, two cache cases, two triggered-mana cases,
18 receipt checks, ten Academy server cases, all six formerly failing browser
cases and the exact tiny-comparison workflow pass. The frontend production
build, Ruff, rustfmt, actionlint and diff checks pass. Final-source server
projection passes all ten selected cases; the release extension is rebuilt and
restored atomically. Gate owns full suites and isolated performance acceptance
above.

Raw diagnostics, intermediate contrary measurements, final test logs, retained
training inputs and final runtime patch/provenance are preserved in gzip tar git
blob `36d7913583675575fdeaf15969a3cb2b00d78dc5` (`git show <blob> > evidence.tar.gz`).
The local copy is `.cache/mtg-244-ci-repair-evidence.tar.gz`. No benchmark, server,
watcher or training child remains running from this step.
