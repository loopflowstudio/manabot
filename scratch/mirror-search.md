# Mirror search continuation

## Final analysis — 2026-11-12

Report rebuilt over all 3,000 outcomes; numbers in the v2 protocol's "Final
analysis" section. Full trace audits of both segments passed (receipts
`reconciliation/{campaign,uniform-native}-validation-20261112.json`). Jack Heart
on the earlier page: “This is not readable imo. its a bunch of numbers and
process logs, not understandable science”. The page now opens with three Q&As,
one chart each, cost, limits; earlier sweep and provenance are collapsed.
Rendered and checked in headless Chromium (light/dark/390 px, all details
open, no overflow). Scripts/logs: `scratch/mirror-final-analysis-20261112/`
(`render_dev.py` re-renders presentation in 15 s from summary.json; the full
build takes ~9 min). Next: publish data, commit, then evidence review.

## Final collection — 2026-11-12

Campaign complete; both segments collected 10:09–10:12 UTC with the existing
collector and published. 3,000/3,000 outcomes: oracle 1,500 scored; uniform
native 1,499 scored + 1 TimeoutError (`policy/mirror-v2-uniform-d20-b0.01-w16/1819058000`).
Etude uniform-native `b9b50aa247df` succeeded 02:51:17 UTC, $0. Receipt:
v2 `reconciliation/collection-final-20261112.json`; Etude read:
`transition/etude-final-20261112.json`. Scripts/logs in
`scratch/mirror-collect-20261112/`. The local collection call hit a 600 s tool
limit during publish, but the mini process finished: its manifest (written only
after all uploads) holds both inventory hashes, and all 3,053 entries hydrated
and verified. Next: rebuild the report over the full grid (analysis step).
Offline Python needs the overlay `--with numpy==2.4.2 --with pydantic==2.12.5
--with torch==2.10.0 --with etude-sdk==0.2.6 --with boto3` and `PYTHONPATH=$PWD`.

## Report clarification — 2026-11-12

The owning report now separates execution, health, learning, training exposure,
generalization, native support and replay identity. A full owner rebuild
preserves every scientific result and input from the dated November 11 snapshot;
no new collection or execution occurred. Reconciliation evidence is in v1
`reconciliation/{analysis-20261112.json,verify-analysis-20261112.py}`. The seed
check reuses the exact-bound November 11 receipt; it is not an expanded audit.
Nine focused report/data checks and Ruff pass. Light/dark and 390-pixel browser
checks pass. Scientific acceptance remains unresolved for evidence review.

## Analysis handoff — 2026-11-11

Analysis uses the unchanged 13:09–13:10 original and 13:51 native snapshots,
not a fresh collection. Etude read at 14:22:52 UTC still reports original attempt
`3a2b62144b9a` cancelled and continuation `b9b50aa247df` running on mini, $0
rental spend. No execution was started, stopped or changed by analysis.

The owning report now surfaces completed whole-deal contrasts and the existing
inclusive execution accounting. Scientific acceptance remains unresolved; the
uniform cells contain only 3/5/2 games. Continue to evidence review with this
dated limitation rather than treating an incomplete campaign as delivered.

Verification at 14:30:15 UTC: all v1 science and v2 numerical results/inputs
unchanged, 1,510 per-game seed headers checked, 3,414 published extension
evidence hashes matched. Nine report/data checks and Ruff pass. The published
verification receipt and runnable check are in v1 `reconciliation/analysis-20261111.json`
and `verify-analysis-20261111.py`; the prior summary is retained there for regression.
This header check does not extend the dated Command/algebra audit or claim replay.

## Native uniform running — 2026-11-11

The oracle-barrier transition is complete. Etude run
`mtg269-mirror-v2-uniform-native`, attempt `b9b50aa247df`, session `c0e8be79ff4c`,
started at 13:45:05 UTC on the owned mini. Seven actual uniform workers were
verified at 13:49 UTC. Do not launch another run or resume the original.
The current Graph execution hands off to analysis, then evidence review; the
scientific study is unfinished and there is no PR yet.

Jack Heart: “Switch at oracle barrier (Recommended)”, then “Switch the uniform
stage to compact+native at the oracle barrier per the conversation decision,
then continue toward report and PR”. Same 3,000 scheduled outcomes, raw sampled
MTG-183 opponent, seven one-thread workers, zero rentals and absolute deadline
1794852989 (2026-11-16 18:16:29 UTC). No new training, main sync or merge.

## Immutable execution identities

- Original: source `0063d781710928df996402957a0b30123d332875`,
  `~/src/manabot.w13-mirror-v2`; data `~/search-scaling/mirror-v2`.
  All 1,500 oracle outcomes completed (500 each at 4/8/16 worlds), zero scored
  failures. Etude `mtg269-mirror-v2` / `3a2b62144b9a` cancelled at 13:07:38 UTC.
  No original game worker remains. Original source/binary/evidence are unchanged.
- Continuation: source `f845b02ef4a26377abad4e1c54bf6443ad3fab0e`,
  `~/src/manabot.w13-mirror-native`; data `~/search-scaling/mirror-v2-uniform-native`.
  Compact v4 receipts and native encoder, execution identity v3. The source
  archive is unmanaged, reusing the frozen environment. Do not overwrite it
  with subsequent report/protocol commits.
- Candidate native SHA256:
  `8addfe1025dc7ccf2d32118913ccfa9a1cce48e4fbd69f21f25492e0b779ad69`,
  `~/search-scaling/mirror-native-build/native/_managym.cpython-312-darwin.so`.
  Original native SHA256:
  `7c22a1e101874ae4818ba8250326277b2fb3b0a55260edcb9da0c162080baf25`.
- Checkpoint SHA256:
  `4f2ad656ba036a33d527212a7ac2badf2c5b9db136916f730846b07eef2fbea4`.
  Immediately before launch all 351 candidate/349 original bindings, both native
  binaries, checkpoint bytes and zero-cost Etude capacity were checked.

Mini SSH: Jack Heart said “mini is at 100.96.227.95”; use `jack@100.96.227.95`.
For launcher reads use the installed schema-13 Etude source via
`PYTHONPATH=/Users/jack/src/etude.ad319b-show-live-progress/src` and
`uv run --project /Users/jack/src/etude --no-sync python`. The frozen game SDK
cannot read schema 13. `client.default().read/attempts/runs` are sufficient for
read-only inventory; preserve the frozen game environment and launcher DB.

## Cancellation recovery

The original shell began seven uniform attempts before cancellation: three
explicit interruption footers and four readable pending streams with missing
callbacks, zero uniform scores/failures. Worker 0's log proves repeated SIGTERM
interrupted traceback formatting and gzip cleanup; the other partial logs do
not establish exact exception names. The first seal rejected these streams.

Commit f845b02e makes stop handling idempotent and provides explicit exact-hash
Etude cancellation recovery for readable pending traces. The v2 predecessor
seals all 1,500 outcomes and seven interruptions, with four recovered callbacks.
Original streams are unchanged: no synthetic footer or score. Every new worker
holds shared predecessor shard locks and executes only missing uniform keys.
Scored failures remain completed keys. Do not rerun the launcher helper.

Recovery receipts, raw seven interrupted streams, all original logs and the
failed/successful seal logs are retained in
`experiments/data/search-mirror-v2/transition/recovery/`; official stop and v2
predecessor receipts are alongside that directory. The interrupted streams
retain 10 Commands, 17 pending events and two completed search receipts (4.68 s).
In-flight work and four terminal-end times are unknown, not zero.

## Dated evidence and report

The local snapshot is deliberately fixed while the remote job continues:

- Original snapshot 13:09:53–13:10:27 UTC: 1,500 sealed outcomes, 1,530 published
  files verified, 299,087 Commands, 141,138 hero decisions. Summed game wall
  931,110.94 s, hero 921,584.70 s. Previous snapshot files remain exact prefixes.
- Native snapshot 13:51:36–13:51:38 UTC: 10 sealed outcomes, uniform 4/8/16 worlds
  3/5/2, zero failed. Audit checks 2,088 Commands, 953 compact receipts, finite
  values, max |Q| 1.05719; game wall 1,713.33 s and hero 1,641.31 s.
- Prior original offline algebra/Command audit covers 1,469 outcomes, 292,767
  Commands and 138,025 receipts, max |Q| 1.15808. The last 31 oracle outcomes
  have source finite checks and seals; do not expand that audit's claimed scope
  or rerun unchanged evidence merely for another pass.

The owning report is `experiments/data/search-mirror-v1/{report.html,summary.json}`;
protocol `experiments/protocols/search-mirror-v2.md`. It retains all six 500-slot
cells: 1,510/3,000 retained, 1,490 missing. The original v1 8,000 outcomes must
remain unchanged. Implementation costs are separate: oracle/full versus
uniform/compact is not an isolated sampler latency comparison. No reliable
sub-second, full-grid equivalence, completed strength or campaign ETA claim.

Exact evidence locations: v2 `reconciliation/` contains final original/native
inventories, retention receipts and audits. `transition/` contains launch,
status, worker headers, admission, recovery, source delta and inclusive execution
accounting. All artifacts use existing manabot.data content-addressed storage,
with bytes published before the manifest; no raw experiment data belongs in git.
Etude rental spend is zero; complete CPU and inclusive monetary costs are unknown.
All setup/recovery elapsed time stays inside the original seven-day ceiling.

## Applicable validation

62 local runner/report/data/training-recovery checks and nine mini recovery
checks pass. Both native admissions precede cancellation: 304 golden views
byte-identical (nine tensor/buffer checks); six checkpoint roots preserve actual
Commands, both final views and leaf signs at atol 1e-6 / rtol 1e-5. Max target
error 1.86e-7, rollout error 1.79e-7. The mini has one intra-op / eight inter-op
threads; laptop controls had inter-op 16. Native/search/encoder code is unchanged
since admission; the only runner AST change is idempotent stop handling.

Single mini step-100 oracle/uniform roots took 3.148/3.205 s with two old workers
active. Local fixed-root 4.98x equal-root uniform median gain is not measured
seven-worker M1 throughput. Earlier broad Python collection was blocked by the
retained psutil macOS signature; no repository-wide type-safety claim. Earlier
Rust debug checks: 533 passed, one ignored; Clippy/fmt passed. No Rust changed
in the transition recovery. Three original encoding warnings remain retained.

The report now labels original and native collection windows separately. Nine
report/data checks and Ruff pass after that display correction; frozen remote
source and execution identities remain unchanged.

The full report rebuild passes v1 invariance: all 8,000-game scientific fields
and prior input manifest entries are unchanged; only extension and analysis
source metadata differ. All 3,414 extension evidence hashes match published
manifest entries. The regression receipt and runnable check are in v1
`reconciliation/transition-report-20261111.json` and
`verify-transition-report.py`, with the prior summary retained alongside them.

## Next review

Use the published dated snapshot for analysis and inspect the existing Etude run
before any future collection or execution. Continue the selected Graph; do not
create another Task, Graph or worktree. Stop only resources whose authorization
has ended. No local observer, benchmark or server remains running; the intended
Etude continuation owns its seven bounded workers and deadline.

Prior local measurement tables, exact quotes and superseded operational snapshots
remain at `f845b02ef4a26377abad4e1c54bf6443ad3fab0e:scratch/mirror-search.md` and in
published v2 `latency/` artifacts. Historical v1 and earlier execution notes are
at `7fd6c202f163bc7dd0d507c7eab121086af84a68:scratch/mirror-search.md`.
