# MTG-272 hand snapshots and API audit — 2026-11-11

## Direction and boundaries

Ryan: "one more request: can you add true hands in batched mode?"
Originating clarification: "yeah, he is training a belief model."
Task scope: "Lets make this a separate task from the versioning changes, and have it take on the more thorough audit of differences between single vs batched mode APIs".

Jack Heart's exact direction:

- c1389729: "assumign its the input to a belief model seems unnecessary, shouldnt wejust label it as a HandInfo or smoething". The steer selects `HandSnapshot` / `hand_snapshots(players)`.
- d793c7d4: "the output of the engine is just waht was in hand, choosing to use it as a belief SL target is the business of hte client". Engine docs describe authoritative contents at the read; clients own uses.
- 14d5db38: "1 A / 2 A / 3a - lets discuss further / 3b A". Implement explicit read delegation in CapturedVector, generation-based prefix rollover and repeatable explicit Torch seeds.
- 3437c194, resolving 3a: "ok, remove auto reset". Gym returns terminal observations and true done flags; callers reset.
- 5fde3453: "do C in this chagne, A in a nother diff"; "not worried about outside users yet". Built-in vector opponents become None/Random. e6247540 corrects the follow-ups: native heuristic MTG-277, fixed-seat batched observations MTG-276, batched fork/search MTG-275. None is implemented here.
- 8507f1b8 at demo: "i see we added a new class but i dont see where the old API uses it"; on scalar unification: "yeah tat sounds good". Add native/PyO3/Gym `hand_snapshot(player)`, remove `slot`, and migrate the three scalar belief clients with identical counts.

Preserve native-seat ownership, complete duplicate counts and read identity.
Hidden hands stay outside viewer observations, policy inputs, shared captures
and viewer exports. No frozen layouts, world/checkpoint identities, deck
disclosure or legal actions change. MTG-271 owns versions, dependencies and
release workflow; no release or isolated dependency-resolution proof here.

## Current approach and deletion status

- `managym/src/agent/env.rs::hand_snapshot` owns the record, using `hand_counts`
  over native zones. `vector_env.rs::hand_snapshots` maps it in slot order.
  PyO3, installed stubs, Gym, CapturedVector and Torch carry that one contract.
  Belief `audit.py`, `sampling_data.py` and `population.py` consume its counts;
  viewer observation reads remain for input/provenance checks and test oracles.
- `Env::reset` and branch construction allocate process-wide capture generations. Repeated seeds,
  replacement slots and Torch rebuilds get fresh identities. Capture origin and
  stale checks remain; generations are neither portable replay IDs nor serialized
  into public captures. `CapturedVector._synchronize` uses generations exclusively
  for prefix rollover, retaining attempted inputs before native mutations.
- Scalar terminal hands survive until explicit reset. Vector reads after done
  describe replacement games, including inactive slots; collectors apply masks.
  Discard failed-call results and recover/reset before reading again; isolated
  failures need failed-slot replacement. Old snapshots remain historical reads.
- `docs/engine.md#hand-snapshots` owns the contract/example;
  `docs/python-api-audit.md` owns the capability comparison and follow-ups.
  Python buffered stepping clears inactive reward/done fields but preserves
  observation rows; the original guide assumption was wrong and is corrected.

**Delete — do not maintain:** all removed: HandTarget/hand_targets and aliases;
HandSnapshot.slot and vector-specific construction; observation-materializing
hand counters in the three clients; Gym auto_reset and unread true-done info;
PassivePolicy/OpponentPolicy::Passive, parser arm, exclusive fixtures and tests,
passive decision-profile CLI branch and Env::pass_priority_action_index.
Duplicate scalar branch constructors with zero generations are also removed;
the shared constructor preserves trace ownership, reset seeds and fork settings.
It inherits disabled instrumentation from `Env::new`; `selected_fork` delegates
the exact clone to `fork`, which alone restores the replay prefix.
No deletion targets remain. Historical receipts retain policy strings; replay
of passive needs the original source. Academy's scalar debugging policy is
outside this cut. No heuristic, fixed-view batch or search feature is added.

Breaking release-note items for MTG-271: Gym auto_reset removed; every explicit
Torch reset seed reproduces the deal; CapturedVector rejects unbuffered native
mutations; built-in passive and its decision-profile CLI branch removed.
There is no docs/releases.md; the engine guide describes the changes.

## Current publication state

The scalar/batch implementation, seed and branch identity repairs, and deletion
cut are complete. No product choice remains open. The publication pass refreshed
the full debug suite, affected Python suites, and isolated 3.12/3.13 consumers
against merged main `9f69a96a1`, including the outcome-table recovery path.
The guide example, pending seed and independent-branch demonstration ran from
both installed wheels. Detailed results are below; earlier results remain dated
evidence rather than claims about the final branch. Publication leaves merging
and Task completion separate. No throughput, CUDA/MPS, vector terminal-hand
retention or full scalar-helper parity claim.

### Pre-repair reproduction — 2026-11-11 (historical)

Two bounded `uv run --no-sync python` probes of the existing extension confirmed
the source findings without replaying games or rerunning suites:

- Reset at 42, read, then `set_seed(99)`, read: snapshot seed changed 42 → 99;
  game ID, revision and cards stayed equal.
- Reset at 42, then call `clone_env()` twice: root game ID 1, clone IDs `[0, 0]`,
  clone revisions `[1, 1]`. The zero-generation materialization paths above were
  inspected in source, not executed in this pass.

The probes used the retained CPython 3.12 environment; uv warned that it did
not satisfy the newly merged 3.13 development request. These are bounded
behavioral reproductions, not current development-environment validation.
That realign pass changed no native code and reran no tests. The identity repair
and lifecycle compression proofs below supersede these reproduced defects.

## Identity repair validation — 2026-11-11

- `uv sync --locked --python 3.13 --extra play --extra dev --extra artifacts` — passed with CPython 3.13.7 and locked dependencies in this checkout.
- `cargo test --manifest-path managym/Cargo.toml -p managym --lib hand_snapshot -j 2 -- --test-threads 2` — five passed; the new failed-reset fixture bypassed admission and panicked. Marked it submitted; `cargo test --manifest-path managym/Cargo.toml -p managym --lib hand_snapshot_retains_live_seed -j 2 -- --test-threads 2` passed. All six focused debug cases now pass across these runs.

- `CARGO_BUILD_JOBS=2 uv run --locked --extra play maturin build --release -i .venv/bin/python -m managym/Cargo.toml -o managym/target/wheels` — passed; matching cp313 extension extracted from the wheel and atomically renamed into `manabot/managym/`.
- `uv run --locked --extra dev --extra artifacts pytest tests/env/test_hand_snapshots.py -k 'seed_changes_only_on_reset or branch_identities_and_capture_guards' -q` — 5 passed, 9 deselected in 14.84 s.
- `uv pip install --no-deps --target <temporary>/installed <cp313 wheel>`, then outside the checkout `uv run --no-project --python <root>/.venv/bin/python -m pytest --confcutdir=. test_hand_snapshots.py -q` with the installed tree on PYTHONPATH — 14 passed in 1.18 s. Installed extension paths, both import aliases, packaged stub and guide example passed. Path proof normalizes macOS `/var` and `/private/var` aliases after the initial literal-prefix assertion failed. Reuses development dependencies; isolated 3.12/3.13 consumer checks remain for gate.
- Changed-file Ruff check/format, Rust formatting and diff whitespace passed. No publication or Task movement.

## Earlier scalar unification proof

The release extension was rebuilt and replaced atomically before compression.

- `cargo test --manifest-path managym/Cargo.toml -p managym --lib hand_snapshots -j 2 -- --test-threads 2` — 3 passed in debug: control-seat parity, complete/empty counts, reset/replacement identity.
- `uv run --extra dev --extra artifacts pytest tests/env/test_hand_snapshots.py tests/belief/test_audit.py tests/belief/test_sampling_data.py::test_real_frozen_collection_keeps_both_viewers_and_three_game_splits tests/belief/test_learning.py::test_supervision_matches_hand_and_rejects_another_revision -q` — 16 passed in 3.98 s.
- `uv pip install --no-deps --target <temporary>/installed <wheel>` outside checkout, then `uv run --no-project --python <root>/.venv/bin/python -m pytest --confcutdir=. test_hand_snapshots.py -q` — 9 passed in 1.55 s. Guide example, installed imports and stub passed. Development dependencies reused; temporary files removed.
- Changed-file Ruff, Rust formatting and diff whitespace passed. No publication, landing or Task movement in that pass.

## Earlier validation and contrary findings

Full records before scalar unification are at
`dfba956ac0c0824a6ebfb05b11335c293242ddea:scratch/questions.md` and
`scratch/qa.html`. They are evidence of those runs, not the final branch.

- Post-rename debug hand tests: 3 passed; installed-wheel snapshots: 8 passed in
  1.55 s, with both imports/stubs and guide example. Same-seed stale capture
  regression passed. No clean dependency-resolution proof.
- Passive removal: 20 debug vector tests and 17 focused Python cases passed
  (unsupported-policy boundaries, historical receipt rejection, buffer parity,
  autoreset replay and model forward). The evidence fixture needed `by_alias=True`.
- Prior compression: 20 debug vector tests; 64 focused Python passes, then one
  exposure substitute fixed to implement generation reads and its rerun passed
  (65 total). All-feature Clippy, Rust/Ruff formatting and whitespace passed.
- QA used a fresh /tmp wheel install: guide example, 1,600 native snapshot
  cross-checks, 9 completed games across 13 identities; Torch 300 steps,
  3 done slots/6 identities, repeatable reseed. Observed snapshot time was
  0.001 ms/call at four slots, not a throughput claim. The before-reset error
  was corrected to name hand snapshots, followed by debug tests and atomic build.
  Torch required separate `torch==2.10.0`; dependency packaging belongs to MTG-271.
- Affected suite: 424 passed, 3 failed in 369.79 s. All three retained TLA
  resource-limit receipt cases returned `binding_mismatch`; rebuilt base
  `402dfd5b2` failed identically. Earlier two-case proof on `61a51a3ff` failed
  at action 370. Frozen evidence is unchanged. QA observed these tests were not
  selected in CI; see the retained report for that dated check.
- Exact-quota recovery's removed `RolloutBatch.obs` assertions now use
  `observations().observation`; focused test passed in 2.56 s. The missing
  `experiments/data/int-13-conditional-strategy-fixture-v1.json` was restored
  from retained artifacts; all 16 conditional-search cases passed.
- Publication: `CARGO_BUILD_JOBS=2 cargo nextest run --manifest-path managym/Cargo.toml --profile ci --test-threads 2` — 1,054 passed, two skipped in 190.888 s (`/tmp/mtg272-publish-rust.log`).
- `PYO3_PYTHON="$PWD/.venv/bin/python" cargo clippy --manifest-path managym/Cargo.toml --all-targets --all-features -j 2 -- -D warnings` — passed; Rust formatting, Ruff on 28 changed files, workflow validation and whitespace also passed.
- `uv run --extra dev --extra artifacts --extra notebook --extra reports pytest tests/training/test_recovery.py tests/sim/test_local_update_sampling.py tests/sim/test_search_command_evidence.py -q` — 58 passed, one failed in 265.97 s. `test_multistage_boundaries_preserve_state_and_completed_work[True-start]` hit Etude `train/run.py::seal_logs` FileNotFoundError unlinking a metrics `.open` file; a writer also reported a missing file. Log `/tmp/mtg272-publish-python.log`. Same node with `--basetemp=/tmp/mtg272-publish-recovery-recheck` passed in 32.68 s (log `/tmp/mtg272-publish-recovery-recheck.log`). Intermittent SDK failure, not a demonstrated fix.

At QA, PR #458's listed CI checks passed while three Integration jobs were
pending. At publication `lf sync` merged main `402dfd5b2` cleanly. These are
historical observations, not current CI status. `lf task status MTG-271 --json`
on 2026-11-11 reported realign with no PR; that checkout was not edited.

## Context and compression

The merge at `7de8b87dc` included upstream MPS support (#457); snapshots remain
CPU records, with no device validation added. Main throughput/C2 notes remain
at `53c2cb2f987f9768af7fd2360bb8c5f77c7113a5:scratch/questions.md`;
earlier b015d5/C2 notes at `757bb6257ef07cf2536ad7f7ec24f0f7309628c6:scratch/questions.md`.
Pre-rename implementation proof is at `64779909e75041ffa9f35b4bc532a9240b688c55:scratch/questions.md`;
pre-opponent concept review at `36d9451ef844fe23d59743f32dbdfaa121ab501e:scratch/`.
This Task has no Wave; there is no selected Wave memory to curate.

Compression reuses `_score_belief_truth` for both single and paired audit scores,
so the same hand metrics have one implementation. The parity test reads each
scalar snapshot once. The audit drops superseded verification prose; the concept
example uses list position instead of the deleted slot field. Historical QA
transcripts retain the API they actually ran. No native source changed in this pass.

- Compression check: `uv run --extra dev --extra artifacts pytest tests/env/test_hand_snapshots.py tests/belief/test_audit.py tests/belief/test_sampling_data.py::test_real_frozen_collection_keeps_both_viewers_and_three_game_splits tests/belief/test_learning.py::test_supervision_matches_hand_and_rejects_another_revision -q` — 16 passed in 5.59 s. Ruff check/format on the two changed Python files and `git diff --check` passed.
- Pre-curation uncommitted notes were moved intact to `/var/folders/m6/r3tllnrs1yq7yfbwm680tss40000gn/T/mtg272-before-compress-5cs77nw1/questions.md`; live direction and latest proof remain above. No publication or Task movement.

## Main sync — 2026-11-11

Assumption (reversible): retain MTG-272 direction as this branch's active context.
Main's MTG-271 direction, exact replies and dependency findings remain at
`89d19398fc580176b6101669821fc870e988c8e4:scratch/questions.md`.
Use main's QA instructions: they reflect its CPython 3.13 development setup
and wheel dependencies, superseding this branch's cp312-only and separate-Torch
instructions. Main's release and dependency changes merge unchanged.

Realign rechecked the `75796f92a` merge against its first parent and the live
snapshot implementations, bindings, stubs and lifecycle tests. The merge adds
training-record guidance and self-play metrics work, with no new snapshot
mechanism or product choice. The current plan and engine guide agree with the
implemented reset seed and shared branch constructor. No selected Wave exists;
upstream experiment memory is outside this Task's scope. Existing focused
results are reused; broader gate checks and the requested demo remain pending.

## Lifecycle compression — 2026-11-11

The identity repairs and their recorded validation were already present when
this pass started. Removed repeated branch defaults and the second exact-fork
implementation; materializations still have no replay prefix, while exact forks
retain it. Reversible choice: `selected_fork` uses `fork`'s before-reset error
message as well as its implementation. Removed the materialization test's
comparison against the same counting helper used by its subject; independent
Python observation parity and native complete-count assertions remain.
The audit now states coverage and unresolved findings without duplicating the
run history retained above. No remaining deletion targets or product questions.

- `cargo test --manifest-path managym/Cargo.toml -p managym --lib hand_snapshot -j 2 -- --test-threads 2` — all 6 passed in debug; Rust formatting passed.
- `CARGO_BUILD_JOBS=2 uv run --locked --extra play maturin build --release -i .venv/bin/python -m managym/Cargo.toml -o managym/target/wheels` — passed; matching cp313 extension replaced atomically.
- `uv pip install --no-deps --target <temporary>/installed <cp313 wheel>`, then outside the checkout `uv run --no-project --python <root>/.venv/bin/python -m pytest --confcutdir=. test_hand_snapshots.py test_search_command_evidence.py -q` — 17 passed in 1.43 s. Installed extension origin, both imports, packaged stub and guide example passed; development dependencies reused, temporary files removed. Formatting and diff whitespace passed. No publication or Task movement.

## Publication verification — 2026-11-11

`lf sync` completed cleanly against main `9f69a96a1`; code under test is
`76728c86b`. This pass changes only audit/working notes and prepared PR copy.

- `CARGO_BUILD_JOBS=2 cargo nextest run --manifest-path managym/Cargo.toml --profile ci --test-threads 2`: 1,057 passed, two skipped in 259.852 s; `/tmp/mtg272-final-rust.log`.
- Affected Python suite (all `tests/env`, belief audit/sampling/learning, changed training tests, both recovery suites, local-update/search-command suites, model agent, rollout/value and parity/seat-balance): 343 passed, three failed in 698.34 s; `/tmp/mtg272-final-python.log`. Failures are exactly the three retained `test_resource_limit_receipt_replays_its_retained_encoding_boundary` cases returning `binding_mismatch`, already reproduced on base `402dfd5b2` in the earlier record. Recovery through merged outcome tables passes; the earlier Etude missing-file failure did not recur and is not claimed fixed.
- All-target/all-feature Clippy with `PYO3_PYTHON` set to the root 3.13 interpreter passed; `/tmp/mtg272-final-clippy.log`. Rust format, Ruff check/format on all 33 changed Python/stub files, release metadata check and diff whitespace passed.
- `uv run --no-sync python scripts/release/prepare.py /tmp/mtg272-final-release-source`, then root `uv run --locked --extra play maturin build --release` for isolated 3.12 and 3.13 interpreters: both release wheels built; `/tmp/mtg272-final-release-build.log`. Root development environment remains CPython 3.13.7 / Torch 2.10.0 / Etude 0.2.15; no extension source changed in this pass.
- Fresh independent macOS arm64 installs, outside the checkout with PYTHONPATH unset and Python isolated mode: 26 snapshot, bookkeeping and Command tests passed on each interpreter. CPython 3.12.12 / Torch 2.10.0 / Etude 0.2.15: 17.58 s; CPython 3.13.7 / Torch 2.15.0 / Etude 0.2.15: 4.02 s. Logs `/tmp/mtg272-final-consumer312.log` and `/tmp/mtg272-final-consumer313-release.log`.
- Both installed consumers passed the unchanged guide example, import-origin/alias and packaged-stub checks, pending-seed/independent-branch demonstration, and `scripts/release/smoke.py` (CLI, gradients, CPU training, checkpoint reload). Logs `/tmp/mtg272-final-contract{312,313}.log` and `/tmp/mtg272-final-smoke{312,313-release}.log`.
- The first training-smoke attempt used a development wheel and failed because it lacked `.manabot-source.json`. Using the documented release-preparation path supplied that manifest; both fresh release consumers then passed. No product code workaround or fabricated source identity was used.

Assumption: publish with the three documented baseline replay failures, as in
the earlier publication, while making the current failures explicit. The
requested API, debug suite, recovery paths and isolated consumers pass. Linux,
accelerators and hosted CI results are not certified by these local runs.
