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
- `Env::reset` allocates process-wide capture generations. Repeated seeds,
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
No deletion targets remain. Historical receipts retain policy strings; replay
of passive needs the original source. Academy's scalar debugging policy is
outside this cut. No heuristic, fixed-view batch or search feature is added.

Breaking release-note items for MTG-271: Gym auto_reset removed; every explicit
Torch reset seed reproduces the deal; CapturedVector rejects unbuffered native
mutations; built-in passive and its decision-profile CLI branch removed.
There is no docs/releases.md; the engine guide describes the changes.

No open product question. Gate must refresh affected suites after scalar
unification; earlier publication and QA results below predate it. Publication,
landing and Task movement belong to delivery. No throughput, CUDA/MPS,
vector terminal-hand retention or full scalar-helper parity claim.

## Latest scalar unification proof

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
No Wave memory exists. Context was 4,612/12,000 tokens and 20,299/98,304 bytes.

Compression reuses `_score_belief_truth` for both single and paired audit scores,
so the same hand metrics have one implementation. The parity test reads each
scalar snapshot once. The audit drops superseded verification prose; the concept
example uses list position instead of the deleted slot field. Historical QA
transcripts retain the API they actually ran. No native source changed in this pass.

- Compression check: `uv run --extra dev --extra artifacts pytest tests/env/test_hand_snapshots.py tests/belief/test_audit.py tests/belief/test_sampling_data.py::test_real_frozen_collection_keeps_both_viewers_and_three_game_splits tests/belief/test_learning.py::test_supervision_matches_hand_and_rejects_another_revision -q` — 16 passed in 5.59 s. Ruff check/format on the two changed Python files and `git diff --check` passed.
- Pre-curation uncommitted notes were moved intact to `/var/folders/m6/r3tllnrs1yq7yfbwm680tss40000gn/T/mtg272-before-compress-5cs77nw1/questions.md`; live direction and latest proof remain above. No publication or Task movement.
