# MTG-277 native heuristic

Outcome: one native heuristic owner for scalar arena players and
`VectorEnv(opponent_policy="heuristic")`, preserving viewer visibility, chooser
semantics, strategy, deterministic ties, optional prefix capacity and rewards.
No rules/world/checkpoint/input-layout change by this Task. Frozen `anchors.py`,
historical registrations and results keep their bytes and identities. No Wave.

Jack Heart (2026-11-11), recorded MTG-272 direction: "do C in this chagne,
A in a nother diff (start it now)". C removes Passive there; A is this port.

## Design and deletion boundary

`managym/src/agent/heuristic/mod.rs::Heuristic` owns rules, fallback and combat
memory; `knowledge.rs` derives public definition ratings once, borrowing the
serialized schema without a second effect vocabulary. Modal bodies remain
opaque to ratings. Scalar `manabot/sim/heuristic.py::HeuristicPlayer` adapts
native selection. `Env` owns lazy batch memory, clones it on forks and clears
it on reset; scalar games construct fresh players. Selection consumes live
viewer Observations, including combat events and `action_space.focus` (omitted
by canonical viewer JSON), never Game state or privileged HandSnapshot.

Delete — do not maintain: Python heuristic rules, derived knowledge, combat
memory and their exclusive namespace fixtures. These are removed; native rule
tests and retained decision parity cover the surviving owner. Do not restore
the Python policy allowlist or Passive. Final policies: none/random/heuristic.

Fresh registrations use native-v2 / heuristic-cpu-v2 and hash the adapter plus
three Rust files. Main now distributes managym separately; its wheel must ship
`src/**/*.rs` and Cargo.lock at original package paths. No source hash fallback
or generated source copy. Prior monolithic package paths are replaced by main's
canonical engine package and compatibility alias, including the type stub.

Concept review found no open product choice. Keep unsupported-policy, overflow
and source-drift errors; use a matching install or new truthful registration
for recovery. The docs describe the built-in rules, API and identity.

## Integration and remaining work (2026-11-13)

GitHub reports PR #458 MERGED on 2026-11-11T23:46:34Z, merge
`0b296e617a87017c4cca71227749eddac0a0481b`. `lf sync --manual` integrated main
`c0de38dcd01f37223bf0ec746d9aff152ee75f2e` with conflict resolution, then
`lf sync --continue` completed. Preserve main's API/package changes and its
process-wide reset generations, originating seeds, HandSnapshot and replay
capture contracts. Exact forks and materialized worlds copy heuristic memory;
reset clears it. All source selections still take live viewer observations.
Main's QA skill supersedes the earlier local launch notes.

Reversible integration choice: source identity locates Rust files through
`managym.__file__` because engine and training now ship separately. The engine
wheel retains Rust source bytes; the training wheel stays pure Python. Frozen
experiment receipts are never rewritten. The parity fixture is a test input
bound to its world digest: a world change replaces it with a new comparison
against the original Python source, never relabeled actions.

Implementation integration and focused checks are complete (results below).
Gate remains
responsible for full debug Rust suite, fmt, all-target/all-feature clippy;
affected Python simulation/evaluation, package-boundary and vector suites plus
`tests/env/test_observation.py`, `tests/training/test_action_cutoff.py` and
`tests/verify/test_native_buffer_capacities.py`. Normal same-runner Engine
Budget must establish unused-policy cost/digests. No paid execution or
artificial load. Publication and landing belong to later caller steps.

Prior upstream PR #458 reported three retained-encoding replay failures
(`binding_mismatch`) also reproduced on base `402dfd5b2`, plus intermittent
Etude `seal_logs` missing-file failure passing alone. Those reports are not
branch checks or gate waivers. Full release training smoke was not run by the
prior implementation passes. Prior QA is in `scratch/qa.html` and concept
review in `scratch/concept-review.html`; both explicitly predate integration.
Full earlier notes, including unrelated inherited b015d5 direction and receipts,
remain at `7fa516a404:scratch/questions.md`.

## Retained checks (2026-11-11)

The original Python owner at 757bb6257 matched native on all 862 decisions in
four bounded fixtures (256-decision cap; basic red/green, Elemental UR/GW and
BG/UG; explicit prefix 2 included). One finished; three reached the cap. Golden
choices and world provenance are in `tests/fixtures/native_heuristic_parity.json`.
This is parity evidence, not strength or throughput evidence.

The first debug run caught a translation error: empty predicate lists count as
absent for Lesson damage. Native now agrees. Other initial failures were invalid
offers/names (corrected to Pop Quiz, Fabled Passage and creature-only pump
targets), or a wrapper fixture using the wrong reset/step API; no additional
behavior fix was needed. A temporary pre/post-compress comparison matched every
catalog card's full ratings and was removed afterwards.

Prior compress verification:
- `cargo test --manifest-path managym/Cargo.toml -p managym --lib heuristic -j 2 -- --test-threads 2` — 13 passed in debug, including keyword and counterspell coverage.
- Required release wheel build and atomic cp312 extension replacement passed.
- `uv run --extra dev pytest tests/sim/test_heuristic.py tests/semantic/test_player_control.py tests/training/test_action_cutoff.py::test_heuristic_scores_only_retained_legal_offers -q` — 23 passed, including all 862 decisions/terminal flags and registration-capacity rejection.
- Rust fmt, focused Ruff lint/format and `git diff --check` passed. The two-slot example in docs/engine.md executed successfully.

Realign corrected docs/engine.md's timing: `Heuristic::timely` delays instants in
own upkeep/draw; fetch abilities are considered earlier. The adapter header now
agrees. Earlier detail and retired scripts are recoverable at
cac743f9e:scratch/questions.md and 757bb6257 respectively. Unrelated b015d5 study
notes remain at 757bb6257:scratch/questions.md and its linked evidence.

## Packaged identity proof (2026-11-11, before iterator simplification)

The release wheel contained all 91 Rust sources byte-for-byte. Outside the
checkout, in a fresh isolated environment with PYTHONPATH unset,
`scripts/release/smoke.py::check_heuristic_registration` registered and validated
successfully, rejected an appended newline in each of the four identity files,
and restored every file. Installed and checkout source digests both equaled
`9a873a22793313c2a992510c12474d4760a5f925f071303e7c2e602e665e6be8`.
This is the digest of that pass, not subsequent source edits.

The first probe caught macOS `/var` versus resolved `/private/var`; the smoke now
resolves both sides of its install-location check. The probe had a 60-second
subprocess deadline. Release wheel build and atomic extension replacement passed;
`uv run --no-sync pytest tests/sim/test_heuristic.py::test_heuristic_registration_versions_native_sources tests/infra/test_release.py::test_release_projects_preserve_native_paths_and_runtime_data -q`
— 2 passed; focused Ruff lint/format passed. Full release training smoke was not
rerun; its entry point includes the registration/drift check.

## Bounded collection evidence (2026-11-11, before iterator simplification)

Command (script at `1455cd5636:scripts/bench_heuristic.py`, removed by compress
on 2026-11-13: it pinned policy input v6, which main now refuses): `uv run --no-sync python scripts/bench_heuristic.py --setup <basic|elemental> --opponent <random|heuristic>`.
Four fresh processes, each with a 60-second deadline, finished in 0.788–0.857
seconds including imports. Same host: macOS 26.0.1 arm64, world w17, release
extension, CPU, one Torch thread, seed 213, two slots, 128 batch steps (256
learner transitions), default skip-trivial, v6/512 facts, no warmup. The training
wrapper includes encoding, tensor views, rewards and auto reset. Learner takes
the first legal offer; no inference or updates. Basic decks match parity;
Elemental is UR versus GW.

| Setup | Opponent | Construct/reset seconds | Step seconds | Transitions/s | Terminals |
| --- | --- | ---: | ---: | ---: | ---: |
| Basic | random | 0.026650 | 0.018142 | 14111 | 6 |
| Basic | heuristic | 0.022819 | 0.017022 | 15040 | 8 |
| Elemental | random | 0.025308 | 0.042664 | 6000 | 4 |
| Elemental | heuristic | 0.023394 | 0.028335 | 9035 | 7 |

Startup excludes imports; lazy knowledge initialization is included at first use
(reset or stepping). Single short samples follow different trajectories with
different terminal counts: ratios do not measure selector speedup. These rates
show bounded native-opponent collection, not sustained training throughput,
strength, unchanged unused-policy cost or Engine Budget acceptance.

## Post-compress verification (2026-11-11, retained)

`Board::creatures` now returns an iterator; ranking/reuse callers alone collect.
The benchmark declares slots/steps/seed once and derives its receipt counts.
No strategy, tie-breaking, visibility or identity algorithm changed.

- `cargo test --manifest-path managym/Cargo.toml -p managym --lib heuristic -j 2 -- --test-threads 2` — 13 passed in debug.
- `CARGO_BUILD_JOBS=2 uv run --extra play maturin build --release -i .venv/bin/python -m managym/Cargo.toml -o managym/target/wheels` — passed; cp312 extension replaced atomically; all 91 packaged Rust files match checkout bytes.
- `uv run --no-sync pytest tests/sim/test_heuristic.py tests/semantic/test_player_control.py tests/training/test_action_cutoff.py::test_heuristic_scores_only_retained_legal_offers tests/infra/test_release.py::test_release_projects_preserve_native_paths_and_runtime_data -q` — 24 passed, including all 862 retained decisions.
- Rust fmt, focused Ruff lint/format and `git diff --check` passed.

The supplemental `uv --isolated --with` installed probe stopped before mutation:
its package overlay lives outside `sys.prefix`, so the smoke's installation guard
correctly rejected it. Use a standalone venv for this proof; do not weaken the
guard or mutate uv's package cache.

Standalone venv proof passed outside the checkout with PYTHONPATH unset and
copy-installed wheel/Torch: `scripts/release/smoke.py::check_heuristic_registration`
registered, validated, rejected drift in all four bound files and restored them.
Installed and checkout digest:
`925efebbdbea044a9b7144d337d98fe2285725b52f0afa788914440f18bd5fce`.
Each subprocess had a 60-second deadline; the temporary venv was removed.

Realign rechecked `source_bundle_sha256(heuristic_source_paths())` on 2026-11-11:
it still matches that installed-wheel digest. Only the plan and the guide's
premature exhaustive policy list changed in this pass; no runtime tests or
benchmarks were rerun. `git diff --check` passed and `lf context` reports scratch
within both configured limits.

`uv run --no-sync python scripts/bench_heuristic.py --setup elemental --opponent heuristic`
— completed under a 60-second deadline: 256 transitions, 7 terminals,
0.027512 s construct/reset, 0.041456 s stepping, 6175 transitions/s. Same declared
settings as the prior table; this single smoke sample is not a speed comparison.
Full release training smoke, full suites and Engine Budget were not rerun.

## Integration findings (2026-11-13)

Initial focused run: 50 passed, one retained-parity failure at decision 144 of
Elemental BG/UG, seed 213 (native 0, old receipt 7). Main changed game behavior
and card digest while retaining world label w17. A temporary in-memory load of
original Python source `757bb6257ef07cf2536ad7f7ec24f0f7309628c6` compared it with
native on the integrated live observations: all 862 decisions matched across
the same four bounded cases, including prefix 2. One game finished, three hit
the 256-decision cap. No temporary Python rule implementation is retained.
`tests/fixtures/native_heuristic_parity.json` records that new comparison
and world digest; the pre-sync choices, which no test could run, remain at
`1455cd5636:tests/fixtures/native_heuristic_parity.json`. The replay test checks
world identity before interpreting expected actions. This supersedes the
initial assumption that old golden choices would remain executable after sync.

The first release-wheel check found broad `**/cards/**` and `**/state/**`
exclusions also removed engine `src/cards` and `src/state`. Narrowed exclusions
retain all engine source files for provenance and heuristic registration.

## Integrated implementation checks (2026-11-13)

- `CARGO_BUILD_JOBS=2 cargo test --manifest-path managym/Cargo.toml -p managym --lib heuristic -- --test-threads 2` — 14 debug tests passed, including forked kicker history and same-seed reset clearing it.
- `uv run --no-sync pytest tests/sim/test_heuristic.py tests/env/test_captured_vector_bookkeeping.py tests/semantic/test_player_control.py tests/env/test_opponents.py tests/infra/test_release.py::test_release_projects_preserve_native_paths_and_runtime_data tests/training/test_action_cutoff.py::test_heuristic_scores_only_retained_legal_offers -q` — 51 passed after adding the new-world parity receipt; includes max-seed terminal/reward parity, all-policy reset/capture prefixes and controlled chooser/closed-view selection.
- Required release wheel build and atomic matching cp313 extension replacement passed. All 96 `managym/src/**/*.rs` files match wheel bytes. `scripts/release/prepare.py` plus `uv build --wheel --no-sources` built both staged distributions; the staged engine wheel also contains all 96 matching files.
- Outside the checkout, with PYTHONPATH unset, copy-installed staged wheels in a fresh standalone venv passed scalar/batch heuristic API use and `scripts/release/smoke.py::check_heuristic_registration`: register, validate, reject each of four source edits, restore and revalidate. Installed and checkout digest: `c8c5657031e523ad8fa46a420726a002bf9495d08e600b04711a0ca820b77079`. Probe deadline 60 seconds; temporary venv removed. Ordinary resolution selected Torch 2.15.0, Etude SDK 0.3.1, NumPy 2.5.4 and Pydantic 2.14.3; this is not a minimum-dependency or second-interpreter proof.
- Rust formatting, focused Ruff lint/format and `git diff --check` passed. `lf context` is within both limits.

Remaining: gate's full Rust/Python suites, all-target/all-feature clippy,
normal same-runner Engine Budget, and full release training smoke. No publication,
landing or Task completion in this implementation step. No background work remains.

## Compress (2026-11-13)

Removed `scripts/bench_heuristic.py` (unrunnable on main, one-off; its numbers
stay above) and the unexecutable pre-sync parity fixture; the integrated
comparison now owns the one fixture path. Test imports moved to the top. No
Rust or runtime change, so no rebuild.
`uv run --no-sync pytest tests/sim/test_heuristic.py -q` — 15 passed.

## Second sync (2026-11-13)

Branch merged main `d1554835be` (MTG-337, `de39d601fb`: large-board benchmarks,
fewer mana-payment scans with unchanged search order, `Observation.from_json`
optional `continuation_json`). It touched no heuristic, opponent or vector-env
file. The installed extension predated the merge, so realign rebuilt it
(release, atomic cp313 replacement) and reran
`uv run --no-sync pytest tests/sim/test_heuristic.py tests/env/test_captured_vector_bookkeeping.py tests/semantic/test_player_control.py tests/env/test_opponents.py tests/training/test_action_cutoff.py::test_heuristic_scores_only_retained_legal_offers -q`
— 50 passed; the retained parity fixture still replays. Gate's remaining checks
above are unchanged, and Engine Budget now compares against main's reference
`d53220d36`.

## Concept review after integration (2026-11-13)

No open product choice. Actions, runners, memory lifecycle (Env reset clears,
fork copies, `build_code_player` builds a fresh scalar player per game) and
registration identity hold after both syncs; no code-behavior change wanted.

Direction taken — one owner for the rule description. The adapter docstring
stated the rules where they no longer run, the native header defined itself by
the deleted Python implementation, and the copies had already drifted once
(realign's instant-timing correction). The `//!` header of
`managym/src/agent/heuristic/mod.rs` now states the rule groups, fallback,
tie-breaks and approximations as its own; `manabot/sim/heuristic.py` documents
only the adapter, why it sits outside frozen `anchors.py`, the identity files
and a pointer to the header and docs/engine.md; `Heuristic`'s doc comment says
Env clears batch memory on reset and scalar games construct fresh players.
Parity with Python source `757bb6257` is evidence (the fixture), not the
definition. Comments only: no behavior, API or fixture change. The source
digest moves (no registration exists under it yet). Assumption: the Task's
"keep the documented rules in the module header" means the module that owns
the rules.

`CARGO_BUILD_JOBS=2 uv run --extra play maturin build --release -i .venv/bin/python -m managym/Cargo.toml -o managym/target/wheels`
— passed; cp313 extension replaced atomically. Compress rechecked: all 96
packaged Rust sources and the installed extension equal wheel/checkout bytes,
and the header's tie-break and fallback claims match `best_card`, `priority`,
`learn`, `target`, `plan_blocks` and `fallback`; no further cut found.
`uv run --no-sync pytest tests/sim/test_heuristic.py -q` — 15 passed.
Workspace Rust formatting, adapter Ruff lint/format and `git diff --check`
passed. Full suites, release smoke and Engine Budget remain for gate; no
background work remains.

## Realign after concept review (2026-11-13)

The concept-review direction is implemented in `89b732137a`: the native
`//!` header owns rules, fallback, tie-breaks and approximations; the adapter
is 43 lines (adapter, frozen-anchor separation, identity files); the
`Heuristic` doc comment states reset/fresh-player lifecycle. Rechecked without
rebuilding: the installed cp313 extension equals the newest wheel's, and all 96
packaged `managym/src/**/*.rs` equal checkout bytes. docs/engine.md agreed with
the header except its fallback paragraph, which named the separate frozen
`ScriptedGreedyPlayer` anchor and omitted the all-decline case; corrected
(docs only, no identity file). Remaining is unchanged: gate's full debug Rust
suite, clippy, affected Python suites, full release training smoke and normal
same-runner Engine Budget.

## Second concept review and README correction (2026-11-13)

No open product choice at `5c6b998b4d`: opponent selection, Env-owned batch
memory, the scalar adapter, `native-v2` identity and the drift/overflow/
unsupported-name errors need no change. `managym.Heuristic.reset()` stays:
direct native callers and the parity test reuse one selector across games.

Its one finding is fixed: `manabot/README.md` carried a third rule list and
defined the fallback by `scripted_greedy`, though nothing binds native
`fallback` to `ScriptedGreedyPlayer.act`. The paragraph now names the rule
groups and links `docs/engine.md#built-in-heuristic-opponent`; selection
sentence, historical-version statement and MTG-152 figures are intact. Prose
only (no identity file, build or runtime test); `git diff --check` passed.
Full suites, release smoke and Engine Budget remain for gate.

## Third concept review (2026-11-13)

At `00e2090eb7`: nothing to change, no open product choice. Reread the adapter,
native header, bindings/stub, `OpponentPolicy`, `register_anchor` and the three
docs. The header owns the rules; `docs/engine.md` is the one user guide and
agrees with it (rules, ties, fallback); README and evaluation guide only link.

## QA after third concept review (2026-11-13)

At `00e2090eb7`, real surfaces only; no code change. Docs snippet ran as
written. Native batch, 200 games per arm, random learner in seat 0, viewer
always seat 0: random opponent W102/L99; heuristic UR v GW W11/L189, BG v UG
W28/L172; rewards ±1. `manabot train` smoke checkpoint →
`manabot.eval.local --opponent heuristic` exit 0, `native-v2`/`heuristic-cpu-v2`,
replay passed (548 decisions, 0 mismatches, 0 private exposures); `--request`
repeat matched all 1,096 decision records. Academy Watch (heuristic v random,
seed 42) played to "seat 1 wins" at decision 337 with no page errors; servers
stopped. Bad names and capacities raise ValueError. Host load ≈500, so no speed
evidence. Report: `scratch/qa.html`.

## Required per-seat API (2026-11-13)

Jack Heart: "Implement the requested per-seat agent/random/heuristic API before delivering PR #518".
Steer c9370d68 additionally requires shared scalar/vector configuration, native
chooser semantics, viewer-safe selection, independent seat memory and bounded
two-built-in stepping. Etude eval migration belongs to Task 050eb6; no other
checkout is edited.

Reversible design choice: `player_policies=("agent", "heuristic")` belongs to
Env/VectorEnv construction, separate from deck/rules PlayerConfig so controller
selection never changes world hashes or frozen setup serialization. The native
Env owns selection, settlement and memory for both scalar and vector callers.
Delete — do not maintain: vector-only opponent advancement and one shared
heuristic memory slot. Preserve the public old argument through a warning with
removal in 0.3.0; retained receipt spellings remain readable.
Two built-ins reset without playing, then advance one decision per step
(scalar None; vector -1). Mixed games settle to the external chooser with a
10,000-decision error bound and never loop resets. Vector terminal autoreset
remains one replacement. Mixed rewards belong to the external seat; two-built-in
rewards belong to the originating chooser.

Implementation and focused validation are complete. Continue the existing
Graph through gate and authorized delivery; no further API/demo approval.
Gate still owns affected suites/full debug Rust checks, clippy, normal Engine
Budget, release training smoke and the consumer interpreter matrix. Preserve
these results when preparing the PR, then delete scratch in the pre-land commit.

Session eb0b316fe3df4c8eae1c561f49cd9d8f relayed Jack Heart's reply on
2026-11-13, after confirming `player_policies` defaults to `(agent, agent)`:
"lg2m, approved". Steer a0c2b822 authorizes remaining validation, publication,
gate, merge and post-merge verification without another API/demo approval;
delete scratch in the pre-land commit and preserve frozen evidence.

Final native debug check: `CARGO_BUILD_JOBS=2 cargo test --manifest-path managym/Cargo.toml -p managym --lib heuristic -- --test-threads 2` — 17 passed (5.86 s test execution). Includes 256 live Elemental decisions against independent selectors, scalar/vector terminal parity and both seats' memory lifecycle. Unused-policy cost remains for Engine Budget.
Release build initially caught use of pre-Bound PyO3 methods in the deprecation warning; corrected to the repository's `import_bound` / `get_type_bound`. The corrected required release build passed; matching cp313 extension replaced atomically and all 96 packaged Rust sources matched checkout bytes.

- `uv run --no-sync pytest tests/env/test_native_player_policies.py tests/sim/test_heuristic.py tests/env/test_captured_vector_bookkeeping.py tests/env/test_opponents.py tests/semantic/test_player_control.py tests/training/test_action_cutoff.py::test_heuristic_scores_only_retained_legal_offers -q` — 70 passed in 13.41 s; 15 expected legacy-argument deprecation warnings.
- Fresh standalone copy-installed wheel outside the checkout, PYTHONPATH unset, 120-second subprocess deadline — all nine scalar/vector controller pairs, terminal rewards, fork/reset, 256-decision live Elemental parity and deprecation passed; manabot and Torch absent. Temporary venv removed. This is engine-only API proof, not the full release training smoke or a CPython 3.12 proof.
- Workspace Rust fmt, focused Ruff lint/format and `git diff --check` passed. No servers, watchers or paid jobs were started.

Positional mixed stepping settles built-ins to the external chooser. Atomic
Command/structured submissions retain their one-command receipts; if the next
chooser is built-in, `Env.step()` continues native play. Mixed terminal rewards
always use the external seat, including this continuation; two built-ins use
the acting chooser. Fixed-player projections omit events, so built-in stepping
retains the previous live projection's combat events, copied on forks and
replaced on reset. The native independent-selector parity test exercises this.
