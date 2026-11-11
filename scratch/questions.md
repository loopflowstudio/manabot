# MTG-277 native heuristic

Outcome: one native heuristic owner for scalar arena players and
`VectorEnv(opponent_policy="heuristic")`, preserving viewer visibility, chooser
semantics, strategy, deterministic ties, optional prefix capacity and rewards.
No rules/world/checkpoint/input-layout change; frozen `anchors.py`, historical
registrations and results keep their bytes and identities. No Wave.

Jack Heart (2026-11-11), recorded MTG-272 direction: "do C in this chagne,
A in a nother diff (start it now)". C removes Passive there; A is this port.

## Approach and deletion boundary

`managym/src/agent/heuristic/mod.rs::Heuristic` owns rules, fallback and combat
memory; `knowledge.rs` compiles public definition ratings once, borrowing the
serialized schema without a second effect vocabulary. Modal bodies remain
opaque to ratings. The scalar `manabot/sim/heuristic.py::HeuristicPlayer` adapts
the native selector. `Env` owns lazy batch memory, clones it on forks and clears
it on reset; scalar arena games construct fresh players.

The selector consumes emitted live viewer Observations, including combat events
and `action_space.focus`. Canonical viewer JSON omits that focus, so parity
fixtures replay decisions. It reads neither Game state nor private hand snapshots.
`docs/engine.md` and `docs/evaluation.md` describe behavior and identity.

Fresh registration uses native-v2 / heuristic-cpu-v2 and hashes the adapter plus
three Rust files via `heuristic_source_paths`; validation still rejects drift.
Wheels ship all `managym/src` at original relative paths: both heuristic identity
and `engine_source_paths()` need those bytes. No hashing fallback or generated
source copy.

### Delete — do not maintain

Python heuristic rules, derived knowledge, combat memory and their exclusive
namespace fixtures are removed. Do not restore them: native rule tests and
retained decision parity cover the surviving owner. The wheel's `managym/src/**`
exclusion is removed. No further deletion target remains in this cut.

Reversible compress choices: retain approximate strategy; share card-choice
ranking and symmetric fight scoring; let `Board::creatures` yield observation
order directly, collecting only for sorting/reuse. The bounded benchmark derives
receipt counts from its actual slots/steps/seed. These do not change strategy.

## Remaining acceptance

On 2026-11-11, realign read MTG-272's direction and PR #458's published diff.
PR #458 was OPEN with no merge commit, also at the packaging pass. It removes
Passive and changes HandSnapshot, capture-generation bookkeeping, explicit-seed
reset and Gym auto_reset. Its description retained the older hand-target name;
Task comments and published code carried the newer contract. Remote main
ac601834d was already an ancestor here; MPS changes were integrated.

1. Integrate MTG-272 after landing, before publication. Preserve live-observation
   selection and lazy memory while reconciling Env constructors/forks, vector
   reset/capture, bindings, stub, wrapper and docs. Verify reset/reward/chooser
   behavior. Final policies are none/random/heuristic; Passive remains here,
   so the docs' three-value API is prospective until integration.
2. Gate: full debug Rust suite, fmt, all-target/all-feature clippy; affected
   Python simulation/evaluation, package-boundary and vector suites, plus
   `tests/env/test_observation.py`, `tests/training/test_action_cutoff.py` and
   `tests/verify/test_native_buffer_capacities.py`. Focused passes below do not
   complete these suites.
3. Normal same-runner Engine Budget must establish unchanged unused-policy
   cost/digests. Bounded collection below establishes neither a training-speed
   threshold nor strength. No artificial load or paid execution.

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

## Packaged identity proof (2026-11-11, before this compress)

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

## Bounded collection evidence (2026-11-11, before this compress)

Command: `uv run --no-sync python scripts/bench_heuristic.py --setup <basic|elemental> --opponent <random|heuristic>`.
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

## This compress verification (2026-11-11)

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

`uv run --no-sync python scripts/bench_heuristic.py --setup elemental --opponent heuristic`
— completed under a 60-second deadline: 256 transitions, 7 terminals,
0.027512 s construct/reset, 0.041456 s stepping, 6175 transitions/s. Same declared
settings as the prior table; this single smoke sample is not a speed comparison.
Full release training smoke, full suites and Engine Budget were not rerun.
