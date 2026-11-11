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

## Current alignment and remaining acceptance

Concept review (2026-11-11): **no design change and no open product choice**.
Keep one `heuristic` strategy across evaluation and collection. Its memory is
part of a player's game lifecycle, not a second policy or a persisted game
record: batch resets clear it, forks copy it, and scalar games get fresh players.
The direct selector's explicit reset remains useful because it does not own a
game. A snapshot alone cannot replace decision history: canonical viewer JSON
also omits target/cost focus. Keep live observations as the selector boundary.

Reviewed normal batch stepping, opponent terminal reward/reset, closed-view and
controlled-chooser decisions, and installed registration failure/recovery against
the implementation and existing fixtures; no runtime checks were rerun. Preserve
errors on unsupported policies, capacity overflow and source drift. Recovery uses
a matching install or a new truthful registration, never relabeled old evidence.
The existing usage docs already describe that experience; no new skill workflow
is needed. `scratch/concept-review.html` shows the before/after and counterexamples.
Next show the integrated MTG-272 lifecycle and required gate evidence below;
conceptual alignment does not complete that acceptance work.

The earlier iteration's packaging defect and instant-timing mismatch are fixed:
`pyproject.toml` includes the Rust sources at their original paths,
`scripts/release/smoke.py::check_heuristic_registration` checks installed
registration and drift rejection, and the adapter header agrees with
`Heuristic::timely`. The post-compress standalone-venv proof below covers the
current heuristic sources. These are completed work, not another implementation
pass. Bounded collection evidence is retained below with its limits.

Rechecked 2026-11-11 against GitHub and both Tasks: [PR #458](https://github.com/loopflowstudio/manabot/pull/458)
was OPEN at `dfba956ac0c0824a6ebfb05b11335c293242ddea`, with no merge commit.
Its title/body now describe HandSnapshot and the reset contracts correctly.
The published diff removes Passive and the Python policy allowlist, adds
process-wide reset generations and privileged hand snapshots, reproduces every
explicit reset seed, and removes Gym auto_reset. Remote main was
`3c169916e2d642b5ead16aa303b979232635a63f`, already an ancestor of this checkout
through `04f601491`; the throughput closeout adds no heuristic requirement.

1. Integrate MTG-272 after landing, before publication. Preserve live-observation
   selection and lazy memory while reconciling Env constructors/forks, vector
   reset/capture, bindings, stub, wrapper and docs. Verify reset/reward/chooser
   behavior, including repeated explicit seeds and terminal replacement at the
   maximum seed. Preserve upstream's single native policy validation owner;
   do not restore `VALID_OPPONENT_POLICIES` while resolving the wrapper conflict.
   Final policies are none/random/heuristic; Passive remains here until that
   integration. The guide no longer claims the three documented choices are
   exhaustive before the removal lands. HandSnapshot stays outside selector
   inputs, observations and captures.
2. Gate: full debug Rust suite, fmt, all-target/all-feature clippy; affected
   Python simulation/evaluation, package-boundary and vector suites, plus
   `tests/env/test_observation.py`, `tests/training/test_action_cutoff.py` and
   `tests/verify/test_native_buffer_capacities.py`. Focused passes below do not
   complete these suites.
3. Normal same-runner Engine Budget must establish unchanged unused-policy
   cost/digests. Bounded collection below establishes neither a training-speed
   threshold nor strength. No artificial load or paid execution.

PR #458 reports three retained-encoding replay failures (`binding_mismatch`)
also reproduced on base `402dfd5b2`, and an intermittent Etude `seal_logs`
missing-file failure that passed alone. These are upstream reports, not checks
run on this branch or waivers of its gate. Compare any matching integration
failures with that evidence; keep frozen receipts unchanged. No product decision
is open for this port. Full release training smoke also remains unrun.

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

## Main sync resolution — 2026-11-11

Assumption (reversible): keep MTG-277's design, acceptance work and verification
as this branch's active context. Retain main's b015d5 closeout and operator
review below as background. Its full notes remain at
`53c2cb2f987f9768af7fd2360bb8c5f77c7113a5:scratch/questions.md`;
they do not select new work for this branch. Main's experiment protocol,
throughput summary and artifact manifest merged without manual changes.

## b015d5 closeout direction, 2026-11-11 (background)


After the evidence review and recommendation to stop this measurement phase,
Jack Heart said "sounds reasonable." He added "but i will say its probably
reasonable to still do 1x seed sometimes", then "i think that should be the
default ,but not e taht we can do 2 seeds".

Final operating guidance: default to one seed per GPU for individual-model
progress; two seeds is an option for aggregate throughput/cost. This supersedes
the operator review's suggestion of two as a starting point, not the measured
results. Existing four-seed w64 evidence remains an optional aggregate recipe.
No frozen run or training configuration changes. Close the bounded measurement
with its stated uncertainty; do not launch another campaign.

## Operator evidence review, 2026-11-11

Jack Heart said "ok, do it" after the completed C2 execution/analysis notifications.
Reviewed C2 here, without a new rental: the frozen reader reproduced readings.json
exactly, all 128 C2 artifacts matched manifest sizes/hashes, and retained rental
and teardown receipts account for all four acquired pods. No live Graph was
observed. C2 measured all four required base cells and the conditional w160l8 K4;
a384l8 K4 was excluded by memory. The scoped throughput conclusions are supported.

The selector's three extra idle rentals were a procedural failure, not authorized
scope: a failed database read was treated as a refusal. Keep their posted
$0.504125 cost in C2's ≤$1.929936 and the total ≤$9.849497. A future selector must
stop on unknown acquisition state and reconcile with the provider before any
retry. All four pods have deletion evidence. No further rental is launched here.

Recommendation: conclude this bounded measurement with its one-host limits,
rather than spend on a second host merely to resolve the marginal 1.30 admission
threshold. Two seeds is a useful large-model starting point for aggregate work;
four w160l8 seeds buy only ~7% more decisions/hour while slowing each seed 1.88×.
Four a384l8 seeds failed the admission memory rule; three was never tested, so
do not call two a physical ceiling. Keep float32. The maintained throughput
summary now carries the C1/C2 findings and supersedes its older status prose.
No claim of task-wide completion, strength improvement, universal GPU ranking,
or new user approval is made by this review.

Historical step-1/2/3 design, implementation, reviews and execution notes are at
`c2c813a475ef012bd0391e510dd7f0cb41859e83:scratch/questions.md`. Original reports,
run projections, bills and cleanup receipts remain published in
`experiments/data/training-throughput-b015d5-v1/`. Those sources retain exact
run IDs and dated direction; no historical result is being rewritten.

## QA pass (2026-11-11)

Installed-API QA passed without fixes; see `scratch/qa.html`. Batched wrapper,
scalar adapter, reseed/max seed and `manabot.eval.local --opponent heuristic`
(replay passed, native-v2 / heuristic-cpu-v2) all worked. Assumption
(reversible): the QA skill's launch notes went into a new repository
`.lf/skills/qa.md` via `lf skill customize qa`; drop it from this PR if it
should land separately.
