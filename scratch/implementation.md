# MTG-329 implementation evidence

## Earlier cutover evidence (not the new train API)

Full transcripts: `bd14801a72d48ad38eeca18d08fc1c9e7f99612a:scratch/implementation.md`.
The older executor extraction is implemented and uses state.pt plus the real
EXP-56 source override at 105abdd. Fresh seed-197 CPU PPO and damped-policy
runs matched retired executor main 6ce618b8a bitwise on all 87 raw/EMA tensors.
Two updates, two streams, 16-transition floor, four games, 357 decisions;
PPO two optimizer updates/179 exposures. Persistence and random streams on
resume intentionally differ. No provider/service proof follows.

Earlier focused real-SDK checks covered immutable policies, saved milestone
repair, numerical incidents, belief provenance, selection caches and tracking.
They are not proof of the revised resume/save/steps API. No local SDK stub or
source-path pin is committed. Requirements/lock still name 0.2.16.

Contrary findings: retained game-error fixture frame schema 7 vs runtime 8
fails binding_mismatch; frozen bytes were kept. A bounded rowless-exception
probe against 105abdd changed a finished run to failed after ValueError or
KeyboardInterrupt. That fix belongs in Etude.

## Independent CLI deletion — 2026-11-13

Source inspection: available Etude checkout HEAD 105abdd has none of the new
resume/steps/save/input methods or run.seed. No rebuild stub was introduced.
The train command no longer imports recipes or the self-play state code.
`uv run --no-sync manabot train --recipe local --out /tmp/unused`: exit 2,
replacement and resume guidance, no training. A separate fresh process with
etude.stopping and self_play imports blocked produced the same guidance.

The removed CLI's exclusive tests are gone; the mixed files retain tracker,
learner, replay/provenance and saved-record tests. CI smoke and installed
smoke now target the intended Etude CLI and need real-SDK validation when it
lands. No claim of a passing training smoke or installed consumer is made.

Validation:
- `uv run --no-sync --with pytest --with typer pytest tests/training/test_train_command.py -q`: 5 passed (1.64s).
- `uv run --no-sync --with pytest pytest tests/test_package_boundaries.py::test_only_listed_modules_import_torch_outside_its_backend -q`: 1 passed (0.41s).
- `uv run --no-sync --with ruff ruff check` on the eight changed Python files: clean; `ruff format` applied. The environment lacks Ruff, so it was supplied as an isolated tool overlay.
- `uv run --no-sync python scripts/check_loopflow_workflows.py`: all nine graphs compile and resolve; boundaries valid.
- `uv run --no-sync --with pyyaml python` parse/compile probe: both changed workflow YAML files, all eight changed Python sources and the emitted installed-consumer recipe compile.
- `git diff --check`: clean. `lf context`: 9,097/12,000 tokens and 37,714/98,304 bytes before these result lines, with full previous notes retained at bd14801a72.

Remaining real training/consumer checks are blocked on EXP-56, not reported
as passing. No server, watcher, provider job or child agent was started.

## Compress — 2026-11-13

CLI cut: removed the release smoke's `train --help` call, inlined the local
recipe's update default, restored the dated `experiments/baseline-nightly.md`;
test_train_command + test_release: 42 passed.
Client cut: `step` adds each update's collector deltas to `counts` in place,
so `counted`, the `base` argument and the copy-back loop are gone; `out` is
required; `learner_settings`, `create_model` and `create_learner` are inlined;
docs/training.md's loop listing and the stop test follow.
Ruff check/format and py_compile clean on the six changed files; `git diff
--check` clean. test_self_play_loop was not run: it needs the unreleased SDK.

## Realign — 2026-11-13

Seen: EXP-56 checkout at ddf6df5 (mid-merge with conflicts) has no
resume/steps/save/input or run.seed; fe0d95c returned state publication to
design review over an interrupted multi-file save. Etude main released 0.3.0
and 0.3.1; this branch and main still pin 0.2.16; MTG-348 holds for 0.3.2.
No code changed and no tests rerun in this pass.

## Proposed-API client cut — 2026-11-13

Wrote the loop/recipe/consumer cut against the proposed API; no stub or SDK
change. Real source dda6b17 has the run methods but importing self_play stops
at `manabot.infra.artifacts`: `ModuleNotFoundError: etude.run`. MTG-348's naming
migration must land before EXP-56 execution, CLI, parity or consumer proof.
The installed pin remains 0.2.16; no new training result is claimed.

Validation: 21 changed/new Python sources and both workflow heredocs compile;
Ruff clean and formatted; minimal guide equals source; diff check clean.
`PYTHONPATH=. uv run --no-sync --with pytest --with 'etude-sdk==0.2.16' --with torch --with psutil --with gymnasium --with nbformat pytest tests/training/test_train_command.py tests/infra/test_release.py -q`: 42 passed (13.35s).
No background work, paid run or delivery.
