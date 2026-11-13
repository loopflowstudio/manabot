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

Removed the release smoke's `train --help` call, inlined the local recipe's
update default, corrected the development guide's `--recipe` mention, restored
the dated `experiments/baseline-nightly.md` to main, and merged six historical
scratch pointers into plan.md.
`uv run --no-sync --with pytest --with typer pytest tests/training/test_train_command.py tests/infra/test_release.py -q`: 42 passed (1.96s); ruff clean on both changed Python files.

## Realign — 2026-11-13

Seen: EXP-56 checkout at ddf6df5 (mid-merge with conflicts) has no
resume/steps/save/input or run.seed; fe0d95c returned state publication to
design review over an interrupted multi-file save. Etude main released 0.3.0
and 0.3.1; this branch and main still pin 0.2.16; MTG-348 holds for 0.3.2.
No code changed and no tests rerun in this pass.
