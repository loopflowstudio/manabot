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

`uv run --no-sync manabot train --recipe local --out /tmp/unused`: exit 2,
replacement and resume guidance, no training (also with etude.stopping and
self_play imports blocked). Exclusive CLI tests removed; mixed files keep
tracker, learner, replay/provenance and saved-record tests.
Validation: test_train_command 5 passed; torch import boundary test passed;
Ruff clean; loopflow workflow check valid; changed YAML/Python compile; diff
check clean. Full transcript: `7634686984:scratch/implementation.md`.

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

## Realign probe against EXP-56 source — 2026-11-13 evening

Seen: MTG-348 done; this branch merged main at 0.3.2 (`c3dfd82476`). 0.3.2
still has `etude.run` and no Run.resume/steps/save. EXP-56 checkout at
9a762a1 plus uncommitted edits (HEAD 98c8f05 by the end) has local
`etude run`, explicit `--resume`, job seeds and `etude.tracking`.
In a throwaway worktree of this HEAD with only `etude.run.` → `etude.tracking.`
rewritten, EXP-56 `src` on PYTHONPATH (nothing committed):
- every loop/recipe module imports (self_play, minimal, local, elemental,
  belief_policy, belief_sampler, baseline, eval.benchmark).
- `etude run manabot/recipes/local.py --name smoke`: exit 0, run
  `track/local/smoke-seed-0/` with `out/policies/0000000{0,1,2}-{raw,ema}.pt`,
  `out/state/current.json` → `save-00000001/{state.json,learner.pt,optimizer.pt,ema.pt}`,
  checkpoints updates-0..2. Matches the smoke callers' expected path.
- `--resume smoke-seed-0 --until updates=3`: exit 0, adds 00000003 policies,
  earlier ones kept, save-00000002 replaces 00000001, outcome finished.
  `--resume` with no target: `run_already_finished: give --until with a later target`.
- pytest test_self_play_loop, test_minimal_loop, test_belief_stages,
  test_recipe_etude: 13 passed (62.97s), 1 warning not inspected.
Not exercised: signal stop, continue, start= linkage, parity, launched runs.

## Proposed-API client cut — 2026-11-13

Wrote the loop/recipe/consumer cut against the proposed API; no stub or SDK
change. (Its etude.run import failure is resolved by the rename; see above.)

Validation: 21 changed/new Python sources and both workflow heredocs compile;
Ruff clean and formatted; minimal guide equals source; diff check clean.
`PYTHONPATH=. uv run --no-sync --with pytest --with 'etude-sdk==0.2.16' --with torch --with psutil --with gymnasium --with nbformat pytest tests/training/test_train_command.py tests/infra/test_release.py -q`: 42 passed (13.35s).
No background work, paid run or delivery.
