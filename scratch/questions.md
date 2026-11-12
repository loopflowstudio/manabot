# 855bff — decisions and retained observations

## Direction — 2026-11-11

Jack Heart: "The benchmark should report mean rows per inference
call (environment_decisions / policy-inference calls from collection_timings)
and the idle fraction (idle_stream_steps / (calls x streams)) per size."
About `experiments/training_throughput_sizes.py`: "prefer not to edit that file".

On the offered standing hosts: "i dont really thin kany of these options are
acceptable tbh". On the reference-commit alternative: "no i think you have it
right. this is a strong reason to have at least one more stable gpu. maybe aws
gpu perf is more consistent?"
Source: comment `262b941c-2188-4426-8ff0-35f7e52eaa10`.

This supersedes the former host-choice question: baseline means commit plus
suite interface; each comparison measures fresh controls in one job on one
machine (one pod for a rented GPU). Provider selection remains outside this
Task. No spending, rental, launch or artificial contention is authorized.

## Headless choices

- Deadline repair: 7,200 seconds standalone, including recording; 36,000 seconds
  comparison on every device. These failure bounds leave headroom over ≈5,500 s
  summed completed CPU attempts and the ≈9 h comparison estimate. Neither is a
  complete invocation measurement. Keep explicit overrides and workload identity.
- Keep the existing CLI and receipt workflow. Baseline metadata has exactly
  `commit` and `suite_interface: 1`; source admission reads a literal without
  importing the other checkout. No old baseline existed to migrate.
- Shared evaluation admits CPU/CUDA; MPS is explicitly training-only. Keep
  cumulative batching counters labelled as including warmup rather than changing
  the training hot path or introducing a second metric store.

No product question remains for demo. `scratch/plan.md` owns the clean full-ladder
proof, committed baseline and real interleaved comparison still required.
CUDA/MPS remain unmeasured; no speedup claim is established.

## Early CPU proofs (before main sync)

Both ran on dirty `402dfd5b2c8aaca6084b5dec446d058d249cddad`; `.runs/` receipts
are local harness proofs, not frozen experiment evidence.

- `uv run --locked --extra evaluation python scripts/bench_training_update.py --suite --size w64 --streams 2 --updates 2 --seconds 180 --out .runs/pipeline-speed-proof`
  completed training; measured update two: 6.144 s / 399 learner decisions.
  Whole-stage: 541 calls, 1.743 rows/call, 0.1285 idle. Evaluation rejected wrong
  authored role names; failed receipt retained, exit nonzero at 32.34 s. The panel
  was corrected to `elemental_bg` / `elemental_gr` with native admission coverage.
- `uv run --locked --extra evaluation python scripts/bench_training_update.py --suite --size w64 --workload evaluation --seconds 180 --out .runs/pipeline-evaluation-proof`
  completed and replayed four games; invocation 13.15 s, rate 0.5197 games/s,
  replay 1.729 s. Both supervisors completed cleanup.

Later default-ladder QA retained a timeout and separate a384l8 completion,
including an accidentally overlapping run that perturbed w64. Exact readings,
commands and limitations are in `scratch/qa.html`; the plan retains the runtime
facts that informed deadline selection. No complete clean ladder was recorded.

## Archived context

Full pre-compression uncommitted notes were moved to
`.runs/compress-context-20261111/{plan,questions}.md` before rewriting them.
Earlier implementation checks and superseded host assumptions remain there;
the current plan keeps the outstanding obligations.
Inherited b015d5 execution/billing notes are historical at
`402dfd5b2c8aaca6084b5dec446d058d249cddad:scratch/questions.md`; main's unrelated
MTG-271 notes remain at `9f69a96a1ca5f0c03a9df5af7217138390d415d9:scratch/questions.md`.
Neither selects work or grants execution authority here.

## Sync resolution — 2026-11-11

Keep this branch’s benchmark decisions; main’s unrelated MTG-285 release notes
remain at `668555c095c61709191ee7c4c59ad767b6903317:scratch/questions.md`.
Combine both Infrastructure memory additions, retaining their separate historical
coverage records. Neither release notes nor historical coverage select new work.

Sync onto `4842222e3f461324ff2c46937466e97761ce72e3` deletes this file on
main. Retain it here because the active benchmark plan still references these
branch-specific decisions and evidence; keep main's release and licensing changes.
