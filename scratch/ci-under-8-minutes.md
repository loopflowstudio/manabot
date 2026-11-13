# CI under 8 minutes (Task 38ef25; includes MTG-263)

## Outcome and constraints

Every ordinary Ubuntu PR job finishes in under 480 seconds, including install
and build, demonstrated on the PR revision. Preserve test coverage and the
300-second single-test limit; never induce load. Engine Budget retains its
same-runner change/reference comparison and is exempt; the macOS nightly is
outside this budget.

## Fix-forward plan (2026-11-12, after #506)

PR #506 merged at `60664d149`; its exact push run `43341403551` failed
the live experiment/notebook fixture with `training resource wall deadline
exceeded` (129 passed, one failed). All ordinary jobs stayed below 480 s.
The earlier green PR does not supersede this failure. Full evidence is in
`scratch/ci-post-merge-verification.md`.

Bound the fixture's four real training attempts to one update of eight
one-seat learner transitions on each of two streams using `exact-quota`.
Give each training stage/regime a 120-second resource allowance for startup
and parallel scheduling. These are failure guards, not work targets; the
300-second pytest watchdog and job budget stay unchanged. Retain all four
complete live evaluation cohorts, overlap, accounting, notebook edits and
outputs, and resume assertions. Assert each saved training record contains
one update and 16 learner transitions.

**Delete — do not maintain:** the inherited full-game training closure and
30/60-second guards have been replaced. Compression removes the singleton
stage loop and pins its one-update workload locally; its stage deadline reads
the regime allowance. No remaining deletion targets. Production paths, the
shared baseline, frozen evidence and existing assertions/selections stay intact.

Remaining: affected file/report validation and Ubuntu CI timing in gate;
publish a fix-forward PR in this Task through
the existing delivery steps, preserving #506 as history.

Sanity checks:

- `lf sync --manual --json` — synced main through `6b400f95f` into this checkout (`9690bf60b`); existing scratch edits retained.
- Focused test before rebuilding — four learners failed at catalog admission (`catalog_tokens.keys: expected array`), identifying the stale local native extension rather than reaching collection.
- `uv run --extra play maturin build --release -i .venv/bin/python -m managym/Cargo.toml -o managym/target/wheels` — passed in 46.12 s; installed the matching cp313 extension atomically. No Rust source changed; Ubuntu CI must exercise its debug extension.
- `uv run --no-sync pytest tests/training/test_experiment_runner.py::test_tiny_comparison_evaluates_live_and_notebook_refresh_preserves_edits -q --durations=1` — 1 passed in 45.42 s (43.98 s test body), including all four real updates and monitoring attempts, live overlap, notebook execution/edit preservation and resume. Local timing does not certify Ubuntu.
- `uv run --no-sync ruff check --fix tests/training/test_experiment_runner.py` and `uv run --no-sync ruff format tests/training/test_experiment_runner.py` — passed; `git diff --check` passed.
- Compression: `uv run --no-sync pytest tests/training/test_experiment_runner.py -q --durations=3` — 10 passed in 151.36 s; live comparison 82.03 s, diagnostic exports 66.86 s. Ruff check/format and `git diff --check` passed. All four training receipts still assert one update and 16 learner transitions. No process remains running.

Jack Heart's Task comment (2026-11-12): "Read its brief and deliver its outcome
here; say so in the PR so MTG-263 can be closed with it."

MTG-263 brief:

"Measure per-file durations, then either balance file shards across matrix
groups or adopt pytest-xdist where tests are process-safe (check tmp dirs,
ports, DuckDB files, subprocess fixtures). Target: no group above ~8 min wall
time; report before/after from CI runs. Do not drop or weaken tests."

## Surviving implementation and evidence

PR #506 delivered six measured Integration groups with logical-core pytest
workers, one dependency sync/native debug build per Integration job, the
per-worker 300-second watchdog, injected-time admission tests and a two-stream
real-worker size smoke. `docs/development.md` owns the ordinary Ubuntu job
budget; `TESTING.md` owns process isolation. These are already on main.

| GitHub Actions run | Revision / event | Result and slowest ordinary job |
|---|---|---|
| 43312163427 | main baseline | green; Integration core 1443 s |
| 43327574361 | `45fe9f866`, dispatch | green; experiment 404 s |
| 43332039325 | `f90ae2893`, dispatch | green; recovery 416 s |
| 43334548912 | `30cfcb2e6`, dispatch | green; evaluation 448 s |
| 43337105047 | `1fc4e1b24`, PR | green; recovery 349 s |
| 43341403551 | `60664d149`, merged push | experiment failed at 433 s; all other ordinary jobs passed, max 418 s |

Runs: `https://github.com/loopflowstudio/manabot/actions/runs/<id>`.
The exempt Engine Budget passed in 613 s on the PR and 685 s on the merged
push. The full post-merge failure and timing record is in
`scratch/ci-post-merge-verification.md`; it supersedes the old green gate and
demo decision. No human demo or approval was recorded.

Contrary findings and limits:

- One recovery runner spent 145 s installing dependencies versus 17–31 s on
  its peers; unchanged evaluation-long tests varied from 153 to 249 s. The
  32-second margin in dispatch 43334548912 is not a timing guarantee.
- Earlier attempts failed because complete-game collection was not bounded
  by its transition floor, and a threaded admission fixture had a 30-second
  future timeout. Both causes were repaired before #506; the post-merge
  experiment failure is separate.
- The historical controlled watchdog proof used a temporary five-second
  limit: five tests totaling 11 s passed with two workers; a 30-second test
  printed stacks, crashed its worker and exited 1. Serial behavior matched.
  Temporary fixtures were removed. This remains evidence for unchanged code.
- The merge audit retained all 69 original selected paths/nodes and ran all
  six evaluation exclusions in evaluation-long. New test files outside a
  selected directory still need explicit workflow selection; no new coverage
  guarantee or hard eight-minute job timeout was introduced.

Detailed prior measurements, command results and gate history remain at
`937f4f152:scratch/ci-under-8-minutes.md`. This plan keeps the current repair
and the evidence that still constrains it.

## Remaining delivery

Gate must validate the affected file/report suites and obtain Ubuntu timing
with its debug extension; local release-extension results do not certify CI.
Publish the fixture-only fix-forward PR in Task 38ef25, retaining #506 as
history. Existing `scratch/pr-body.md`, `scratch/pr-title.txt` and
`scratch/.pr-copy-ref` describe #506 and must be rewritten at delivery.
The historical QA page's gate reference points to the archived plan above.
The Task stays open until post-merge verification succeeds.
