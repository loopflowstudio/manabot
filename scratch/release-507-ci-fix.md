# PR #507 release metadata failure (2026-11-12)

Failed head: `9b6cc73e93c95f5061000636b37df67f3df003dc`.
Sync completed at `4d54d47bb235cda8f6707c50ae203d9f48c929c6`.

Run 43346467367 preflight rejected `managym/pyproject.toml` because it
remained 0.2.2 while manabot was 0.2.3. Run 43346467478's Python Unit Tests
reported two failures in `tests/infra/test_release.py`: the exact engine
dependency still required 0.2.2, and paired versions disagreed. Both failures
reproduced locally. Main at `f1b0109ae` passed Python Unit Tests.

Aligned the engine Python project, three Rust crates, exact engine dependency
and both lockfiles at 0.2.3. Added the paired-version and local release-check
requirements to the release guide.

Validation: release metadata check with `--tag v0.2.3` passed; all 37 release
tests passed in isolated CPython 3.12 and 3.13 pytest environments. A fresh
`uv sync --locked --extra dev` built and installed both packages at 0.2.3.
`uv lock --check`, locked offline Cargo metadata and `git diff --check` passed.
