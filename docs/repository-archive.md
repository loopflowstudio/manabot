# Repository source and retained evidence

The repository keeps current executable code, contracts and offline regression
fixtures. Historical measurements and replay corpora live in the existing
content-addressed S3 store; generated agent/runtime files stay local.

## Retired code and tooling

The 2026-10-23 cleanup retires these implementations at source
[`62d2aa9ad`](https://github.com/loopflowstudio/manabot/tree/62d2aa9ada5c0184ccbf24705dccd956a233a972):

| Removed surface | Reason / surviving owner |
|---|---|
| `.claude/commands/` | Profiling instructions for the retired C++ engine and `.design/` workflow. |
| `.lf/steps/` and `.lf/directions/` | Not discovered by the installed Loopflow catalog. Current validation is in [validation by change area](development/validation.md), linked by AGENTS; browser isolation is in [Academy](../academy/README.md#isolated-browser-validation). Dated experiment directions remain in Git history. |
| `.lf/skills/implement.md`, `kickoff.md`, `review-slice.md` | Narrow historical Academy notes were shadowing general skills. Current runtime guidance is in Academy's README; the standard skills now resolve normally. |
| `.lf/flows/elemental-*` | Hard-coded sequences for completed and merged MTG-161 through MTG-164. |
| `.lf/flows/auto-code.yaml` | Invalid: it attempted to invoke the `code` Workflow as a Flow. |
| `manabot/study/` | Superseded Allies/Lessons game recorder, measurements, report and balance experiment. The live target-inspection primitive and its behavioral tests moved to `sim/action/targeting.py` and `tests/sim/test_targeting.py`. |
| `semantic/structural.py`, `structural_katas.py`, `transfer.py` | Historical structural-kata/opcode probes, their separate fitting code and model families had no production consumers. Their drivers and dedicated tests are retired too. Current semantic projection, policy catalog and production encoders remain. |
| `verify/run_exp11_judge.py` | Historical launcher whose three experiment-specific dependencies were already deleted. |

`.lf/logs/`, `.lf/metrics/`, `.lf/scratch-stash/`, `.lf/tmp/` and native `.dSYM`
files are no longer tracked. Existing local bytes were preserved. Ignore rules
keep new runtime output out of source control.

## Fetch historical artifacts

The archive retains exact original bytes, with digests and sizes in
[`experiments/data.manifest.json`](../experiments/data.manifest.json):

```bash
uv run --extra artifacts python -m manabot.data fetch experiments/data/archive
# Or select one corpus:
uv run --extra artifacts python -m manabot.data fetch experiments/data/archive/conformance/semantic-kernel-w12-v1
```

An archived file's path is `experiments/data/archive/<original repository path>`.
The migration contains 202 files / 35,010,322 bytes: old semantic-kernel corpora,
superseded authored-match receipts, older Learn cohorts, Jeong replays, Elemental
w10 evidence, and structural-kata/opcode inputs and contracts. Uploads were
followed by a complete restore using an empty cache: 146 distinct objects /
28,201,845 bytes were downloaded and verified against their original hashes.
No receipt, world, result or registration was re-serialized.

Archived receipts retain their original paths and source identities inside the
JSON. To reproduce an old experiment, use its recorded source/world and restore
those files at their original paths in that checkout. Fetching archived bytes
does not admit them under the current runtime. Historical Markdown reports and
protocols stay available in the repository; their original commands describe
those historical executions.

## Old-world corpora archived by Task afbd6a (2026-10-30)

Old worlds run from the git revision that recorded them, so corpora that only
the code of those revisions reads moved to the archive, byte-for-byte, with a
complete empty-cache restore verified against the originals:
`authored-match-v1`, `authored-match-w11-v1`, `authored-match-parity-v1`,
`authored-match-parity-w11-v1`, `public-commitment-parity-v1`,
`semantic-kernel-w12-v1` through `semantic-kernel-w15-v1`, and the frozen Learn
cohort `learn-lesson-w4-v3` with its wrapper `scripts/verify-learn-lesson`
(archived as `experiments/data/archive/scripts/verify-learn-lesson`). The
wrapper verified only that w4 registration, which the current world rejects.
The same Task deleted the w1–w11 card sources (`content/semantic/v1`,
`w7`–`w11`) and the Python card compiler; recover them from git history.

## What remains in source, and why

- The four w17 corpora in [`conformance/`](../conformance/README.md) are read by
  Rust conformance tests, CI and Academy parity checks.
- `academy.learn_lesson_evidence` remains: Academy tests use its attempt
  verifier, and its `register`/`run`/`verify` modes record a new Learn cohort.
- `.lf/config.yaml`, current `.lf/wave/` state and referenced `.lf/chapters/`
  records remain. The chapter records are linked by the Rules Wave; they are
  accepted planning evidence, not process scratch. Current rules-specific and
  Academy flows remain discoverable. The optional-payments Task still has an
  unsettled recorded Flow, so its authored flow is retained.
- `semantic/learning.py` projects rules programs into model inputs; despite its
  name it is used by current model/world/catalog code. It is not the retired
  structural-probe trainer.
- `remote/_legacy.py` still has active CLI callers. Retiring it requires
  removing a supported consumer, not just deleting an orphan import.
- Browser visual references, protocol fixtures, model fixtures and the rules
  audit index have current consumers. They are not moved merely because they
  are large. The retained Comprehensive Rules PDF is a reference document, not
  experiment output.

The mixed `arena/` package was subsequently removed on 2026-10-23, completing
the naming direction deferred from MTG-179. Reusable game execution, replay,
registration and policies live in `sim/`; protocols, admission, statistics and
comparisons live in `eval/`; checkpoint staging and followers live in
`experiment/`. Current imports and CLIs use those owners without compatibility
shims. Frozen receipts keep their arena-named fields and original source paths.

A dead-code sweep on 2026-10-23 removed modules whose only consumers were their
own tests (the first-light renderer, the MTG-174 local evaluator chain, the
launchd local runner, arena competency/profile helpers, remote export progress,
pooled data generation, Study evidence projection, the selected-BranchDriver
teacher) together with unreferenced functions they left behind. Frozen
protocols, receipts and benchmark notes that name those paths are unchanged;
recover the code from git history.
