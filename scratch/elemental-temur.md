# Temur implementation

Reconciled 2026-10-21. Status: implemented locally; release acceptance and delivery
remain unfinished. Jack Heart's accepted direction is dated 2026-10-20.

Accepted outcome, constraints and provenance: [elemental-design.md](elemental-design.md).
The duplicate accepted brief previously here is retained in that complete source.

MTG-162 is completed and PR #331 merged at `fc0d4439ca40b7a1e8f9c3e59e533efe592c45b6`,
which remains HEAD and the retained `origin/main`. No later upstream changes
are present in that reference. MTG-163 has no current PR or completed delivery.

Rules' objective remains one faithful authority for human play and training.
The accepted Task explicitly extends the older Allies/Lessons content boundary
for these six cards without authorizing Wave/chapter edits.
`.lf/wave/rules/MEMORY.md` is unavailable; the local Rules scope contains only
GOAL.md and no immediate child directories. No Wave files were changed.

## Implemented

All six owned cards have native and compiled definitions, with a nine-definition
Temur validation pack, typed semantic operations and learning schema 5. Runtime
changes cover restricted/granted mana, upfront mana-ability costs, flashback and
its stack-departure replacement, cast provenance, nonlegendary copies, Dragon
buffs, and resolution-time waterbend. World w9 carries player/card widths 40/41
and mutable-state hash version 6. The durable ownership and evidence contract is
`docs/rules/elemental-temur.md`.

Current-world conformance, Academy authority/parity receipts and the Learn demo
use new w9 paths. Historical corpora, receipts, schemas and `erd1` evidence are
unchanged. The fixed authored tape now has 156 Commands; its pinned stale-object
proof uses `96@3`, accepted at revision 15 and rejected after departure at 30.
The reveal test uses a directed Learn choice rather than assuming a random tape
happens to reveal two particular Lessons.

## Implementation lessons

Completed cuts: replaced the free-output-only `can_cover_cost` estimator with a
cost-before-output source search; removed early stack removal from countering
and spell finalization so zone movement owns flashback and cast provenance;
removed the interpretation of waterbend taps as mana activation. Existing fixed
mana production, triggered mana, target legality, immutable definitions, Commands
and undo remain the shared authorities. No alternative card-name rules path or
legacy compatibility adapter was added.

Removed the remaining `waterbend_tap_action` mana-tap interpretation and the
exclusive `rallier_waterbend_feeds_badgermole_cub_mana` interpreter proof;
`waterbend_taps_do_not_trigger_badgermole_cub` now checks the surviving rules.
Waterbend feasibility derives candidates from the probed battlefield instead of
carrying a second count. Chosen-color construction and upfront mana-ability
payment now have shared owners; Herbalist enumerates its 15 unordered pairs
directly, without permutations or deduplication. Fixed-source mana and triggered
bonuses share one calculation. These mechanisms replace the original free-output
estimate and mana-producing waterbend interpretation; neither remains planned.

## Verification and remaining work

Recorded post-correction evidence from the preceding implementation/compression
passes: 11 Temur debug tests, 6 waterbend tests, compiled/native closure, 19
semantic conformance/interpreter tests, and 22 Python Temur/setup/world checks
passed. The release extension was rebuilt and installed; all-target/all-feature
Clippy, Rust formatting, Ruff and whitespace checks passed. These results are
reused here, rather than represented as a new complete gate run.

The w9 corpus and authority/parity receipts were refreshed after removing the
last waterbend bonus path. Previous uncommitted evidence is preserved in
`.runs/compress-temur-before-refresh/`; historical worlds are unchanged. The
corpus reproduces all 12 terminal replays and 3,047 Commands. The Academy tape
has 156 Commands and its live, headless and replay surfaces agree.

Earlier implementation checks also passed 30 arena/monitoring tests and full
learnability (`.runs/learnability/temur-w9-20261021/report.md`) before the final
waterbend correction. That report checks generic input learning and production
PPO; it explicitly does not train the semantic-program cohort or establish
full-game strength.

Remaining acceptance: gate's current-world arena/monitoring and full learnability
reruns, full affected debug Rust/Python suites, configured browser release path
and pinned Linux visual references. Browser/visual certification remains with
gate/CI; this headless reconciliation supplies no visual acceptance. Publication,
waited landing and verified Task completion remain undelivered.
Final ten-deck admission and full-matrix integration remain MTG-164's accepted
scope.

Evidence limit: the inherited `phase-overlap.json` matrix in the w9 corpus is
pinned source inspection, not an executed independent engine. Some local
outcomes still describe older rules (including Learn without a Lesson pool and
fixed-color Suki payments). Those rows are not current w9 defect findings or
proof of agreement. Reassessing them requires updating their local evidence and
classifications; the existing five mismatch labels alone cannot certify w9.
The 12 replay results establish agreement only between the two shared-primitive
executors. No new product decision or substantial implementation gap was
identified in this reconciliation.

Check: `git diff --check` — passed; documentation-only reconciliation, existing runtime results reused.

## Publication assessment — 2026-10-21

Publication is blocked; no PR was created, merge armed, or Task completed.
Fetched `origin/main` and confirmed the actual base remains `fc0d4439`.

The combined arena/monitoring, semantic, Learn setup, parity and release-matrix
contract run returned 172 passed and 17 failed. Failures split into 13 coverage
evidence failures, two stale schema-error assertions, one runtime-policy numeric
comparison and one stale parity receipt. Logs: `/tmp/temur-publish-pytest.log`.

Resolved the two stale schema-error assertions for supported schemas 2–5;
all nine learning-projection tests then passed. Refreshed only the new w9 parity
receipt; its three tests passed, with 156 Commands and no surface divergence.
The initial full debug Rust run found the player snapshot's obsolete 24-byte
bound. It must retain two 12-byte mana pools plus life/flags, so its bound is now
32 bytes; the enclosing UndoEntry limit remains 72 bytes. Rust formatting and
whitespace checks pass. No runtime behavior was changed in this assessment.

Full learnability passed on the current post-waterbend implementation in
`.runs/learnability/temur-publish-20261021/report.md` (56.6 seconds, including
production PPO). All selected arena/monitoring checks passed in the combined
run. The semantic/Academy collection check collected 438 tests without collisions.

Unresolved publication blockers:

- `content/semantic/v1/coverage.evidence.json` still names
  `rules::stage2_cards::badgermole_cub_composes_with_waterbend_payment`, which
  was replaced by `waterbend_taps_do_not_trigger_badgermole_cub`. This breaks
  13 coverage-gate tests. Reconcile current coverage with the corrected rules
  while preserving historical evidence; do not reinstate the incorrect rule.
- `tests/semantic/test_runtime_policy.py:117` fails batched-versus-sequential
  candidate-logit equivalence. It also fails when run alone (1.50 seconds), so
  this is reproducible outside the combined suite. The values print alike;
  their actual numeric difference and acceptable contract need investigation,
  not an unexplained tolerance relaxation. Log: `/tmp/temur-publish-runtime.log`.
- The configured browser release scenario and strict pinned Ubuntu 24.04
  x86-64 / Chromium visual comparison remain unverified. This headless run has
  no rendering environment; local Docker reports Linux aarch64 and alone is
  not evidence for the pinned profile. Existing v6 references describe older
  source, not a certification of this working change.

Per pr-publish, these failures do not meet the publication boundary. The next
authorized delivery step must resolve them and complete the required gate
before invoking `lf pr publish`.

The complete debug Rust rerun subsequently passed (exit 0), including all 289
directed rules tests, checked conformance reproduction, fork/rollback contracts
and doc tests. Ruff check/format passed for the updated projection tests.
Publication logs are retained in `.runs/pr-publish-temur-20261021/`.

## Gate — 2026-10-21

The stale coverage reference and semantic runtime-policy comparison blockers
above are resolved. Current coverage now names the corrected non-mana waterbend
test and its generated artifact is refreshed; historical corpora are untouched.
Measured batched/sequential candidate-logit differences are at most 2.384e-7.
The failing near-zero value differs by 1.043e-7. The test permits four float32
epsilon of absolute roundoff, retains `rtol=1e-5`, and additionally requires
identical decoded Commands. No production scoring or numerical precision changed.

Checks for this tree (logs in `.runs/gate-temur-20261021/`):

- `cargo fmt --manifest-path managym/Cargo.toml --check` — passed.
- `cargo clippy --manifest-path managym/Cargo.toml --all-targets --all-features -- -D warnings` — passed.
- `cargo test --locked --manifest-path managym/Cargo.toml` — passed, 479 tests including 289 directed rules tests, conformance and exact forks/rollback.
- `uv run --extra play maturin build --release -i .venv/bin/python -m managym/Cargo.toml -o managym/target/wheels` — passed; installed the cp312 extension, byte-identical to the extension used for the tests (SHA256 `47b1b197c8e0db582d324fc7354f1d06acc0512cdb9e5b8e195479bfd8d7f7b0`).
- `cargo run --locked --manifest-path managym/Cargo.toml --bin semantic_conformance -- fuzz --root conformance/semantic-kernel-w9-v1 --seed 24301 --cases 32 --max-commands 512 --failure-dir target/semantic-conformance/failures` — passed, 32 cases / 4,630 Commands.
- `uv run scripts/generate_coverage_gaps.py --check` — passed.
- `uv run --extra dev ruff check --fix <changed Python files>` and `uv run --extra dev ruff format <changed Python files>` — passed; runtime-policy test formatting/static checks also pass after annotating its JSON-boundary parameters.
- `uv run python -m manabot.verify.learnability --out .runs/learnability/temur-gate-20261021` — passed in 66.5s, including production PPO (0.59375 before, 1.0 after; no-update 0.640625; 65,536 transitions). Concurrent checks prevent treating wall-time changes as a performance comparison; this is generic-input learning evidence, not semantic-cohort training or playing strength.
- `npm --prefix frontend ci` — installed locked dependencies after the first frontend attempt found missing local tools.
- `npm --prefix frontend run check` — passed, zero errors/warnings.
- `npm --prefix frontend run test:e2e:release -- --grep 'Learn modes|checked Learn demo'` — passed, production build plus both headless Chromium Learn tests in 15.0s, including Back/reconnect and all three outcomes.
- Full configured release browser matrix and strict Ubuntu 24.04 x86-64 visual comparison — deferred to capable CI; local Learn tests do not certify pinned visual references.

Broader Python suite results follow below. Retained navigation positions need
more than digest updates: the original DISCARD tape diverges at Command 5
(`Pass priority` instead of `Do not attack with Badgermole Cub`). An attempted
refresh deliberately stopped at that mismatch and wrote no fixture. Historical
navigation, checkpoint, advice and public-commitment receipts remain unchanged.
The existing positive live-reproduction contracts must be reconciled with w9
without relabeling frozen evidence or weakening assertions. Gate alone does not
publish, land, complete the Task, or select another Flow step.

`uv run --extra dev --extra notebook --extra artifacts --extra reports pytest tests/semantic/ tests/academy/ tests/arena/test_models.py tests/arena/test_mirror_admission.py tests/training/test_monitor_evaluation.py tests/model/test_attack_choice.py tests/model/test_world.py tests/verify/test_observation_parity.py tests/test_package_boundaries.py -q` — 483 passed / 14 failed in 437.04s. All selected semantic, arena/monitoring, model, observation-parity and package-boundary tests passed.

One failure was a stale error-message assertion: the retired likelihood
checkpoint correctly requests an explicit compatible **w9** binding, while the
test expected w4. It now derives the expected world from `WORLD` and continues
to require typed failure plus an unchanged authority digest. Its focused
verification is recorded below.

Thirteen unresolved failures remain release blockers:

- Five navigation fixtures: three changed state digests, plus DISCARD and
  PAY_OR_NOT prefixes that no longer follow the current legal-action stream.
- Five retained-checkpoint advisor tests: current request/authority identities
  differ from frozen evidence, and the retained checkpoint cannot satisfy the
  required w9 binding. Do not migrate its weights or rename its world to green
  these tests.
- One curated belief recommendation test: the intended flip and top actions
  survive, but current visit margins are `[14, 48]`, not retained `[66, 38]`.
- Two public-commitment parity tests: replaying the historical tape through the
  live service returns a message without the expected `status`; the helper
  raises `KeyError` before it can establish provider closure.

These findings need current-world evidence and explicit separation from the
retained experiment contracts. No passing release certification is claimed.

- `uv run --extra dev pytest tests/academy/test_live_advice.py -q` — passed, all 3 tests after the world-name assertion repair (1.46s); Ruff check/format passed for that file.
- `git diff --check` — passed after final documentation edits.
