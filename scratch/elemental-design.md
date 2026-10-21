# One faithful ten-deck Elemental world — MTG-164

Accepted direction: Jack Heart, 2026-10-20, in MTG-157. Reconciled 2026-10-21.
The exact accepted [lists and Lesson pools](../docs/rules/elemental/decks.md),
[machine-readable selection](../docs/rules/elemental/decks.json) and
[cube snapshot](../docs/rules/elemental/cube-snapshot.csv) are durable inputs.
The [implementation contract](../docs/rules/elemental/world.md) maps cards to
rules owners and describes admission and bounded evidence.

## Accepted outcome and constraints

All ten 40-card lists and their exact Lesson pools form one new versioned
world usable through Academy and ordinary manabot training. The selected pool
contains 89 nonbasic names. Cube copy limits apply independently to each main
deck plus its Lesson pool; reuse between decks is allowed and basics are
unlimited. UG remains Temur Lessons with Iroh, Grand Lotus and Ran and Shaw;
BG remains landfall plus sacrifice. No balance certification is implied.

Complete printed rules execute through the authoritative native and compiled
paths. Humans and agents have complete legal choices, correct visibility,
revision-bound Commands, deterministic replay and exact isolated forks.
Per-matchup rules, proxies, reduced mechanics and silent truncation are outside
the accepted approach. Earlier packs, schemas and frozen evidence retain their
bytes and identities; old checkpoints do not become compatible merely because
some shapes match. The content expansion supersedes the older chapter's deck
boundary for this Task only. Wave and chapter files remain read-only.

Jack Heart's 2026-10-21 interface decision preserves the same viewer-safe data,
legal offers, action granularity and Commands for humans and agents. Labels
and grouping may clarify that contract. Human-only auto-payment, hidden action
selection, extra knowledge and newly introduced cancellation/undo are not
accepted. The separate Academy redesign is outside this Task.

Jack Heart authorized autonomous implementation, validation, publication and
merge on 2026-10-20. No additional demo, PR-review or merge-click decision is
pending. Paid training, cloud rentals, strength campaigns, general curricula,
deck-agnostic heuristics and broader content expansion are excluded.

## What changed our understanding

The predecessor deliveries are present at 1c92d12e (combat), fc0d4439
(sacrifice/landfall) and a04dcdc2 (Temur). MTG-163's completed merged delivery
was verified before implementation; dependency waiting is no longer remaining
product work.

Composing their packs alone missed Gran-Gran, Combustion Technique and Enter
the Avatar State. The first two needed compiled admission; Enter also needed
its omitted temporary Avatar subtype. Those gaps are closed through typed
operations, incarnation-local state, cleanup/rollback/hash coverage and shared
observations. Existing two-setup checkpoint binding also needed an explicit
ten-setup roster, including mirrors; widening the roster does not establish
learning on untrained matchups.

The 2026-10-21 implementation and compress results describe the recovered
source, not main alone. A checkout update preserved that source and its new
artifacts in stash `115f3b0f93fc26e31a6d9e5c7b1f8e7053cb058a`; it was restored
without conflicts during reconciliation. The backup remains retained. The
restored w10 parity receipt's complete source manifest equals current source.

## Implemented behavior and retained evidence

Development world w10 admits `elemental-world-v1`: 100 definitions comprising
89 selected nonbasics, five basics and six referenced tokens, with ten exact
compiled setups. Academy selectors, local asset identities, recorded roots,
ordinary policy/PPO inputs, checkpoint save/catalog/load and arena admission
carry that explicit identity. Foreign or changed setups, missing pack identity
and older checkpoint worlds are rejected. Frozen artifacts remain retained.

The immutable native catalog is parsed once. Academy reuses validated compiled
setups, checkpoint execution shares roster checks, and frozen learning schemas
retain explicit opcode limits. Schema 6 includes the new operations and Avatar
feature; earlier schemas retain their own vocabularies.

Retained execution evidence from implementation and compress includes:

- Debug native full-definition coherence and all 100 ordered matchup cells,
  including mirrors, completed within the declared 30,000-action caps.
- The prospectively declared Command cohort completed 14 games and 13,126
  Commands, with 268 isolated fork/stale checks. Both roots and viewer
  projections agreed. The largest observed offer set was 20; universal
  fixed-capacity sufficiency is unproven.
- The w10 semantic corpus records 14 cases and 3,491 Commands. Academy parity
  records 156 Commands and 942 viewer checks. These executors share primitives;
  this is not independent-engine parity. The Phase table is source evidence.
- Focused runtime, actual CPU PPO/update/reload, Study, navigation and frontend
  checks passed. The final compress run passed 95 Python checks, including
  current-world arena admission and checkpoint monitoring, plus four debug
  Rust checks and the release extension build. These results are reused.

Delete — do not maintain: no remaining targets. The incomplete Enter rules
and obsolete subtype-irrelevance comment were replaced. Earlier versioned
packs and receipts remain contracts.

## Remaining acceptance

### Concept review — 2026-10-21

Outcome: no design change and no open product choice. This review follows
Jack Heart's accepted ten-deck world and shared human/agent interface decisions;
it grants no additional scope. The existing usage contract in
`docs/rules/elemental/world.md` already expresses this experience and needs no
proposed replacement or new skill instructions.

| User action | Concept and state owner | API consequence |
| --- | --- | --- |
| Choose UG against BG, reverse seats, or choose a mirror | Two exact setups within one Elemental world; native compiled content owns lists and Lesson pools | Academy game configuration and `MatchHypers.authored` resolve `authored_deck_setup`; both seats carry the same explicit pack |
| Make a legal choice, including Learn or payment | Viewer-safe offer and revision-bound Command; managym owns the transition | Academy submits the offered Command; agents use the same authority and action granularity |
| Save and reload a policy, then select another admitted pair | Checkpoint binds world, complete setup roster, programs and input schema; collection still uses its actual pair | Save/catalog/load validate the binding; execution checks the selected pair against it |
| Rebuild a recorded root or inspect an isolated fork | Recorded setup/asset identities and retained root, rather than a second rules authority | `GameConfig.to_rust` validates exact lists, pools and installed pack; replay/fork retains the authoritative state |

Failure and recovery preserve those same concepts. An Elemental/outside pairing
is rejected; select two roster members and retry. A stale Command must not
advance the match; recover the current viewer frame and choose a current offer.
A modified Lesson pool, missing pack, changed asset manifest or old checkpoint
must fail admission, rather than reconstruct a different game. Retain the
recorded evidence and use its matching versioned content/runtime, or start a
separate current-world game. This does not promise migration of old weights or
new resume, undo, cancellation or artifact-restoration machinery.

The consequential distinction is compatibility versus training coverage: one
checkpoint may execute every admitted pair without having learned each pair.
The ten-setup roster therefore cannot replace actual matchup attribution in
training records. Presentation manifests remain validated views of native
setups; their default pair does not restrict world admission. Avatar grants
remain incarnation-local visible state, never a change to printed identity.

Counterexamples that would invalidate this model are a mirror losing its Lesson
pool, reload narrowing a checkpoint to its representative pair, a replay
silently substituting installed content, or a human continuation hiding an
agent-visible choice. Existing bounded evidence supports the model but does
not establish universal fixed-capacity sufficiency. The next proof remains the
acceptance gates below, including release interaction/recovery checks; this
read-only concept review adds no runtime certification or implementation work.

Implementation is present; delivery acceptance remains incomplete. Gate owns
the remaining affected suites, full prescribed learnability diagnostic
(including reward-driven PPO), release browser interactions and pinned Linux
visual comparison. Existing arena/monitor results remain usable unless source
changes invalidate them. Headless reconciliation supplies no new human-play,
visual, balance, strength or runtime-budget certification.

The inherited checkpoint-advice, belief-recommendation, public-commitment and
RUL-9 frozen-provenance gaps described in the predecessor contracts are not
closed by the new Learn/navigation fixtures. Gate must retain concrete failures
and their delivery impact; historical receipts cannot be rewritten as current
proof. New source changes can invalidate the source-bound Academy parity
receipt even when its Command tape is unchanged.

Publication, waited landing, verified merge and Task completion remain later
caller-owned delivery steps. No new product decision is needed for this scope.

Check: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run --extra dev --extra notebook --extra artifacts --extra reports pytest tests/semantic/test_elemental_world.py -q -k 'exact_accepted_lists or full_roster_checkpoint or avatar_grant'` — 3 passed, 16 deselected; restored parity source-manifest equality and `git diff --check` also passed.

## Publication inspection — 2026-10-21

The actual Task base is f15196e1; publication selects only implementation and
versioned evidence paths. Untracked `.lf/` contributions and all scratch notes
remain local. The authored Flow publishes before gate; publication does not
claim merge readiness. `git diff --check`, Rust formatting, joint collection of
the new Elemental module and release prompt contract (20 tests), frontend
`svelte-check` (0 errors/warnings), and curated-pack Vitest (5 passed) passed.
Frontend checks use the already installed Node 24.12.0 because Homebrew Node
25 cannot load libllhttp.9.3.dylib. Focused Python admission/navigation/Learn/
arena/monitor, full debug Rust and full learnability checks are in progress;
results will be retained here before this publication step ends. Release
browser interactions and pinned Linux visual comparison remain with gate, as
do the previously documented inherited frozen-evidence limitations.

Publication results: PR https://github.com/loopflowstudio/manabot/pull/337 is
ready, head 316b9f76, base main; auto-merge is off and no Task settlement was
requested. Its 92 paths contain no scratch or `.lf/` contributions. Focused
Python command above passed 65 tests in 464.07s. Full learnability passed in
428.7s, including all three capacity seeds and production reward PPO (65,536
transitions; preference score 0.59375 → 1.0 versus no-update 0.640625). Retained
local diagnostic: `.runs/learnability/elemental-164-publish/report.md`; its source
HEAD is the pre-commit f15196e1 with the tested implementation tree digest.
Full debug Rust continues in `/tmp/manabot-164-publish-rust.log` (session 72281);
no failure observed yet. Python and frontend logs are
`/tmp/manabot-164-publish-python.log` and
`/tmp/manabot-164-publish-frontend.log`.

The refreshed PR copy records the 65 Python and full PPO diagnostic passes,
frontend checks, and explicit remaining full Rust/browser/Linux visual gates.
Read back and verified ready head 316b9f76 with auto-merge off. Full Rust
continues at `/tmp/manabot-164-publish-rust.log`; gate should consume that
existing run rather than duplicate it. Publication does not mark it passed.

## Gate — 2026-10-21

Reviewed the 92-path implementation against f15196e1, including runtime,
checkpoint roster admission, projections, schema/opcode retention, generated
closure, selector/treatment changes and docs. No runtime edit was warranted.
Added `frontend/e2e/elemental-world.spec.ts` to the release suite: three real UI
cases cover UG/BG, reversed seats and a mirror, all ten selector entries,
Lesson pools, opposing privacy, pack identity and a submitted Command.
An initial test draft tried changing selectors during an active game; selectors
intentionally disappear then. That attempt was stopped and replaced by three
fresh-context cases; it is not counted as a product failure or passing check.

Command/result ledger (Node commands use installed Node 24.12.0):

- `uv run --extra dev ruff check $(git diff --name-only f15196e1 -- '*.py')` — pass.
- `uv run --extra dev ruff format --check $(git diff --name-only f15196e1 -- '*.py')` — pass, 24 files.
- `cargo fmt --manifest-path managym/Cargo.toml --check` — pass.
- `cargo clippy --manifest-path managym/Cargo.toml --all-targets --all-features -- -D warnings` — pass.
- `npm --prefix frontend run check` — pass, 0 errors/warnings.
- In frontend, `npm run test:e2e:release -- --grep-invert 'release stack proves versioned visuals'` — production build and 13 browser checks pass, 1.7 minutes.
- In frontend, `ACADEMY_RELEASE_API_PORT=8013 ACADEMY_RELEASE_FRONTEND_PORT=5185 npx playwright test --config playwright.release.config.ts elemental-world.spec.ts` — 3 pass, 8.4 seconds; real production preview/backend, no mocks.
- `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run --extra dev --extra notebook --extra artifacts --extra reports pytest tests/semantic/test_compiler.py tests/semantic/test_learning_projection.py tests/model/test_attack_choice.py tests/verify/test_observation_parity.py tests/academy/test_advice.py tests/academy/test_belief_recommendation_flip.py tests/academy/test_public_commitment_parity.py tests/experiments/test_rul9_played_workloads.py -q` — 77 pass, 4 fail, 118.84 seconds. Log `/tmp/manabot-164-gate-python.log`.
- `git -c core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol diff --check f15196e1` — pass; the retained cube CSV uses CRLF. Preserve its evidence bytes. Plain branch diff check flags those CRLF endings; working diff check passes.
- Reused publication's 65 focused Python checks and full learnability report at `.runs/learnability/elemental-164-publish/report.md`; no runtime source changed during gate.
- The inherited local full debug run log ends without an overall result and its process is no longer live. It is not counted as a pass. CI `cargo test` at 316b9f76 passed (2m14s), with debug assertions enabled, resolving this requirement without replaying the full local matrix.
- CI at 316b9f76 also passed Linux main/long integration, Academy/Linux, protocol, semantic coverage/conformance and pinned Ubuntu visual gate (6 Python, 14 browser checks). macOS jobs remain queued; their results are deferred to capable CI.
- CI Clean-machine Play failed: `frontend/e2e/clean-machine.spec.ts:286` calls `chooseDeterministicAction` after reload, but `data-update-seq` stays 1 for 30 seconds. Initial play and visible reload equality passed; no proof receipt. Cause unresolved. Log `/tmp/manabot-164-gate-clean-machine-ci.log`.

Python failures are concrete inherited limits: belief flip visit margins [14,48]
versus frozen [66,38] (top actions still [1,0]); two public-commitment checks
raise KeyError 'status' at live replay line 376; RUL-9 raises 'landed parity
receipt changed'. Preserve their frozen inputs. The initial attempted
collection used a nonexistent checkpoint-advice filename and collected nothing;
it was corrected to the real test modules above, not counted as a check.

Durable delivery evidence and limitations now live in
`docs/rules/elemental/world.md`. No visual baselines, frozen receipts, Task/Wave
state, unrelated `.lf/` contributions or publication settings were changed.
Gate does not certify merge readiness: clean-machine post-reload execution and
four inherited proof failures remain blockers. Test/config/doc edits are left
for the caller's next delivery step; no commit, push or merge performed here.
