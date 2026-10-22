# MTG-174 shared mana repair

Accepted direction: shared payment authority with visible choices and exact
continuations. Scope: bounded repair; no paid/scientific execution or publication.

Diagnosis, matrix, hashes and acceptance (423 debug Rust, 14 replays, 51 Python):
`863938740:scratch/elemental-mana-repair.md`. Professor and Prism/Treasure roots
do not establish the unretained GPU failure's scalar cause. Actual outcome:
`experiments/elemental-uniform-baseline.md`.

## Accepted implementation at 863938740

`PaymentModel` in `flow/mana.rs` owns offers and executable witnesses: exact
sources, pool remainders/restrictions, spend purpose, life, waterbend, and
per-use tap/sacrifice reservations. `Decision::ManaUpfront` carries payer,
source, ability, recursive parent and its effect frame. Upfront payment precedes
source costs and visible output. Mana abilities create no stack object.
Fixed production preserves source order; only nested automatic prefixes minimize
commitments while proving the complete parent continuation.

Delete — do not maintain: grouped boolean `can_cover_cost`, separate fixed-only
planner, greedy production loop, recursive waterbend feasibility, whole-object
`reserved_payment_source` / `pending_mana_source` / `produce_mana_without`, and
obsolete duplicate frontend/demo authorities. Preserve the current upstream
Academy surface and one native payment authority.

## Reconciled delivery boundary — 2026-10-22

Jack Heart's accepted shared-payment direction remains unchanged. Merge
`f7316d9e0` includes main `0d9814995` and integration restoration `8f91f2dcb`:
Academy/Study/demo/CI consumers, bundled native package and current experiment
paths now agree. Reconnect retains six lands for both outcomes after the filter
fee. Current uniform sampling supports both-seat learning; historical schema-1
one-seat recipes and frozen scientific evidence retain their original meaning.

The merge succeeded; only stash restoration failed on `scratch/questions.md`.
The caller restored the tracked patch and all 18 local files, preserved both
questions versions and retained stash `e4760d5`. No unresolved paths remain.

Existing evidence: `8f91f2dcb` passed 187 Python and 33 debug Temur tests
(`.runs/mtg174-integration-repair/`). Compression passed 33 debug Temur and the
reconnect test with a rebuilt, atomically installed extension. The caller's sync
checks passed 10 collector/both-seat/recovery/checkpoint tests. These results
support bounded integration, not scientific launch readiness.

Reduction: validate mana colors before detaching the decision, avoiding game/frame
clones. The automatic budget ends at its first Spend; chosen activations are
excluded until then, so no nesting depth is needed. Activation admission reuses
the exact-witness predicate without rechecking its constructed first step.
Reconnect label ordering is explicit; no behavior or frozen evidence changes.

Released/pinned SDK 0.2.0 has `etude.eval.query` but lacks `etude.reports` and
the old `etude.train.report.pages/write` API. The isolated owner test now gates
on those actual capabilities before importing the owner. Standard collection
skips that suite explicitly; the retained exact source SDK runs its six tests.
No dependency bump, report reimplementation or released-0.2 owner support is
implied. Original report receipts and all failed scientific evidence are intact.

Remaining for gate/CI: current-world arena/monitor admission and full learnability
including PPO; concept review is recorded below. The historical report runtime remains isolated; production
owner/report compatibility needs separate validation before any future use.
MTG-174 still lacks an updated endpoint and scientific evaluations. The original
run failed after four updates; its inclusive $0.051831583766142525 estimate and
unknown scalar GPU failure prefix remain limits. Further paid execution needs
a new explicit decision plus fresh source/runtime, protocol and budget admission.
No such decision, owner restart, publication or merge is part of this repair.

Check: `uv run --extra dev pytest tests/training/test_elemental_live.py tests/training/test_etude_sdk.py -q -rs` — 4 passed, 1 explicit historical-SDK skip; `uv run --no-project --python .runs/mtg174-report-runtime/bin/python python -m pytest --noconftest tests/training/test_elemental_live.py -q -rs` — 6 passed; broader gate/CI deferred to delivery.

## Concept review of the restored merged model — 2026-10-22

Outcome: **nothing to change; no open product choice in this repair**. Review of
`f7316d9e0` against main `0d9814995`, including the restored compression, leaves
Jack Heart's accepted payment experience and model unchanged. Academy has no
remaining diff against that main: Command binding, navigation, viewer-safe
projection, Study and reconnect stay with their existing owners. The payment
usage contract in `docs/rules/optional-payments.md` already describes the result.

| Acting agent's action | Shared state and transition |
|---|---|
| Activate Prism with Treasure and an empty pool | `activate_mana_ability` opens `ManaUpfront`; reserve Prism's incompatible uses before paying {1}. |
| Activate Treasure and choose its color | Commit Treasure's costs, expose `ManaChoice`; `advance_mana_upfront` spends its one mana and commits Prism's costs. |
| Choose Prism's output | Restore the original payment/frame or priority; no mana-ability stack object or automatic color selection. |
| Reconnect after a lost response | Recover the actual pending color; duplicate Command changes neither state nor events. Decline remains explicit where legal, retaining activation costs/output; a waterbend contribution closes decline. |

The compression validates colors while the continuation is attached and ends
the automatic source budget at the first Spend. Neither changes the contract.
Typed pool arithmetic owns restrictions/remainders; the payment model owns
feasible transitions; Game owns events and continuation execution. No second
boolean planner or browser payment authority survives. Source reservations allow
tap-then-sacrifice while prohibiting an owed tap or selected Treasure sacrifice
from being spent twice.

Representative adversarial evidence is substantive, not just solver agreement:
the bounded Prism/Treasure trees enumerate every offered branch, replay complete
prefixes and roll back exact witnesses, independently asserting Treasure +1,
Prism -1/+1, Goblin -1 and stack counts. Nearby tests reject seedless filters,
restricted-mana laundering and double-use, preserve pooled required colors, and
distinguish waterbend from mana-tap bonuses. This is not exhaustive proof of the
historical 15-family matrix. Separate repeatable or tap/sacrifice mana uses still
require extending the finite-source model; its registry test and README make
that supported-content limit explicit. No large-board performance claim follows.

Failure/recovery preserves committed costs, parent targets/frame, source identity
and revealed information. Engine fork/undo is search machinery, not a newly
approved human cancellation action. Invalid colors are rejected before detaching
the continuation; Academy recovery does not reconstruct payment from card text.

Next proof belongs to the caller's already-running post-sync checks against the
rebuilt canonical `manabot.managym` native module: focused Python, arena/monitor
and the single full learnability run including PPO under its 480-second watchdog.
This review neither reran them nor claims their outcome. SDK 0.2 capability skips
and six historical-runtime owner passes do not establish current owner launch
readiness. Preserve the failed scientific evidence and all execution limits
above; publication judgment remains with the caller, with no merge arming,
Task completion, owner restart or new scientific authorization from this review.

## Final post-sync delivery check — 2026-10-22

Inspected the completed checks against the restored compressed source after
main sync f7316d9e0. Debug Temur target: 33 passed in 1.28s
(`.runs/mana-pursue-final-native.log`). Focused Python, current-world arena,
mirror admission, monitoring, collector/recovery and decision-surface checks:
73 passed, one explicit historical-owner capability skip in 65.75s
(`.runs/mana-pursue-final-python.log`). These supplement the earlier broader
423-native and 187-Python evidence; those older counts are not relabeled as
post-sync checks. Formatting, ruff and diff checks pass.

Full `uv run python -m manabot.verify.learnability --out
.runs/learnability/mtg174-mana-pursue-final` passed in 40.9s, including all
three capacity seeds and the production PPO control (65,536 transitions,
1,541 diagnostic games, checkpoint reload). This is credential-free local
learnability evidence, not a ten-deck strength result or the paid baseline.
The imported canonical native module was
`manabot/managym/_managym.cpython-312-darwin.so`, rebuilt and atomically
installed from the pinned-root wheel; SHA256
`0aeccce9deefc0dfbe806c8b18b13f9984332cc9ed603c9063138d4d3f54fa5d`.

Concept review found no remaining product mismatch in the accepted repair.
Publication is the next delivery boundary. No merge arming, Task completion,
owner restart, paid allocation or scientific-run retry is included.
