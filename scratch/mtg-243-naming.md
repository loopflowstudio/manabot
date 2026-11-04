# MTG-243 — complete catalog naming

## Accepted direction — 2026-11-04

Jack Heart authorized complete, learnable catalog-wide naming beyond 512 names
as one reviewable PR, with no merge or training experiments. After publication
of `9db184b9aa3df9286b07abdafa2d41db0d93e651` in PR #424, Jack Heart requested:
“Let's make sure to both measure those risks and attempt to ameliorate them.”
That accepted pass covers legal-row scoring, compact pending/rollout storage
through both learners and incidents, bounded same-host costs, paired public-state
competency, actual executable-program attribution, and the five CI compatibility
failures. The code Graph is at implementation; publication and interactive review
remain later steps. No auto-merge, paid execution or strength experiment is authorized.

The original implementation and publication evidence are retained at
`9db184b9:scratch/mtg-243-naming.md`; the uncommitted research/accepted-pass text
was preserved at `.lf/logs/naming-risk/plan-before-reduction.md` before curation.
This pass follows the accepted Task directive; no Wave files were edited.
Earlier T7 direction is historical at
`4463e57f5:scratch/questions.md`.

## Design and implemented ownership

`ContentPack::name_candidates` owns immutable sorted public names and a digest.
`Decision::NameCard` shares that vocabulary; only the accepted choice constructs
one `ExileNamed` and resumes the existing suspended frame. Commands retain the
same modal order and labels. `SearchNamed` retains private inspection, remaining
selections and fail-to-find semantics. Search, replay, forks, LLMs and Academy
all retain the complete atomic choice; filtering never commits a choice.

Explicit input v8 carries `name_candidates[A,2]`: a categorical public name and
optional one-based definition join. A is max(configured actions, vocabulary).
Names never consume card slots. Fresh defaults remain v6; v1–v7 shapes and
omitted checkpoint defaults remain frozen. Default overflow rejects; explicit
first-legal may truncate legacy inputs and is not complete naming support.
V8 still reserves the installed naming vocabulary under that policy.

The action networks now gather only legal rows, share object/program computation,
and scatter logits to original coordinates, including holes and reorderings.
Flat and compound policies use one action-embedding owner. Continuation checks
select join columns before gathering rows, avoiding repeated full payload copies.
Prepared inference retains its policy-scoped program cache.

`PackedActions` owns offsets, original slots, complete legal action fields and
full behavior probabilities/log probabilities. `_Pending` retains those rows;
`RolloutBatch` concatenates them in time/trajectory order and keeps object fields
on CPU. Both learners validate all behavior in bounded CPU chunks and materialize
only selected minibatches. Returns, references, selected indices and numerical
invariants remain unchanged. Numerical incidents (schema 2) keep compact offsets,
slots, capacity and distributions plus failing minibatch evidence. Recovery
snapshots still occur only with empty pending/transition storage.

The equal-rule fixture reproduced identical logits for two differently named
public decklists before identity reached the object encoder. V8 now adds the
same name vectors to public definition references before card/known-hand/decklist,
continuation and history pooling. Raw program embeddings remain the candidate
scorer's semantic input, avoiding a duplicate name addition. No hidden membership
is inferred; native closed decklists remain zero. Definition-free names add no
object identities. The finalized unreleased v8 checkpoint binding includes
`naming_representation=public-identity-v1`; earlier v8 previews and changed
names/order/definition joins fail before weights load. No warm-start migration.

Schema 8 / capture 5 / state hash 16 remain the original Task transition.
Historical capture 3/4 explicitly select schema 7 for read-only validation and
preserve their original identity/digest. Schema 7 rejects new name references;
live readers still require schema 8. Retained schema-7 replay fixtures now assert
exact current scalar/vector frames and unchanged offers/decision fields, with
only the declared schema/fingerprint transition excluded. Frozen files are intact.

## Delete — do not maintain

Completed: owned per-name effect branches and per-resolution sorting; batch-wide
padded action networks; dense retained action/distribution padding; whole-rollout
learning device expansion; stale tests demanding schema-7 bytes from schema 8.
Removed `RolloutBatch.from_dense`: its two test fixtures now use the production
`PackedActions.pack` entry point directly. Boolean/integer indexing already owns
its NumPy output, so packing and pending object slices need no second copy.
Inspection tests reuse one materialization or read retained object fields directly;
the history fixture materializes only its selected row. No further deletion target
remains in this design.

The only dense scoring reference is `tests/model/naming_reference.py`, frozen
from the published action computation for numerical checks. No production fallback
or alternate learner was added. The scratch packing prototype is retained only
in `.lf/logs/naming-risk/prototype-probe.py`.

## Evidence and limitations

Directed checks cover mixed naming/search/ordinary observations, sparse action
coordinates, attention/FiLM on and off, values and parameter gradients, prepared
inference, malformed joins, full compound naming support, exact storage/copy
ownership, both loss gradients, bounded minibatch materialization, numerical
incidents and native collector recovery. CI now selects the packed-action and
hand-calculated credit tests explicitly. The original 928-name legality and
policy-reachability fixtures remain; vector and Academy coverage use the installed
catalog. There is still no single 928-name vector/Academy end-to-end proof: the
large vocabulary is injected at the native test/model-fixture boundaries, not
admitted as a production world.

The equal-rule competency check changes only two installed encoder token rows,
not native definitions or deck admission. Without the public identity join the
paired logits are identical. With it, a two-example least-squares fit of the
existing final linear head yields opposite preferences; gradients reach names
and program tokens, reordering preserves preferences, and closing the differing
decklists makes logits identical. A preceding 16-step Adam probe did not converge.
This demonstrates representability, not SGD convergence, useful exploration,
transfer, sample efficiency or game strength. The action-type-uniform reference
still offers no relevance prior within a single naming action type.

### Bounded cost observations

`uv run --extra dev python scratch/naming-risk-probe.py` compares the published
Python model against this implementation on the same four real observations,
one CPU thread, width 16, attention + FiLM, one warmup and three measured passes.
No optimizer, stress workload, oversized allocation or training experiment.
Results are in `scratch/naming-risk-results.json`. Native paths affected only by
the historical reader change are reported as current attribution, not a speedup.

Legal supports are [344, 21, 4, 4]: action networks process 373 rows rather than
1,376 (2,048 with interior padding). Action/distribution storage is 237,568 →
43,308 bytes (81.8% less); including unchanged object fields, 620,480 → 426,220
bytes (31.3% less). Live observation buffers still occupy 604,096 bytes for these
four rows. Dense output logits and minibatch axes still scale with vocabulary.

Elapsed milliseconds, median [min, max] of three observations:

| Phase | Published 9db184b9 | Retained |
| --- | ---: | ---: |
| Full forward | 96.754 [24.060, 133.317] | 28.311 [20.172, 41.945] |
| Backward only | 793.138 [690.672, 886.133] | 554.426 [417.337, 656.149] |
| Forward + backward | 843.185 [539.592, 983.388] | 810.091 [771.600, 887.931] |
| Action networks | 0.796 [0.639, 1.210] | 0.389 [0.370, 0.390] |

These timings are variable and establish no end-to-end throughput improvement.
Before the relation-copy refinement, the first pass observed forward 23.368 →
21.835 ms and forward+backward 305.510 → 307.333 ms; action networks were then
0.561 → 0.661 ms. Preserve that contrary evidence rather than treating the later
samples as an established speedup. Full program encoding/backprop remains the
principal measured cost. Model setup medians are 34.081 → 31.682 ms after warmup.

The installed catalog has 356 executable definitions, 41,892 expanded tokens
and a 713-token maximum program. Ordinary/naming subsets request 4 / 344
programs (190 / 41,159 tokens). The final program-encoding medians are
0.999 / 39.977 ms before and 1.079 / 70.980 ms after, with wide sample variation
and unchanged program computation. The 928-name fixture adds mostly names
without programs; it is not a 928-program measurement. CPU is the configured
probe target; CUDA is unavailable and available MPS was not exercised. GPU
correctness, peak memory and intended-size learner throughput remain unmeasured.
Current native projection, encoding, fork and decision-projection samples, and
packing/materialization times, are recorded separately in the JSON.

Published CI run `41914195610` passed Engine Budget with unchanged digests:
search 116.194 → 121.779 s (+4.8%), worst reply 3.417 → 3.621 s (+6.0%),
steps 18,749.7 → 17,511.8/s (−6.6%). Its reference is `ea316ff14`, not this PR's
immediate base `8e8d702f8`; do not attribute its whole delta to naming. The new
head still requires CI. Prior native allocation attribution remains at the
published design reference above; no naming-engine algorithm changed this pass.

## Remaining Graph work

No product decision blocks the installed-catalog scope. Gate owns affected
model/training/recovery suites, package boundaries, current-world admission,
full debug Rust/Clippy, Academy/browser coverage and any current-source receipt
refresh. Frozen evidence must remain untouched. Release extension rebuilt and
atomically installed; the focused native decoder test passes with debug guards.
All five published CI failure cases pass locally in focused checks.

Check: `uv run --extra dev pytest tests/training/test_packed_action_rollouts.py tests/training/test_rl_objectives.py tests/training/test_regimes.py::test_named_treatments_execute_real_optimizer_batches tests/training/test_collection.py tests/training/test_two_seat_learning.py tests/training/test_action_cutoff.py tests/training/test_credit_assignment.py tests/training/test_game_errors.py tests/training/test_numerical_health.py::test_extreme_real_collector_to_learner tests/training/test_numerical_health.py::test_failed_update_retains_state_and_excludes_prior_overhead tests/model/test_recent_events.py -q` — 110 passed, 1 skipped; Ruff and diff checks passed. Full affected suites and acceptance checks remain with gate.

Publication should update PR #424 with compact storage/scoring, the reproduced
identity gap and its fix, these measured costs and limitations, then stop at
interactive review. `scratch/naming-risks.html` holds the current risk assessment;
`pr-review.html` retains the prior source walkthrough with an update notice.
No merge, auto-merge or experiment occurred. Local HTML/screenshots remain review
artifacts rather than product source; updated pages have no new rendering proof.

MTG-241 remains a separate continuation owner. Its reported FrameCompletion,
top ordering code 27 and SelectCards code 28 do not collide with naming's modal
obligation 10. That code is not synced here; compatibility awaits ordinary sync.
Production admission of extra validated names and default adoption of v8 remain
separate future decisions. The current universe includes real back-face names
but does not claim all legal Magic names or change deck admission.
