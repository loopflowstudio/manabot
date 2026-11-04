# MTG-243 — complete catalog naming

## Accepted direction — 2026-11-04

Jack Heart authorized complete, learnable catalog-wide naming beyond 512 names
as one reviewable PR, with no merge or training experiments. After publication
of `9db184b9aa3df9286b07abdafa2d41db0d93e651` in PR #424, Jack Heart requested:
“Let's make sure to both measure those risks and attempt to ameliorate them.”
That accepted pass covers legal-row scoring, compact pending/rollout storage
through both learners and incidents, bounded same-host costs, paired public-state
competency, actual executable-program attribution, and the five CI compatibility
failures. The risk-reduction implementation is committed at `5a57096f2` and
main `1546aae3f` is merged at `d11ca430d`; publication of these changes and
interactive review remain outstanding. No auto-merge, paid execution or strength
experiment is authorized.

The original implementation and publication evidence are retained at
`9db184b9:scratch/mtg-243-naming.md`; the uncommitted research/accepted-pass text
was preserved at `.lf/logs/naming-risk/plan-before-reduction.md` before curation.
The 2026-11-04 reconciliation follows the accepted Task directive and records
durable lessons in the selected coverage Wave through the code Graph.
Earlier T7 direction is historical at
`4463e57f5:scratch/questions.md`.

## Reconciled upstream context — 2026-11-04

Main `1546aae3f` adds the published Standard roadmap and
`managym/PRINCIPLES.md`, renames the Wave directory from engine to coverage,
and introduces separate engine-performance and infrastructure scopes. The sync
changes no runtime or tests relative to `5a57096f2`; existing focused results
remain applicable, with full acceptance still outstanding. Coverage has no
immediate child memory directories; the two sibling memories contain no lessons
yet. Both memory and scratch fit their configured context targets.

Jack Heart's accepted full post-rotation Standard direction supersedes the old
Wave's restriction to Allies/Lessons. The roadmap treats MTG-243 as a foundation
and anticipates naming integration for prepared spells; it does not add prepared
spell execution or broad card authoring to this Task. MTG-245 owns training
methodology, and the roadmap's later training campaigns supply no execution
authorization here. Its API sketches remain proposals.

Main `8e8d702f8` capacity binding and explicit first-legal policy remain
incorporated. MTG-241's latest coordination reports `FrameCompletion`, top
ordering code 27, `SelectCards` code 28 and `OrderReplacement` code 29, with
retained event batches, paid announcements and postambles. These changes are
not synced here: this checkout still has `FrameFinalize` and maximum obligation
26; naming retains modal obligation 10. There is no reported ordinal collision,
but combined behavior and observation compatibility await ordinary sync.

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

## Completed simplifications

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
catalog. The large vocabulary is injected at the native test/model-fixture
boundaries, not admitted as a production world. Greater-than-512 native legality
and model reachability have separate witnesses; a combined large-catalog
scalar/vector/Academy acceptance proof remains outstanding, without broadening
production deck admission to supply it.

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

## Remaining validation and review

### Concept review — 2026-11-04, after risk reduction

Outcome: **no design change and no open product choice within this Task**.
The review of `d11ca430d` and the current working design retains Jack Heart's
accepted complete installed-catalog choice and risk-reduction direction. The
next implementation direction is unchanged; this finding supplies evidence to
loop-or-next and does not select its route or approve delivery.

The experience is one deliberate name selection, then the existing optional
named-card search. A person filters the published offers and chooses “Name Gray
Ogre”; a policy scores the same complete support. Filtering changes no game
state. The bound Command commits the name, constructs one `ExileNamed`, and
resumes the retained frame as `SearchNamed`. Selecting a matching card commits
its exile; “Stop searching and shuffle” ends the search without reversing it.
No confirmation, draft-name state, cancellation or browser-owned legality is
needed. Public candidate identity, executable definition and physical card
remain distinct because they carry different rules and visibility contracts.

The consequential mismatch in the first implementation was contextual:
different name vectors alone could not distinguish equal-rule cards in public
state. The implemented v8 public-definition identity join resolves that mismatch
without changing the interaction or granting hidden membership. Legal-row
scoring and `PackedActions` are internal representations of the same complete
choice, not a second action space. Original slots and full behavior evidence
survive collection, both learners and numerical incidents; only minibatch
materialization recreates dense rows. Checkpoint vocabulary/representation
binding owns compatibility. These changes remove unnecessary per-name programs
and retained padding without inventing a new user-facing lifecycle.

Normal path, failure and recovery agree with that contract: before selection,
reconnect restores naming offers; during search, it restores the committed name,
remaining selections and authorized inspection. A duplicate Command cannot
repeat an exile, and a fresh stale Command is rejected. Retained-session recovery
does not imply process-crash durability. A vocabulary mismatch must reject a
checkpoint before loading weights; recovery requires a compatible runtime/input
binding, not silent name remapping. Learner recovery remains at its existing
empty-pending boundary rather than promising mid-rollout restart.

Usage wording reviewed: “Find an action, choose a name, then choose matching
cards or stop searching.” Academy's existing filter and recovery contract need
no new interaction instructions. `docs/training.md#catalog-naming-input` already
states explicit v8 selection, complete vocabulary capacity, visible identity,
compact storage and checkpoint rejection; its current guidance remains unchanged.
No repository skill needs a new workflow. Broader production name legality and
fresh-default adoption of v8 remain separate future decisions, not unresolved
choices blocking the accepted scope.

Counterexamples remain decisive: a legal name beyond row 511 that cannot be
scored, preferences lost under action reordering, hidden deck membership changing
logits, rebound Commands after filtering/reconnect, lost behavior probabilities,
or restored searches reversing committed exiles. Existing focused witnesses
address these boundaries separately. The next proof is the already-required
combined greater-than-512 scalar/vector/Academy workflow, followed by the
affected gate checks. Neither a two-example head fit nor variable small CPU
samples establish training strength, accelerator behavior or intended-size
throughput. This review inspected source and tests; it reran no runtime checks.

No product decision blocks the installed-catalog scope. Gate owns affected
model/training/recovery suites, package boundaries, current-world admission,
full debug Rust/Clippy, Academy/browser coverage and any current-source receipt
refresh. Frozen evidence must remain untouched. Release extension rebuilt and
atomically installed; the focused native decoder test passes with debug guards.
All five published CI failure cases pass locally in focused checks.

The retained implementation check at `5a57096f2:scratch/mtg-243-naming.md`
records the exact focused command: 110 passed, 1 skipped, plus passing Ruff and
diff checks. No runtime tests or cost probes were repeated for this documentation
reconciliation. Full affected suites, the combined large-catalog proof and
final-source CI remain outstanding; GPU and intended-size throughput claims
remain unmeasured rather than implied by those checks.

The pending PR #424 update covers compact storage/scoring, the reproduced
identity gap and its fix, and these measured costs and limitations. Its
publication boundary remains interactive review. `scratch/naming-risks.html`
holds the current risk assessment; `scratch/pr-review.html` retains the prior
source walkthrough with an update notice. PR #424 remains unmerged; no
auto-merge or experiment occurred. Local HTML/screenshots remain review
artifacts rather than product source; updated pages have no new rendering proof.

Production admission of extra validated names and default adoption of v8 remain
separate future decisions. The current universe includes real back-face names
but does not claim all legal Magic names or change deck admission.

Check (realign): `git diff --check` — passed; `lf context` — memory and scratch below both configured limits; runtime validation reused from `5a57096f2`, full acceptance deferred to gate/CI.
