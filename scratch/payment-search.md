# MTG-337 stress states: audit, harness, optimisations, skills

## Direction

Three earlier designs are withdrawn: the single-replay CI test, the suite-wide
work contract, and generated catalog ceilings. The repository owner's words,
from the Task comments (2026-11-12/13):

- "This seems like a bit too on the nose of a solution", "and doesnt go far
  back enough in the 5 whys", "like why did we ship code with a perf issue?"
- On the work contract: "this is too much rigamarole. i think we just need to
  do a better job of identifying plays where computation can blow up, and
  proactively benchmarking and optimzinig htem"
- "and maybe we can identify the kidns of game states that can cause
  pefromance issues ingeneral (large board states with either diverse boards
  or highly duplicative boards) and speed test them through many different
  layers"
- "This should both be a 1-time audit and update our skills for like implement
  or maybe something in the card coverage graph or soethng like that"; "i dont
  want to have development be protocol driven, but i do want to encourage
  doing the wokr that matters"
- Latest steer (comment `353491cd`): "Rebuild toward the latest direction ...
  Drop the single-replay CI test as the deliverable."

Auto-Ship was withdrawn in comment `d55bd16b` ("lf task hold"); nothing since
restores it.

## Implemented

- `bench_states` has 18 catalog-backed recipes and 13 layers. Default sizes
  are 4/8/16, restricted-filter 2/3/4, recorded 13. Ordinary rules actions
  prepare combat declarations and damage, modal/target choices, chosen-object
  sacrifice, trigger ordering, a deep stack and Saga naming. Every checkpoint
  uses the same layer list; missing action families print no step row.
- JSON lines report median-of-three ms and deterministic payment/layer/trigger
  work. JSON output formatting is outside timing. The digest includes legal
  actions, ordered first payment witnesses, step outcomes and search scores.
- Rust exports exact observations, continuation frames and atomic captures
  beside recipes. `scripts/bench_states.py` measures the Python encoder and
  text renderer over them. `Observation.from_json` accepts explicit separate
  continuation JSON through the existing canonical parser; no frozen schema
  changes. The regression test preserves every policy buffer across this join.
- Developer docs, TESTING and the card-batch, engine-measure and performance
  review skills point at this audit. Optional Academy frame measurement needs
  a live session and is omitted. The recorded closed-decklist root reports
  search unavailable; its existing text `announcement` stage is also unavailable.

## Delete — do not maintain

Removed `managym/tests/payment_search_tests.rs`, `with_call_limit`, its limit
state and debug-only payment instrumentation. Counters are benchmark-only.
The reporter fixture remains byte-identical (SHA-256
`42c6bec5f6a94fbb189ef40a92c7f197489e75756d3e813eaff063bd996596fa`),
with its Ruff exclusion and detailed `bench_payment replay` reader retained.

## Current constructed-state audit (2026-11-13, release, one machine)

Final table: `.runs/mtg337/constructed-final.jsonl`: 1,425 rows, 186 checkpoints,
17 constructed states, digest
`f0539bcff108e87cef7ea5bc8654078a118616ed3f13858a624d9543ef57fe5d`.
The final executable preserves all 320 comparison work rows below. The new
family's exact exports are `distinct-recipes.jsonl` and Python results are
`distinct-python.jsonl` beside it: 14 rows over 7 checkpoints; both consumers
complete, maximum 1.182 ms. The `recorded` failure is below, not in this digest.
These are local review artifacts, not committed experiment evidence. Put the
table and findings in the eventual PR body. The earlier executable's audit,
replay and Python findings remain in `7e450d47f:scratch/payment-search.md`;
do not present their numbers as a source-built audit of the merged HEAD.

| State | Sizes | Priority payment calls | Largest row (checkpoint/layer, size) | Median ms |
|---|---|---|---|---:|
| duplicate-lands | 4/8/16 | 4/4/4 | priority/search, 16 | 0.964 |
| herbalist | 4/8/16 | 18/22/30 | priority/search, 16 | 3.517 |
| paid-filter | 4/8/16 | 6/6/6 | priority/search, 16 | 0.856 |
| counted-tap | 4/8/16 | 40/43/75 | cast-choice/search, 16 | 1030.870 |
| waterbend-x | 4/8/16 | 51/39/39 | priority/search, 16 | 25.343 |
| duplicate-creatures | 4/8/16 | 0/0/0 | cast-choice/search, 16 | 3.431 |
| duplicate-tokens | 4/8/16 | 0/0/0 | cast-choice/search, 16 | 3.272 |
| diverse-mana | 4/8/16 | 136/110/188 | cast-choice/search, 16 | 359.905 |
| restricted-filter | 2/3/4 | 103/106/106 | priority/search, 4 | 103.932 |
| distinct-restrictions | 2/3/4 | 48/204/112 | priority/search, 4 | 76.359 |
| diverse-statics | 4/8/16 | 0/0/0 | cast-choice/search, 16 | 23.317 |
| attachments | 4/8/16 | 0/0/0 | cast-choice/search, 16 | 88.249 |
| triggers | 4/8/16 | 0/0/0 | damage/search, 16 | 1.876 |
| zones | 4/8/16 | 0/0/0 | cast-choice/search, 16 | 15.361 |
| mixed | 4/8/16 | 18/42/85 | cast-choice/search, 16 | 269.628 |
| choices | 4/8/16 | 0/0/0 | object/search, 16 | 3.666 |
| names | 4/8/16 | — | name/search, 16 | 0.709 |

- Payment and rollout work has two shapes. Counted-tap's slowest row makes
  8,296 plans / 515,556 calls, at most 67 calls per plan; mixed makes 10,213
  plans / 51,479 calls, at most 13 per plan. Restricted-filter's slowest row
  makes 1,966 plans / 51,897 calls with up to 2,375 in one plan. The finite
  distinct family reaches 1,893 calls in one plan at size 4.
- The earlier size-8 restricted-filter search stopped producing rows and was
  interrupted without a completed count. Do not raise its defaults to retry it.
- Combat declarations/damage, targets/modes, chosen sacrifices, naming groups,
  trigger ordering and zones remain measured at real decision boundaries.
  No strategic action is truncated. A whole combat or ordering round still
  requires multiple choices; the harness does not materialize all sequences.
- Static diversity saturates at four cards; mana diversity saturates at seven.
  Cold characteristics costs at these sizes do not warrant a separate layer
  optimization. Clone/observation work grows with objects and capacities.
- Academy frames need a live session; the optional frame measurement is
  omitted. The earlier Python audit found a text announcement-stage gap;
  fixing renderer coverage remains outside this Task.

## Optimization pass (2026-11-13)

`PaymentModel::uses` reads the board only for a counted tap cost, and only as
far as its use cap needs: a tapped, sacrificed or limited source stops after
one activation's candidates; repeatable costs still count every candidate.
The transition order, resource model, bounds and memo are unchanged. There is
no new runtime cache or schema.

Compress pass (2026-11-13): the bound is one cap and one bounded count instead
of two nested branches; stress-state docs merged the two restricted-size
paragraphs. `bench_states counted-tap diverse-mana restricted-filter mixed`
(release, rebuilt) reproduces all 320 rows' digest `dcd55c3f…46df`; debug
`cargo test --lib flow::mana::tests` passes 21. The installed CPython
extension predates this behavior-preserving edit; gate rebuilds it.

Controlled release comparison: `.runs/mtg337/optimization-before.jsonl` and
`optimization-after.jsonl`, rebuilt from HEAD and the optimized source with
identical features. All 320 rows have identical deterministic work, legal
surfaces, step results, ordered first witnesses and corpus digest
`dcd55c3f472dc6563cfa53b7e088fe50587addd26fcae62833b0d12de6b646df`.
`compare_optimization.py` beside those artifacts checks every non-time field.
The initial comparison used a stale executable and is discarded; rebuilding the
unchanged source resolved its different digest. Do not reuse the stale timing
comparison as evidence for this optimization.

| State / checkpoint / size | Before ms | After ms |
|---|---:|---:|
| diverse-mana / cast-choice / 16 | 417.088 | 354.221 |
| mixed / cast-choice / 16 | 290.851 | 260.467 |
| counted-tap / cast-choice / 16 | 1057.352 | 1022.781 |
| restricted-filter / priority / 4 | 106.482 | 103.133 |

These are one-machine medians of three, not timing assertions or broad speed
claims. Search-call counts are unchanged: the improvement removes work inside
a call. Counted-tap and restricted-filter have only small timing differences.
Residual work is explicitly named in `--list` for each slow state:

- Diverse-mana and restricted-filter need a stronger exact nested-payment
  bound. Subtracting activation fees from output is unsafe: restricted pool
  mana may pay a filter while being unusable for its parent (the existing
  `payment_bound_allows_restricted_mana_to_fund_a_filter` test covers this).
  A tighter bound must account for spending-purpose conversions and saved
  outputs, beyond this local cost reduction; no partial pruning is added.
- Counted-tap and mixed repeatedly plan over different tapped sets, pools,
  life and saved obligations during rollouts. Reusing only the root cost or
  source identity would give stale answers. Cross-step reuse needs a separate
  ownership/key/invalidation design; a per-plan limit would not help.

Catalog audit: `ManaRestriction::` appears at Tea Shop (Ally spells/abilities),
Hideout (Lesson/Shrine spells), Herbalist (Lesson spells), and Powerstone
(except nonartifact spells). Added `distinct-restrictions` with 2/3/4 distinct
restrictions and a paid Prism. Token recipes now name their catalog keys rather
than assuming Ally. No catalog state can realize arbitrary N distinct
restrictions, and this bounded family does not establish 2^(N+1) growth.
`--list` and developer docs state the limit; sizes outside 2..4 are rejected.

Validation: debug `cargo test --lib flow::mana::tests -- --test-threads 2`
passes all 21 tests, including counted taps, limited uses, saved callers,
restricted filter funding and full first-witness comparisons.

Source-built Engine Budget comparison: 102,209 steps / digest
`17459788495858550277` unchanged; all three search seed digests unchanged
(`16220228006789220642`, `5713229298486337703`, `18315821755026080079`).

Counterexample after rebuilding (do not hide): the existing debug example
construction test fails on the final `recorded` recipe. The source-built
**unchanged baseline** also fails `bench_states recorded --layers actions`
with `Action index 2 out of bounds: 2`. With the improved diagnostic the
optimized debug build identifies zero-based step 145, action 2. This is
pre-existing relative to the scan reduction, consistent with the stale
executable found above; the earlier 426-step success cannot describe HEAD.
The full construction test is left failing, the fixture remains byte-identical,
and no index is remapped or historical world rebound. Default all-state audits
therefore fail as well. A current-world replay needs semantic decision mapping
or an explicitly chosen compatible historical runtime; neither can be inferred
from the index-only recording. Stop dependent recorded-corpus claims and leave
this design/compatibility decision to review.

CPython 3.13 release wheel rebuilt and matching extension atomically installed.
Rust format and diff whitespace checks pass.

Focused new-state debug check: `cargo run --features benchmark --example
bench_states --manifest-path managym/Cargo.toml -- distinct-restrictions
--layers actions` succeeds at all 2/3/4 sizes with debug invariants enabled.

## Remaining

1. Resolve recorded replay compatibility before claiming complete corpus
   validation or a passing gate. Do not repair it by changing the frozen script,
   skipping its test or remapping action indices without semantic evidence.
2. Gate, review, PR body with the state x layer table and the optimization,
   residual reasons and replay failure above. Shipping stays held (comment
   `d55bd16b`); the text `announcement` gap is outside this Task.

Earlier replay CI implementation history is retained at `6691ed956:scratch/payment-search.md`.
