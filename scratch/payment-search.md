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

Auto-Ship was withdrawn in comment `d55bd16b` ("lf task hold"); comment
`9f42783b` later approved shipping with the recorded-state removal below.

## Implemented

- `managym/examples/bench_states.rs`: 17 catalog-backed recipes, 13 layers.
  Default sizes 4/8/16; restricted-filter and distinct-restrictions 2/3/4.
  Checkpoints are reached by ordinary rules actions (combat declarations and
  damage, modal/target choices, chosen sacrifice, trigger ordering, a deep
  stack, Saga naming). JSON lines carry median-of-three ms and deterministic
  payment/layer/trigger work; the digest folds legal actions, ordered first
  payment witnesses, step outcomes and search scores.
- `--recipes` exports exact observations, continuation frames and decision
  captures; `scripts/bench_states.py` times the Python encoder and the text
  renderer over them. `Observation.from_json(..., continuation_json=...)`
  joins the frames through the existing canonical parser (no schema change);
  `tests/verify/test_continuation_input.py` checks every policy buffer.
- `PaymentModel::uses` (`managym/src/flow/mana.rs`) reads the board only for a
  counted tap cost and only as far as its use cap needs. Transition order,
  resource model, bounds and memo are unchanged.
- `docs/development.md#stress-states`, TESTING and the card-batch,
  engine-measure and engine-performance-review skills point at the audit.

## Delete — do not maintain

Done: `managym/tests/payment_search_tests.rs`, `with_call_limit` and its
debug-only instrumentation (counters are benchmark-only); the `recorded`
recipe with its embedded recording, size override and build/checkpoint
branches. Compress pass 2 (2026-11-13) removed what only `recorded` used:
library `benchmark_payments::reporter_game` (the reader is back inside
`bench_payment replay`, as on main), the harness's closed-decklist
`unavailable` search row and empty-board characteristics skip, and the Python
script's `unavailable` text row. A text-renderer error now fails the script.
Nothing further is queued for deletion. The reporter fixture stays
byte-identical (SHA-256
`42c6bec5f6a94fbb189ef40a92c7f197489e75756d3e813eaff063bd996596fa`) with its
Ruff exclusion.

## Constructed-state audit (2026-11-13, release, one machine)

Full default run: 1,425 rows, 186 checkpoints, 17 states, digest
`f0539bcff108e87cef7ea5bc8654078a118616ed3f13858a624d9543ef57fe5d`
(`.runs/mtg337/constructed-final.jsonl`; reproduced after compress pass 2 in
`compress2-all.jsonl`). Local review artifacts, not committed evidence; the
table and findings go in the PR body. Timings are one-machine medians of
three, not assertions.

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


- Two payment shapes. Many small plans: counted-tap's slowest row makes 8,296
  plans / 515,556 calls (at most 67 per plan); mixed 10,213 / 51,479 (at most
  13). Few large plans: restricted-filter 1,966 / 51,897 with up to 2,375 in
  one plan; distinct-restrictions reaches 1,893 in one plan at size 4.
- A restricted-filter search at size 8 stopped producing rows and was
  interrupted without a count. Do not raise its defaults to retry it.
- Static diversity saturates at four cards, mana diversity at seven. Cold
  characteristics cost at these sizes warrants no separate optimization.
- Python consumers: every constructed checkpoint completed in the earlier
  16-state run (`python.jsonl`) and in distinct-restrictions (14 rows over 7
  checkpoints, maximum 1.182 ms). Academy frames need a live session and are
  not measured.
- Catalog audit: `ManaRestriction` appears on Tea Shop, Hideout, Herbalist and
  Powerstone only, so `distinct-restrictions` is bounded at 2..4 and cannot
  establish a growth rate in the number of restrictions.

## Optimization evidence (2026-11-13)

Source-built before/after with identical features
(`.runs/mtg337/optimization-{before,after}.jsonl`, checked field by field by
`compare_optimization.py`): all 320 rows of `counted-tap diverse-mana
restricted-filter mixed` keep their work, legal surfaces, step results and
first witnesses; digest
`dcd55c3f472dc6563cfa53b7e088fe50587addd26fcae62833b0d12de6b646df`.
An earlier comparison used a stale executable and is discarded: rebuild
benchmark binaries before comparing.

| State / checkpoint / size | Before ms | After ms |
|---|---:|---:|
| diverse-mana / cast-choice / 16 | 417.088 | 354.221 |
| mixed / cast-choice / 16 | 290.851 | 260.467 |
| counted-tap / cast-choice / 16 | 1057.352 | 1022.781 |
| restricted-filter / priority / 4 | 106.482 | 103.133 |

Search-call counts are unchanged; the gain is work inside a call. Engine
Budget, source-built: 102,209 steps / digest `17459788495858550277` and the
three search seed digests (`16220228006789220642`, `5713229298486337703`,
`18315821755026080079`) unchanged. Debug `cargo test --lib flow::mana::tests`
passes 21.

Residual costs, named in `--list`, not addressed here:

- Nested-payment bound (diverse-mana, restricted-filter, distinct-
  restrictions). Subtracting activation fees from output is unsafe: restricted
  mana may pay a filter while being unusable for its parent
  (`payment_bound_allows_restricted_mana_to_fund_a_filter`). A tighter exact
  bound must account for spending-purpose conversions and saved outputs.
- Cross-step payment reuse (counted-tap, mixed). Rollouts replan over
  different tapped sets, pools, life and saved obligations; reuse needs its own
  ownership/key/invalidation design. A per-plan limit would not help.

## Recorded game

Task comment `9f42783b`, for Jack Heart: "Decision on the recorded state: (b).
Remove `recorded` from bench_states and its example test; document that the
reporter recording replays only on its original engine (v0.2.3 era) because
actions are stored as list indices. Keep the reporter fixture file
byte-identical and `bench_payment replay` as is. Then gate, open the PR with
the audit table, the optimisation, the named residual costs (nested-payment
bound; cross-step payment reuse) and the replay finding, and land it. Shipping
is approved."

Run evidence: on this branch and on main at `2f00cbbf6d` (release,
source-built) the replay stops with `AgentError("Action index 2 out of
bounds: 2")`, zero-based step 145; the break is upstream of this branch. On
the `v0.2.3` tag (built from `git archive`, needs the repository's `content/`
beside `managym/`) it completes: 426 steps, 2,100 plans, 9,382 calls, 7,167
states, max 180 calls in a plan, digest `16547847303527648701`. The Task
brief's 9,371 / 7,161 / 169 are PR #504-era numbers, not this tag's.
`docs/development.md` "The reported game" documents this.

## Remaining

Compress pass 2 checks: debug `cargo test --features benchmark --example
bench_states --example bench_payment` passes 1; the release default run keeps
the digest above; `scripts/bench_states.py --recipes-file
.runs/mtg337/distinct-recipes.jsonl` completes; Ruff and `cargo fmt` clean.
The library edit touches only `benchmark`-feature code, so the installed
extension is unaffected.

1. Gate, PR body with the state x layer table, the optimization, the two
   residual costs and the replay finding, then land. The text renderer's
   `announcement` stage gap (seen only on the recorded checkpoint) is outside
   this Task.

Earlier narrative (stale-executable discovery, replay CI design):
`e8eb0ab296:scratch/payment-search.md`, `6691ed956:scratch/payment-search.md`.
