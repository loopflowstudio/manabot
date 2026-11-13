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

- `bench_states` has 17 catalog-backed recipes and 13 layers. Default sizes
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

## Audit (2026-11-13, release, one machine)

Complete native table: `.runs/mtg337/states.jsonl`; exact exports:
`.runs/mtg337/recipes.jsonl`; Python table: `.runs/mtg337/python.jsonl`.
These are local review artifacts, not new experiment evidence or a committed
report. Put the table/findings in the PR body at delivery.

1,377 native rows, 180 checkpoints, 360 Python rows. Native digest:
`3f139b957ea5c988535f6ed377adcc69cebb86032ceb7bc2d6b62183c65ce5fd`.
All 1,185 cells from the earlier audit retain identical work after adding
combat-damage checkpoints. No runtime algorithm changed.

| State | Sizes | Priority payment calls | Largest row (checkpoint/layer, size) | Median ms |
|---|---|---|---|---:|
| duplicate-lands | 4/8/16 | 10/10/10 | priority/search, 16 | 0.976 |
| herbalist | 4/8/16 | 19/23/31 | priority/search, 16 | 4.324 |
| paid-filter | 4/8/16 | 5/5/5 | priority/search, 16 | 0.648 |
| counted-tap | 4/8/16 | 41/44/76 | cast-choice/search, 16 | 1341.922 |
| waterbend-x | 4/8/16 | 53/41/41 | priority/search, 16 | 33.736 |
| duplicate-creatures | 4/8/16 | 3/3/3 | cast-choice/search, 16 | 3.419 |
| duplicate-tokens | 4/8/16 | 3/3/3 | cast-choice/search, 16 | 3.373 |
| diverse-mana | 4/8/16 | 144/106/181 | cast-choice/search, 16 | 533.195 |
| restricted-filter | 2/3/4 | 100/99/99 | priority/search, 4 | 322.237 |
| diverse-statics | 4/8/16 | 3/3/3 | cast-choice/search, 16 | 24.930 |
| attachments | 4/8/16 | 19/35/67 | cast-choice/search, 16 | 96.848 |
| triggers | 4/8/16 | 3/3/3 | damage/search, 16 | 1.773 |
| zones | 4/8/16 | 15/27/51 | cast-choice/search, 16 | 30.250 |
| mixed | 4/8/16 | 20/43/86 | cast-choice/search, 16 | 352.012 |
| choices | 4/8/16 | 7/7/7 | object/search, 16 | 3.883 |
| names | 4/8/16 | — | name/search, 16 | 0.614 |
| recorded | 13 | — | priority/vector, 13 | 0.914 |

Per-path conclusions:

- **Payment:** interchangeable Forests and Prism remain constant-work at these
  sizes. Diverse/restricted producers are not generally cheap: restricted-filter
  search spends 3,705/68,191/88,826 calls at 2/3/4 sources. An initial size-8
  search stopped producing rows and was interrupted, with no completed count.
  This recipe repeats three card-source restrictions; Powerstone tokens have
  another. Arbitrary pairwise-distinct restrictions are not reproduced. Exact nested-filter pruning remains a
  larger rules-search change and is named in `--list` rather than hidden.
- **Action enumeration:** zones priority uses 15/27/51 calls at 4/8/16, with
  complete choices retained; below 0.2 ms here. No actionable isolated blowup.
- **Attackers/blockers and damage:** individual declarations remain small;
  first-block timings grow roughly with board size. A whole combat still has
  per-creature decisions and repeated scans. The damage checkpoint measures
  the final pass through real damage/SBA processing, verified by departures
  or game over. No choice truncation is proposed.
- **Targets and modes:** Pillar Launch targets and Azula modes are reached
  through casting; target sets grow with objects, mode count stays fixed.
  **Chosen objects and names:** Joo Dee exposes a real sacrifice choice over
  the board; Sozin exposes catalog-wide bounded naming groups. Their individual
  steps stay below 0.1 ms at tested sizes; no combinatorial choice materialization
  was found at these boundaries.
- **Layers:** the four-card static mix and attached Aura/equipment board keep
  one cold board rebuild per row. The small tested costs do not justify a new
  layer optimization; finite catalog diversity is explicitly named.
- **Trigger ordering:** entry matching examines 8/16/32 sources and finds
  4/8/16 triggers. One ordering choice is about 0.015/0.022/0.040 ms. A whole
  ordering round still makes N choices; the harness exposes one boundary at a
  time, not a factorial set of orderings.
- **Search:** the slow search rows have two shapes (from `states.jsonl`
  `plans`/`max_plan_calls`):
  - *One large plan:* diverse-mana spends 13,553 calls in one plan at size 8
    (721 at 16, so not monotonic in N); restricted-filter 3,559/5,092 per plan
    at 3/4 sources. This is payment-search pruning in `flow/mana.rs`.
  - *Many small plans:* counted-tap cast-choice search makes 860/2,412/8,364
    plans at 4/8/16 with at most 137 calls each (533,168 calls, about 1.3 s
    at 16); mixed makes 10,571 plans of at most 13 calls (352 ms). Per-plan
    bounds would not help; the cost is replanning across playout steps and
    the per-call cost of counted tap costs.
- **Observation/clone/Python:** work grows with objects/records and allocated
  capacity. The Python encoder and text renderer complete all supported
  checkpoints; the recorded announcement is the single text coverage gap.
  Largest encode-py: 1.104 ms (attachments, cast-choice, size 16).
  Largest text: 10.957 ms (attachments, cast-choice, size 16).

## Validation and next step

- Debug example construction/combat check: 1 passed in 3.29 s.
- Focused exported-continuation policy-parity/rejection test: 1 passed in 0.64 s.
- Complete release native audit and Python audit completed, with explicit
  unavailable rows above. The previous repeated native audit had identical
  deterministic work and digest; added damage cells intentionally change the
  final corpus digest.
- Release replay: 426 steps; digest `16547847303527648701` unchanged;
  2,100 plans / 9,382 calls / 7,167 states, maximum 180 calls per plan.
  Earlier debug work was 2,234/10,379/8,030: debug invariant probes do extra
  work, so compare the same features/build profile.
- CPython 3.13 release wheel built; extension replaced atomically. Ruff,
  Rust formatting and diff whitespace pass; Loopflow graphs/skills validate.

- Compress (2026-11-13): one reporter-script reader,
  `benchmark_payments::reporter_game`, serves `bench_payment replay` and the
  `recorded` state; `scripts/bench_states.py` passes `--sizes` only when given,
  so its export uses each state's defaults (it had forced 4,8,16 onto
  restricted-filter); the docs' states table gave way to `--list`. Release
  `bench_states`: 1,377 rows, digest `3f139b95…` unchanged; replay counts and
  the herbalist/recorded recipe export byte-identical.

## Remaining

The harness, audit, list and skill edits exist; the optimisation deliverable
does not. Comment `b0a6f0a4` step 3: "Optimise the ones that blow up ...
Where one is too large for this Task, leave it benchmarked and named rather
than half-fixed", delivering "the optimisations made, with before/after
numbers". No runtime algorithm has changed, so:

1. For each slow cell above (diverse-mana and restricted-filter single plans;
   counted-tap and mixed plan counts), either optimise it with before/after
   `bench_states` work and time, keeping legal actions, first witnesses and
   the `bench_steps`/`bench_search` digests unchanged and validating in debug,
   or record in `--list` and the PR body why it is too large for this Task.
   Calling them all deferred without that reason does not meet the steer.
2. The steer names "pairwise-different restricted sources plus a paid filter,
   2^(N+1) states" as a benchmark. restricted-filter repeats three card-source
   restrictions; whether the catalog can build N pairwise-distinct ones is
   unchecked. Build it if real cards allow, otherwise say so in `--list`.
3. Gate, review, PR body with the state x layer table. Shipping stays held
   (comment `d55bd16b`); the text `announcement` gap is outside this Task.

Earlier replay CI implementation history is retained at `6691ed956:scratch/payment-search.md`.
