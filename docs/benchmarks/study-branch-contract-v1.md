# Production Study branch contract v1

Status: executable RUL-6 measurement contract.

## Question

Can the production Etude Study adapter repeatedly fork one exact historical
managym root, publish its viewer-relative native offers, apply an
object-incarnation-bound structured command, and return to the identical
recorded decision within an interactive latency and memory budget—without
private-card exposure, source/sibling drift, untyped recovery, or fallback?

## Admitted path

The only measured lifecycle is:

```text
GameSession.fork_study
  -> StudyForkProvider.fork
  -> retained Env.clone_env
  -> StudyBranch.structured_offers
  -> Env.step_structured
  -> StudyBranch.return_to_recorded
```

The admitted driver is `full_clone/current_game_v1`. The admitted command path
is `structured_offers/step_structured_v1`. A replay reconstruction, legacy
submission bridge, fixed action, card-name dispatch, client legality decision,
candidate cap, retry against a different root, or any other alternate path is
a contract failure.

## Fixture

- Product surface: `etude.server.GameSession` and `etude.study_branch`.
- Match: UR Lessons versus GW Allies.
- Engine seed: 7.
- Hero source-match policy: first Etude-published offer.
- Villain source-match policy: last Etude-published offer.
- Stops: surface every priority window (`auto_pass=false`).
- Study workload: the earliest player-0 retained decision whose native offer
  set contains a targeted `cast` offer with an object source.
- Structured answer: the first authority-published candidate for each required
  choice role.

The fixed source match must terminate in at most 2,000 hero commands and all
four Etude authority fallback counters must be zero before Study measurement.

## Cells

### Sequential interactive cell

After 64 untimed warmups, execute 2,000 complete fork/publish/apply/return
cycles. Record raw monotonic nanoseconds for each phase and end to end. Every
cycle must:

- publish the identical viewer-0 offer projection;
- preserve the exact source `ObjectRef` entity and incarnation;
- move that entity from the acting player's hand to the stack only on the
  child branch;
- expose no opponent hand cards in the returned observation;
- return the same canonical address, frame, offer, command, presentation
  cursor, continuation, and source digest;
- close the branch;
- report one accepted structured command, one committed engine action, and
  zero fallback commands.

### Retained-sibling cell

Retain 512 branches and their native offer sets concurrently. Apply the same
command on alternating branches. Re-publish and compare every untouched
sibling, then return every branch. Record process RSS after garbage collection,
after fork, after offer publication, after alternating applies, and after all
returns.

Every returned sibling must retain the exact source and canonical decision.
Every applied sibling must preserve viewer privacy. No sibling may change the
source offer projection or object reference.

### Failure cell

The receipt must execute and type-check all of these cases:

1. submit before offer publication;
2. unknown offer id, followed by proof that the native offer set was consumed;
3. unsupported native decision surface;
4. invalid replay address;
5. validly encoded but missing replay address;
6. another viewer's replay address;
7. retained-root drift detected during return, consuming the open branch;
8. the same retained-root drift detected on every later fork;
9. caller mutation of the JSON-projected incarnation, with execution still
   bound to the authority-held native offer and the intended entity moved.

Rejected native commands must not change the child digest. None of these cases
may fall back to another executor or root.

## Gates

For the canonical 2,000-cycle / 512-sibling cell:

| Gate | Budget |
|---|---:|
| Fork p95 | <= 1.0 ms |
| Structured apply p95 | <= 1.5 ms |
| Return p95 | <= 1.0 ms |
| End-to-end p95 | <= 3.0 ms |
| Sequential throughput | >= 500 cycles/s |
| Retained-sibling RSS delta | <= 128 MiB |

All exactness, privacy, incarnation, source/sibling drift, replay mutation,
presentation mutation, untyped failure, rejected-command mutation, and fallback
counters must be exactly zero. At least one non-negative object incarnation and
one real child-only zone change must be observed on every applied cycle.

These gates apply only to the exact source closure, compiled extension, host,
fixture, and dimensions recorded by the artifact. They are an interactive
Study integration budget, not a portable engine microbenchmark.

## Evidence and verification

The raw artifact is `experiments/data/rul-6-study-branch-v1.json`; the rendered
summary is `experiments/rul-6-study-branch-v1.md`.

The artifact records:

- contract and artifact SHA-256 values;
- an explicit source closure and digest;
- compiled managym extension path, bytes, and SHA-256;
- platform, Python, CPU, memory, timestamps, argv, and dimensions;
- workload replay/address/source/offer/return identities;
- every raw latency sample and all RSS phase samples;
- recomputable summaries and gates;
- exactness, privacy, incarnation, failure, and execution counters.

`uv run scripts/bench_study_branch.py verify` recomputes the artifact hash,
source and contract identity, every latency summary, every gate, and every
zero-mismatch invariant. It fails closed on missing, stale, malformed, or
non-canonical evidence.
