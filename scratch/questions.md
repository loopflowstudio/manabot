# MTG-337 assumptions and open choices

Active plan: `scratch/payment-search.md`. Assumptions taken while writing the
docs, all reversible:

- The harness is not a CI job and asserts nothing; the steer says "no work
  contract, no suite-wide assertions, no ceilings machinery". The earlier
  `payment_search_tests.rs` and its scoped call limit are removed rather than
  kept beside it ("Drop the single-replay CI test as the deliverable").
- The reporter's recording stays, unchanged, as the `recorded` state.
- State names, the layer list and the three default sizes in
  `docs/development.md#stress-states` are a first cut; the audit may add,
  merge or drop states, and the doc table follows the harness's `--list`.
- Python-side layers (Python encoder, LLM text, Academy frame) are a separate
  script reading recipes printed by the Rust harness, reporting time only. The
  steer asks for Academy frame and LLM text "if they are cheap to reach"; drop
  those rows if they are not.
- The audit table goes in the PR body, not a committed report; the docs keep
  only how to produce it.

Shipping remains on hold from comment `d55bd16b`. The fresh-build replay
compatibility issue below needs resolution before complete validation.

Implementation findings (2026-11-13):
- The restricted-filter recipe uses three card-source spending restrictions
  (Herbalist, Tea Shop, Hideout); Powerstone tokens have another restriction. It
  does not claim to reproduce arbitrary pairwise-different restrictions.
- Rust exports observations and decision captures beside recipes, so Python
  measures the identical injected state without duplicating builders.
- Academy frame construction depends on a live session's match, revision,
  viewer and stop state; it is not a cheap consumer of an exported observation.
  The optional frame row is omitted; text rendering is reachable directly.
- The initial audit stalled at restricted-filter size 8 search and was stopped
  with SIGINT. Its defaults now stop at 2,3,4, and `--list` names this growth.
  Exact mixed-restriction/filter pruning needs spending-purpose conversions and saved-output accounting; see the optimization pass in the plan.
- Frozen viewer JSON omits continuation. Exact Python policy input therefore
  requires an explicit `continuation_json` argument on the existing reader,
  using the existing canonical frame parser. This is a read-only join, not a
  new schema or execution authority. The export carries native frames separately.
- The recorded checkpoint is an `announcement` continuation (Katara X choice);
  the existing text renderer rejects that stage. The Python audit reports the
  unavailable row, and still fails on malformed captures or other renderer
  errors. Fixing LLM announcement coverage is outside the performance harness.

Optimization pass choices (2026-11-13):
- Keep the safe local scan reduction, leaving exact nested-filter pruning and
  cross-step payment caching outside this Task for the reasons in `--list`.
- The catalog supports four distinct restrictions, so add a bounded 2/3/4
  family with Powerstone instead of inventing more restrictions or cards.
- Rebuild benchmark binaries before comparing: an existing executable had a
  different witness digest from HEAD; the source-built baseline and optimized
  build agree on every non-time field. Earlier audit values remain historical.

Open design issue from the fresh source build:
- The unchanged baseline cannot replay the reporter's index-only recording
  (`Action index 2 out of bounds: 2`); optimized debug diagnostics locate step
  145. Earlier passing replay/audit notes used a stale executable. Keep the
  bytes and failure intact; recorded-corpus completion needs a compatible
  historical runtime or independently evidenced semantic mapping, not guessed
  index edits. This blocks complete validation, not the independent scan
  reduction or constructed-state measurements.
