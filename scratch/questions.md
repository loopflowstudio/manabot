# MTG-433 open questions and assumptions

Assumptions taken in `plan.md` (kickoff + concept review, headless); each is
reversible before merge.

1. **Strength condition resolved.** Jack Heart, Task comment
   `9cbbc4d9-f4cb-4065-8398-54442d78d980`: “waive the AGENTS.md
   strength-comparison gate and retire policy inputs v9 and v10 now for
   0.3.0 without running a strength comparison.”
2. **Offline `ObservationBatch` persistence.** Added validated
   `ObservationBatch.to_bytes()` / `from_bytes()` in Rust so `capture.py`,
   `parity.py`, and `decision_memory_probe.py` persist and reload native batches
   through the sole v11 admission/preparation path without keeping dense dict
   admission or breaking the CSR join certificate invariant. Alternative passed
   over: retiring disk parity captures or retaining dense admission for probes.
3. **Drop dict facades (`Mapping` on `ObservationSelection`, `PrefixObservation`,
   `CompactObservation`, `select_rows`, `blank_memory`).** Native selections
   now carry the frozen `RetainedMemory` dataclass and compute
   `reference_distribution` directly on packed actions. The mapping interface,
   dense exports and compatibility wrappers still have consumers.
4. **Counterfactual/invariance interventions.** `combat_probe` and
   `action_cases.identity_removed` perturb `Prepared` via `dataclasses.replace`
   (running `prepare` once instead of 5× for `combat_probe`);
   `reorder_offers`/`reorder_objects` permute `managym.Observation` and
   re-encode via `encode_observation_batch`, deleting `layout.object_references`.
5. **Scalar `Env` returns `managym.Observation`**, including `scenario_refresh()`
   (dropping its redundant 2-tuple). `Env.last_raw_obs` is kept as an alias to
   limit churn; removing it is a follow-up. `RandomMatchupPlayer` in
   `sim/anchors.py` moves to `len(obs.action_space.actions)` with `np.arange`
   (preserving uniform RNG draws; frozen registrations keep recorded digests).
6. **`experiments/tla_frontier.py` moves to v11** (same capacities) because it
   is the base recipe and pins v9 at import. Its executed runs remain described
   by the regime record. Alternative passed over: freeze it and make
   `mana_taps.py` the base.
7. **`manabot.env.vector_env.VectorEnv` is deleted** rather than ported to
   packed buffers.
8. **`LocalUpdateTeacher(native_encoder=False)` is removed**; recorded specs
   that set it false are refused naming `v0.2.5`.
9. **Old search receipts with dense buffers are refused**, not replayed.
10. The optional vector statistics cost trial remains reproducible at
    `a409c94ff3b77f2eab8d10611e6d97743c07fef6`. Its before arm loads
    `76510eaf3f`, which predates packed layout registration. Retired the
    script on main and kept its invocation and both revisions in the report.
11. **`VectorEnv.observations_for_players` optional layout override.** Writing
    `docs/engine.md#fixed-seat-observations` surfaced that removing
    `write_observations_for_players(..., layout=...)` leaves no way to take a
    fixed-seat `ObservationBatch` on a readable-mode `VectorEnv` or under a
    consumer's own `ObservationLayout` without calling `set_observation_layout`
    first. Assumed the existing signature (`observations_for_players(players)`
    using the registered layout) for now; adding optional
    `*, layout: ObservationLayout | None = None, action_overflow: str | None = None`
    is a small reversible addition in Pass 4 if desired.
    **Pass 4:** kept `observations_for_players(players)` on the registered
    layout; no override added. `RecordedVectorEnv` now defines it as a method
    and the `SnapshotVector` protocol names it.
12. **Belief demo root.** Use the first decision after keeping opening hands.
    At the opening modes the actor has no object focus. On the same seeded
    position both input paths gave the same action logits with and without
    Lightning Bolt in the opponent's hand. At the first in-game choice the CLI
    reported a maximum probability change of 0.000002414.

13. **Offline capture format.** Keep a single NumPy archive per collection call
    or learner minibatch. Its `batch` entry holds the native bytes. The writer
    records interned memory once per distinct chunk within the call and saves
    the learner's actual reference probabilities with its other targets. Native
    batch bytes bind the packed layout and producer byte order. Historical
    dense captures replay at the revision that wrote them.

14. **Pass 2 compatibility resolved.** V9/v10 configuration and native layout
    admission are retired together. The packed archive reader preserves the
    existing v11 format header.

15. **Published Torch vector wrapper resolved.** Removed its automatic-refill,
    tensor-reuse and reward-shaping wrapper with the v9/v10 retirement.
    Terminal slots in the native and recorded vectors are explicitly refilled.

16. **Local-search benchmark default.** Compare compact and full evidence on
    the installed runtime. The previous default, reference `841b1c50c`, imports
    the removed `manabot.belief.sampling` module. The November 9 measurement
    is reproducible with its original benchmark checkout `cb75f2166b`. Current
    source-compatible revisions can still be compared with `--reference`.

17. **Historical eligibility evidence.** The retained v4 fixture lacks permanent
    and spell identities. Preserve its text check and assert native v11 refusal.
    The constructed Rust position checks the live target-eligibility columns.

18. **Decision-memory program tokens outside the vocabulary (found in 3b).**
    On Elemental random play, native decision-memory reference details carry
    program tokens (kinds 1, 2, 3; e.g. `(1, 0x02f6b128)`) that appear nowhere
    in `managym.record_vocabulary()`. Native `admission.admit` refuses them;
    the deleted dense path and the retained-prefix featurizer look them up
    with `Tables.tokens`, whose `_lookup` silently returns a neighbouring row.
    Assumed out of scope for MTG-433: the strict check stays, and
    `test_fresh_inputs_stay_within_the_bound[memory]` is a strict xfail
    naming the defect. Needs its own bug Task (native memory token emission
    or `agent/record_vocabulary.rs`), and the retained-prefix featurizer
    should then refuse unknown tokens too.

19. **Sync onto main at `d09f43c286` (2026-11-24).** Main brought MTG-437
    (0.2.6 jace transport: pickling, `RolloutBatch.concatenate`, encoder
    shortcuts) and the published-API contract. Resolution: kept main's
    `packed_seat_axis` name and `ObservationLayout.packed` (always built, v11
    only); kept both native pickle (memo-shared buffers, structural checks) and
    this branch's `to_bytes`/`from_bytes` archive (full join validation) —
    merging them into one format is a follow-up choice. Ported
    `ObservationSelection.concatenate` / `RecordedObservations.concatenate` to
    `RetainedMemory`, gave `RetainedMemory.__reduce__` so unpickling reruns the
    capacity check, dropped the encoder shortcuts and their test, and ported
    `tests/training/test_rollout_transport.py` to native-only rows (dense
    merge and `native and dense` cases deleted).
    **Open:** `scripts/public_api.py check` fails: the snapshot records the
    0.2.x surface this branch removes (`PackedActions.native/fields/unpack`,
    `RecordedObservations.memory*`, `RolloutBatch.policy_rows/ordered_obs`,
    `managym.input_v10_state`, `observation_layout(legacy=, turn_state=)`,
    deprecated encoder aliases). Per patch-clean main this PR is a held
    0.3.0 removal; recording it (`record --allow-break --reason`) or adding
    0.3.0-deprecated aliases for the non-deprecated breaks is undecided.
    **Pass 4:** recorded with `record --allow-break --reason "MTG-433: 0.3.0
    removal …; hold for the 0.3.0 cut"`. Aliases for `PackedActions.fields`,
    `write_observations_for_players` or `encode_active` would need the dense
    paths this PR deletes. The PR must merge with the other 0.3.0 removals
    immediately before the 0.3.0 cut. Reverse by restoring the snapshot.

20. **Rust row reader lives in the library (Pass 4).** The plan named an
    `inspect.rs` reader, but crate unit tests (`src/agent/*`, `src/flow/*`)
    cannot mount `tests/support`, and the `fixtures` feature is not enabled for
    `cargo test -p managym` alone. `managym::agent::observation_rows::InputRows`
    is the one reader (live rows in their dense column reading, no padding), the
    Rust analogue of `ObservationSelection.project`.

21. **Incremental decision-memory cursor removed (Pass 4).** `MemoryEncoding`,
    `encode_suffix` and `RowWriter::reset_memory` served only the dense
    transport's per-slot suffix writes; packed rows always wrote the whole log.
    `decision_memory::write` writes the whole log; `EncodedMemory` (test reader
    of a log) keeps its four tables. `observation_extents::Join` keeps only the
    object joins the batch archive validates (`Extent`/`Span` served the dense
    extent scanner).

22. **Last 0.2.x release name (realign after Pass 4).** The branch names
    `v0.2.5` as the last release supporting v9-v10 (refusal messages in
    `manabot/infra/config.py`, `layout_bindings.rs`, `archive.rs`,
    `flat_mc.py`, docs, CLAUDE.md). Main has since tagged `v0.2.6`, which still
    reads v9/v10, and the PR is held for the 0.3.0 cut, so a later 0.2.x could
    follow. Assumed: name whatever 0.2.x tag is last when the PR merges
    (currently `v0.2.6`); the sweep is in the plan's remaining work.
    **Resolved before publishing (2026-11-25):** swept to `v0.2.6`, the last
    0.2.x tag; `release/v0.3.0/NOTES.md` keeps "deprecated in v0.2.5".
    Rejection, archive, local-update and Etude migration tests: 48 passed.
