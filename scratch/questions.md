# MTG-433 open questions and assumptions

Assumptions taken in `plan.md` (kickoff + concept review, headless); each is
reversible before merge.

1. **Strength condition.** AGENTS.md conditions v9/v10 removal on "a v11 pilot
   at least as strong as the best v9 one". No such result is recorded; the best
   player (frontier Small u1952, run `59718fe34945`) is v9. The first implementation slice preserves both versions.
   Removing dense preparation needs a qualifying comparison or an explicit
   change to the condition.
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
    is a small reversible addition in Pass 3 if desired.
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

14. **Pass 2 compatibility.** Keep the existing dense path for v9/v10 until
    the strength decision in item 1. Their native layouts reject even an empty
    batch with “observation batches require policy input v11”. Keep packed
    v11 migrations independently reviewable in the meantime.
