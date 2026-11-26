# MTG-487 ship-review notes

## Jack Heart — 2026-11-26 13:24

> 1a 2 what is this? 3  what is hand incarnation

## Jack Heart — 2026-11-26 13:26

> i think we are changing the name Offer to nomencalture around decision group

## Jack Heart — 2026-11-26 13:26

> so those names sshoudl probaly align with the code its building on top off
> but otherwise seems fine

---

## Reading applied

- **Outcome:** Return to `build` for Pass 6 (`1a`).
- **Align with `origin/main` (PR #623, `decision_group` / `decision_surface`):**
  - Sync onto `main`, port `AttackPlanNormalizer` / `BlockPlanNormalizer` + modulo-equivalence `combat_undeclared` from `combat_preview.rs` into `managym/src/flow/decision_group/combat.rs`, and keep `combat_preview.rs` as PR #623's thin deprecated wrapper.
  - Expose optional `aliases` on `managym.decision_group.Offer` (and `InteractionOffer.aliases` on the Academy wire) instead of introducing a new legacy-shaped `Env.decision_offer_aliases_json` method on `Env`; reconcile `structured_offer` → `decision_surface`.
- **Keep as built:** `MATCH_STATE_HASH_VERSION = 32` and all kickoff / concept-review object-equivalence rulings (`3a`).
- **CI & tests:** Fix the ~22 pre-folding CI test assertions across Rust, Academy UI, Academy Contracts, and Integration shards; add a dedicated London mulligan bottom-putting test in `managym/tests/interchangeable.rs`; adopt `managym/perf/baseline.json` from a green CI `Engine Budget` run.
- **Deferred findings:**
  - `compound_offers()` candidate deduplication: **dropped** (already `@deprecated` on `origin/main` for 0.3.0 removal in favor of `Env.decision / Env.preview / Env.submit`).
  - `autotapper.rs` color/output tiebreak before `PermanentId`: **filed as [MTG-495](https://linear.app/loopflow/issue/MTG-495/tie-break-autotapper-equal-reluctance-lands-by-coloroutput-class)** (`task_62704c2f24884e28ab710e8d4c907316`) in `coverage`.
