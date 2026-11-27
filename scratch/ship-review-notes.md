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

## Jack Heart — 2026-11-26 13:28

> 3a 4b 5b

## Jack Heart — 2026-11-26 16:46

> /ship-review
> A

---

## Reading applied (Pass 9)

- **Outcome:** Return to `build` for Pass 9 (`A`):
  - **`3a`:** Keep all kickoff / concept-review object-equivalence rulings as built (confirmed).
  - **`4b`:** Deduplicate `CandidateSource` lists in `managym/src/agent/decision_surface.rs` (`compound_offers`, `structured_priority_offers`, `structured_attacker_offers` — targets, waterbend taps, attackers, blockers) through `fold_action_space` / equivalence classes with count-based roles where appropriate.
  - **`5b`:** Tie-break equal-reluctance, equal-followup mana sources in `managym/src/flow/mana/autotapper.rs` by deterministic color/output class before `PermanentId` / discovery order, update `autotapper.rs` tests, and remove the `autotapper`-shortcut early stops in `managym/tests/interchangeable.rs` so the behavioral oracle verifies through automatic payment shortcuts (previously filed `MTG-495` was already deleted).
  - **`bench_steps` throughput:** Profile and trim the hot path causing `bench_steps` throughput (`35,772.6/s` vs `39,802.6/s`, −10.1%) to dip 1.1% under the −10% CI floor (`36,184.2/s`), while keeping `bench_search` (−17.9%) and all 25 passing CI checks green; then adopt `managym/perf/baseline.json` at gate/CI.

## Jack Heart — 2026-11-26 23:29

> 1a
> 1b
> and mark complete when shipped

---

## Reading applied (delivery)

- **Outcome:** Confirmed final boundaries (`list[SubjectRef]` aliases on `decision_group.Offer` & `InteractionOffer`, hash v34, `CandidateValue::Subject` count roles on deprecated 0.2.x compound surface) and proceed to `deliver` with `auto` enabled (`1b`) and complete MTG-487 once merged and verified.
