# MTG-439 — forced mana payments skip, whatever the source

## Outcome

Under `ManaPayment::Manual`, a payment stops only when the player can end it in
two distinguishable states — including on the normal Magic turn where the player
just played a land for turn and then casts a spell. Tapping a choice-of-color
source is the same thing as tapping a basic: one offer per simple mana tap,
worded to match fixed sources (`"Tap Kyoshi Village for {G}/{W}"`, `"Tap Birds
of Paradise for any color"`), counted by the same rule that already lets
exact-mana basics pay without a prompt, and executed by a surplus-free
automatic payer. The color modal follows only when its colors lead somewhere
different — for both announced and resolution payments — while preserving
`ConformancePair` lockstep (`skip_trivial=false` vs `true`) and making the
engine faster per game.

What the engine offers, measured from base `cc32318a35` (manual payment, floats
off, Gray Ogre `{2}{R}`):

| Untapped lands (including a land played this turn) | Base | Target |
| --- | --- | --- |
| Gruul Guildgate, Forest, Forest (Guildgate must make `{R}`) | prompt: `ActivateAbility` Guildgate, `PayCost` | no prompt |
| Guildgate, Mountain, Forest (either color pays) | prompt: Guildgate, `PayCost`; then modal `Add {R}` / `Add {G}` | no prompt |
| three Guildgates | six prompts: three source prompts, each followed by a color modal | no prompt |
| Guildgate, Mountain, Mountain, Forest | prompt: Guildgate, `PayCost` only — basics hidden because fixed units equal 3 | Guildgate (`Tap Gruul Guildgate for {R}/{G}`) and all three basics offered beside `PayCost` |
| Mountain × 3 | no prompt | no prompt |
| Mountain × 4 (even if 4th Mountain entered this turn) | three prompts choosing among identical Mountains | no prompt |
| Plains, Plains, Kyoshi Village for `{1}{G}` (even if 2nd Plains entered this turn) | prompt: Kyoshi Village, `PayCost` (and `PayCost` over-tapped all 3) | no prompt; taps 1 Plains + Village, leaves 1 Plains untapped |
| Soulstone Sanctuary (old) + Soulstone Sanctuary (played this turn) for `{1}` | prompt | prompt (manland sickness matters) |

In Academy, the stop for Kyoshi Village + 3 Plains at `{1}{W}` offers `"Tap
Kyoshi Village for {G}/{W}"`, three `"Tap Plains for {W}"`, and `"Pay the
cost"`. Clicking `"Tap Kyoshi Village for {G}/{W}"` skips the color modal and
finishes immediately (1 Plains tapped, 2 Plains untapped), even when the 3rd
Plains was played this turn. Resolution payments (`TurnFaceUp`, `Ward`) offer
per-source taps for both basics and simple choice sources on real choices, skip
when forced or interchangeable, and skip redundant color modals across both
exact and partial multi-tap resolution payments.

## Direction received

Quoted from the Task's steers and ship-review decisions (2026-11-25), newest
governing:

- Ship-review (`scratch/ship-review-notes.md`):
  - **Performance (`1`):** *"We should investigate this. this should make search go FASTER if done right"* — investigate and eliminate overhead so skipping redundant payments makes games/search faster rather than regressing `Engine Budget`.
  - **Decisions (`2a 3b 4b 5c`):** keep played-this-turn plain lands interchangeable (`2a`, keeping `"Tap Kyoshi Village for {G}/{W}"`), offer per-source manual taps for both basics and duals on resolution payments (`3b`), unify LLM text wording for choice taps with basics (`4b`), and render Moves log mana pips + resolved color in this PR (`5c`).
- Steer `1952c51e`: `"Tap <Name> for {G}/{W}"` for 2–4 explicit modes and `"Tap <Name> for any color"` for 5-color sources.
- Steer `69c80076`: do NOT expand choice-of-color mana sources into flat per-color tap actions (preserves the 64-action cap under Great Divide Guide); keep one action per choice source with unified tap-for-mana wording, modal color choice only when needed, and consistent auto-skip under `ManaPayment::Manual`.
- Steer `5d5904ae`: audit both (a) forced dual-land payments skipping consistently with basics, and (b) identical untapped basics / redundant dual-land taps not forcing a manual stop when all combinations leave the same mana/board state.

## Built (Passes 1–5 + compress + QA)

- **Passes 1–4 (`72a12fa277`..`38aeeb6716`):** exact simple-tap skip, surplus-pruned `automatic_choice_taps` shared with `preview.rs`, native `"Tap <Name> for ..."` labels, `interchangeable.rs` (including played-this-turn plain-land relaxation with `can_become_creature` and `power_up` guards), Academy `mana_color` modal title (`"Choose the mana to add"`), Academy tests & docs (`managym/README.md`, `docs/academy.md`).
- **Pass 5 + compress + QA (`7e5f5b48c4`, `87c42e1734`, `e1bbe1659a`):**
  - `resolution_mana_actions` on `Decision::Payment` (`TurnFaceUp`, `Ward`) offering per-source fixed and simple-choice taps on real choices (`3b`).
  - LLM text observation (`manabot/semantic/text_observation.py`) rendering simple choice taps with the basics' shape (`"Tap for mana P1 (Kyoshi Village). Native description: Tap Kyoshi Village for {G}/{W}"`) (`4b`).
  - Academy Moves log rendering pips via `<ManaText>` in `Rail.svelte` and narrating resolved mana colors in `managym/academy/recording.py` (`5c`).
  - Initial hot-path pruning and authored match parity / fixture updates.

## Pass 6 — Fix conceptual, conformance, performance, and CI defects (Next)

Concept review against `HEAD` (`e1bbe1659a`) and CI run `45682485004` (`032fa20a3f`)
found four real mismatches that must be fixed in code before gate. Execute in
this order (slice 1 first, because it repairs the core engine invariant and
unblocks conformance):

### Slice 1 — Restore `skip_trivial` lockstep (`ConformancePair`)
- **Problem:** `automatic_payment_action` (`managym/src/flow/mana.rs:2220-2228`) currently skips multi-action `Decision::ManaChoice` spaces when `self.skip_trivial && space.actions.len() > 1 && self.equivalent_mana_choices(player, space)`. Because `ConformancePair` compares `reference` (`skip_trivial=false`, stepping only `actions.len() == 1`) against `optimized` (`skip_trivial=true`), 4 frozen conformance cases (`frozen_elemental_world_gr_vs_ug`, `frozen_elemental_world_ug_vs_gr`, `frozen_temur_163`, `frozen_temur_164`) panic with `reference/optimized divergence`.
- **Model to build:**
  - Move `equivalent_mana_choices` out of `automatic_payment_action` and into a dedicated `mana_choice_actions(player, choices)` method used by `Decision::ManaChoice` in `managym/src/flow/decision.rs:1908-1921`.
  - When all feasible modes are equivalent, `mana_choice_actions` collapses the offered list to the single deterministic mode `[Action::ChooseMode { player, mode: first }]` (`len == 1`), matching how `mana_spend_choices` already collapses equivalent spend orders with `limit = 1` (`mana.rs:3865`). This also avoids the redundant `model.plan` loop `automatic_payment_action` previously ran after proving equivalence.
  - Restore `automatic_payment_action` to handle `ManaPayment::Auto` only (`if !self.automatic_mana_payment(player) { return None; }`).
- **Proof:** `cargo nextest run -E 'test(frozen_elemental_world) | test(frozen_temur)'` has zero `reference/optimized divergence` panics (only expected digest shifts before Slice 4 regeneration).

### Slice 2 — Separate resolution-payment modal equivalence from announced-payment subtraction
- **Problem:** `equivalent_mana_choices` (`mana.rs:2885-2942`) runs announced-payment single-unit subtraction (`left.other_than(&unit).spendable_units(&rest.less(candidate)) == 0`) on `Enclosing::Resolution`. Because resolution payments (`Decision::Payment`) never run `commit_produced_mana`, any previously tapped mana stays in `mana_pool` against the unreduced `cost`, causing `left.other_than(&unit)` to reject equivalence on 3+ mana resolution payments (e.g. `TurnFaceUp {2}{W}` or `Ward {3}`: tap `Plains`, then tap `Kyoshi Village` beside `Mountain + Forest`). QA's `e1bbe1659a` only patched exact completion (`next_pool.can_pay(cost)`).
- **Model to build:**
  - Keep `cost_after_produced_unit` strictly for `Enclosing::Announced` (where units commit immediately).
  - For `Enclosing::Resolution(cost)`, evaluate each mode's `next_pool = pool + output` directly against the unchanged `cost`: require equal spend effects, zero unspendable/excess usable surplus (`next_pool.spendable_units(cost) == next_pool.total()` and `next_pool.usable_total(cost) == next_pool.spendable_units(cost)`), and equivalence if EITHER:
    1. every mode satisfies `next_pool.can_pay(cost)` (exact completion), OR
    2. every mode leaves the same effective remaining `ManaCost` after spending `next_pool` unambiguously against `cost` (e.g. `{W}{G}` and `{W}{W}` against `{2}{W}` both leave `{1}`), OR
    3. every mode yields the same `PaymentTaps::Determined` completion via `same_determined(next_pool, cost, true)`.
  - Keep the `ManaFloats::Off` policy on `can_decline()` (`Ward`) payments: skip redundant color modals when floats are off; preserve the modal whenever `offers_extra_payment_taps(player)` is true.
- **Proof:** Add a regression test in `managym/tests/source_taps.rs` covering `TurnFaceUp {2}{W}` / `Ward {3}` with `Plains + Kyoshi Village + Mountain + Forest`: tapping `Plains` then `Kyoshi Village` skips the `{G}/{W}` modal and keeps `Mountain` vs `Forest` open.

### Slice 3 — Fix `CachedAnnouncedActions` soundness and eliminate hot-path overhead (`Engine Budget`)
- **Problem (CI run `45682485004`, job `139446648328`):**
  - `CachedAnnouncedActions` (`mana.rs:587-595, 2348-2393`) keys on `self.state.revisions()`, which `TrackedState` (`game.rs:568-595`) explicitly does NOT advance for `stack_objects`, `pending_triggers`, `trigger_placement`, `delayed_triggers`, or `cast_caller` — all five read by `interchangeable.rs`. It also heap-clones `Payment` + `Vec<Action>` + `Box` on every manual stop.
  - `Engine Budget` is slower than reference on every speed metric: `bench_steps` median elapsed `7.680s → 7.954s` (+3.6% wall time despite 2,618 fewer steps, throughput `15,495.5/s → 14,617.8/s`, −5.7%), `search total` `162.027s → 170.657s` (+5.3%), `search worst reply` `3.887s → 4.400s` (+13.2%, over the +10% cap).
- **Model to build:**
  1. **Sound cache ownership:** Remove the `Revisions`-keyed `CachedAnnouncedActions` cache from `SharedSources`. Bridge `advance_payment` → `payment_action_space` safely (for example, store the computed actions only for the currently published/pending `PendingChoice::Pay`, cleared whenever `PendingChoice::Pay` changes or state mutates, or avoid stale reads across untracked fields) — verified with debug parity checks.
  2. **Arithmetic-first `interchangeable.rs`:** In `determined_source_counts`, check cheap optionality (`fixed_direct` / `payable_without`) BEFORE calling `board_scan()`, `unbound_mana_source()`, or `same_mana_permanent()`. On the common `Forest + Plains ×3` board, seeing that `Forest` is optional returns `None` immediately with zero board scans and zero `characteristics` calls. When `unbound_mana_source` is needed, hoist board-wide scans (`permanents.attached_to`, `effects`, `cast_permissions`, `exile_links`, `stack_objects`, `pending_triggers`, `trigger_placement`, `delayed_triggers`) once per call over the small candidate set rather than once per permanent.
  3. **Hot-path affordability & profile deduplication in `mana.rs`:**
     - Split `choice_tap_payable` (boolean `model.payable(...)`, no witness allocation) from `choice_tap_witness` (`model.plan(...)` for execution/preview only), and use `choice_tap_payable` inside `automatic_payability`.
     - Deduplicate `check_useful(output, is_fixed)` in `useful_payment_taps` by `(source payment profile, output)` so repeated lands (e.g. 6 Great Divide Guide lands × 5 colors) run DFS once per distinct profile instead of once per permanent.
     - Pass `&PaymentModel` into `activation_witness` from `mana_ability_actions_with` instead of rebuilding `self.payment_model(player, None)` per non-simple ability.
     - Short-circuit exact simple-unit counts when `!helped && !options.other_choice && !color_matters` and total untapped units across all sources `<= missing`.
  4. **Profile on the same machine against base `cc32318a35`:** Run release `bench_steps` and `bench_search` (including `--features benchmark` `layer_metrics`/`payment_metrics` and `action_counts`) on both `cc32318a35` and `HEAD`. Eliminate overhead until `bench_steps` wall time and `steps/s` beat base and `bench_search` `total_s` and `worst_s` are within/below base. Record the paired numbers in `scratch/plan.md`.

### Slice 4 — Repair remaining CI failures and regenerate conformance receipts
- **Rust tests (`job 139446647602`):**
  - Update 2 rules tests expecting identical-basic stops: `rules::cr_405_stack::cr_405_1_lifo_resolution`, `rules::cr_602_activated_abilities::cr_602_multiple_activations_stack_and_accumulate`.
  - Update 3 card tests expecting exact-source manual stops: `clb::carnelian_orb_gives_haste_only_to_a_dragon_its_mana_paid_for`, `kld::cultivators_caravan_mana_names_the_caravan_and_crew_names_the_tapped_creatures`, `ori::pyromancers_goggles_copies_a_red_instant_its_mana_paid_for`.
  - After Slice 1 passes lockstep, regenerate `conformance/semantic-kernel-w17-v6` checked receipts (`semantic_conformance` binary) so all 14 `semantic_conformance_tests` pass.
- **Python tests (`jobs 139446648158`, `139446648210`, `139446648248`, `139446648387`):**
  - `tests/academy/test_bracelet_offers.py` (×2): `equip_both()` has 14 identical Islands so `PAY_COST` auto-skips; update helper.
  - `tests/academy/test_table_projection.py::test_the_spell_being_paid_for_is_announced_with_its_target`: give hero a genuinely distinct second mana source (e.g. `Mountain` + `Forest` or `Kyoshi Village`) so `PAY_OR_NOT` stops cleanly.
  - `tests/semantic/test_pooled_mana_classes_input.py` (×2): update `"Activate" in action.description` lookup for `Animal Attendant` to match the unified `"Tap for mana ... Animal Attendant"` wording.
  - `tests/semantic/test_player_control.py`, `tests/env/test_match.py`, `tests/verify/test_immolator_source_power.py`: inspect and update for skipped identical-basic payments.

### Gate / CI (after Pass 6 is green locally)
- Push Pass 6, verify all CI jobs green (except expected `Engine Budget` digest diff), download the `engine-budget` artifact, and adopt `managym/perf/baseline.json` via `uv run scripts/engine_budget.py record --reason "..."` ONLY once speed metrics beat or stay within base budget.

## Determinism rule & contracts

- `interchangeable.rs`: two untapped sources are interchangeable only when they share payment profile, owner, controller, token/kind/form/definition, layered characteristics, `spell_entry`, `summoning_sick` (for creatures, non-lands, and self-animating lands via `can_become_creature`), `entered_this_turn` (when `power_up` exists), and have no attachment/stack/trigger/effect/grant/payment/suspended reference. Single-unit count shortcut applies only when every participating source produces one unit per useful tap (`Sol Ring` guard stays).
- Choice sources retain one `ActivateAbility` per ability with native `action.description`; fixed taps retain `TapForMana`. Observation layout, token kinds, protocol shapes, and v9/v11 pinned pilots stay unchanged.
- `skip_trivial` skips only singleton (`actions.len() == 1`) decision spaces; `equivalent_mana_choices` collapses `Decision::ManaChoice` to length 1 inside the action-space builder.

## Exclusions

- Per-color tap actions (superseded by steer `69c80076`).
- Non-simple mana abilities (sacrifice, life, mana/filter, tap-N, counters, once-per-turn, non-mana suffix, output-altering triggered mana bonus) remain explicit `"Activate ability: <Name>"`.
- Extending `autotapper.rs`'s hand-followup scoring to choice sources.
- Open multi-source color modal deferral (CR 605.3b).
- Merging identical sources into one offer; undo; any change to `Auto`.
- Pre-existing decision-panel raw `{R}` text in "Committed to this cost".
- Frozen replays and experiment evidence are not regenerated.

## Acceptance checks

```
cd managym && CARGO_BUILD_JOBS=2 cargo nextest run --profile ci --test-threads 2 -E 'binary(source_taps) | binary(semantic_conformance_tests) | binary(rules_tests)'
cd managym && CARGO_BUILD_JOBS=2 cargo nextest run --profile ci --test-threads 2 -E 'test(carnelian_orb) | test(cultivators_caravan) | test(pyromancers_goggles)'
uv run --extra dev pytest tests/academy/test_dual_land_payment.py tests/academy/test_payment_preview.py tests/academy/test_bracelet_offers.py tests/academy/test_table_projection.py tests/academy/test_recording_narration.py tests/semantic/test_pooled_mana_classes_input.py tests/semantic/test_player_control.py tests/semantic/test_text_observation.py tests/env/test_match.py tests/verify/test_immolator_source_power.py tests/model/test_pinned_pilot.py -q
```

Rebuild the extension first (`CLAUDE.md`, "After changing Rust under `managym/`").

## Done when

- Exact and interchangeable dual-land / basic-land payments (including plain lands played this turn) skip cleanly across both announced and resolution payments, in engine, LLM text, and Academy.
- `Decision::ManaChoice` collapses equivalent modes inside the action-space builder (`len == 1`), `automatic_payment_action` handles `ManaPayment::Auto` only, and all 14 `semantic_conformance_tests` pass with zero `reference/optimized divergence`.
- Multi-tap resolution payments (`TurnFaceUp {2}{W}`, `Ward {3}`, `Ward {2}`) skip redundant color modals cleanly without running announced-payment single-unit subtraction on pooled mana.
- `CachedAnnouncedActions` has no stale `Revisions` key, and paired release benchmarks show `bench_steps` faster than base `cc32318a35` and `bench_search` within/below base (no +10% regression on `worst_s` or `total_s`).
- All Rust, Python, Academy UI, and conformance CI jobs are green, and `managym/perf/baseline.json` is adopted from a green `Engine Budget` artifact.
