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

- **Pass 6 + compress + QA (`c43f83e3d3`, `046392c616`, `c4ab6840a1`):**
  - Modal collapse moved into `Game::mana_choice_actions` (singleton `Decision::ManaChoice`; `automatic_payment_action` is `Auto` only; all 14 frozen conformance cases pass with zero divergence).
  - `Enclosing::Resolution` compares `(next_pool, cost)` directly; one-shot `StagedAnnouncedActions` replaces the `Revisions`-keyed cache; `interchangeable.rs`/`mana.rs` hot paths reordered and compressed; receipt `conformance/semantic-kernel-w17-v6/evidence-forced-mana-v1.json` recorded; Rust/Python test fallout and `test_no_runner_turns_skipping_off` doc-comment fixed.
  - Note for live Academy demos after main's merge: Auto-tap is toggled off via **Settings → Auto-tap off**.

## Gate / CI

- **CI run `45719643785` (`c5f481ce27`):** `Rust Tests`, `Python Unit Tests`, `Academy UI`, `Protocol v1 Conformance`, `Semantic Kernel Conformance`, `Clean-machine Play`, `Academy Integration (play)`, all 6 `Integration` shards (including `model`), and all `Release packages` jobs passed (`Academy Integration (contracts)` remained in progress when the gate fix batch was prepared). `Engine Budget` failed on the expected step-count/digest diffs (`115,941` vs `118,559` steps) and `search worst reply` at `3.030 s` vs `2.994 s` cap (+36 ms over; `search total` `117.096 s` vs `112.484 s` ok, `steps throughput` `23,551.9/s` vs `24,285.2/s` ok).
- **Gate fixes applied & verified locally:**
  - Short-circuited `production_bonuses` (`managym/src/flow/mana.rs`) when `tapped.matched.is_empty()`, eliminating redundant two-player `layer_views()` scans on every automatic land tap and `ManaChoice` check.
  - Added `need_taps: bool` to `determined_source_counts` (`managym/src/flow/mana/interchangeable.rs`) to skip building/sorting `Vec<UsefulTap>` when `!options.need_taps`.
  - Added `a.cost.mana_value == 0` guard to `Enclosing::Resolution`'s `single_unit_remaining` (`managym/src/flow/mana.rs`).
  - Added length guard to `_pool_delta_mana` (`managym/academy/recording.py`), formatted `managym/academy/recording.py` and `tests/academy/test_dual_land_payment.py` with `ruff format`, and fixed import sorting in `tests/academy/test_bracelet_offers.py`.
- **Remaining before merge:** Once CI finishes on the published fix head, confirm `Engine Budget` timings clear the +10% cap, download its `engine-budget` artifact, and record `managym/perf/baseline.json` via `uv run scripts/engine_budget.py record --reason "..."`.

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
