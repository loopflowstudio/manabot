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
different. Resolution payments (`TurnFaceUp`, `Ward`, etc.) treat simple choice
taps identically to fixed sources.

What the engine offers, measured from base `3be5a0cfb2` (seed 347, floats off,
manual payment, Gray Ogre `{2}{R}`):

| Untapped lands (including a land played this turn) | Base | After |
| --- | --- | --- |
| Gruul Guildgate, Forest, Forest (Guildgate must make `{R}`) | prompt: `ActivateAbility` Guildgate, `PayCost` | no prompt |
| Guildgate, Mountain, Forest (either color pays) | prompt: Guildgate, `PayCost`; then modal `Add {R}` / `Add {G}` | no prompt |
| three Guildgates | six prompts: three source prompts, each followed by a color modal | no prompt |
| Guildgate, Mountain, Mountain, Forest | prompt: Guildgate, `PayCost` only — basics hidden because fixed units equal 3 | Guildgate (`Tap Gruul Guildgate for {R}/{G}`) and all three basics offered beside `PayCost` |
| Mountain × 3 | no prompt | no prompt |
| Mountain × 4 (even if 4th Mountain entered this turn) | three prompts choosing among identical Mountains | no prompt |
| Plains, Plains, Kyoshi Village for `{1}{G}` (even if 2nd Plains entered this turn) | prompt: Kyoshi Village, `PayCost` (and `PayCost` over-tapped all 3!) | no prompt; taps 1 Plains + Village, leaves 1 Plains untapped |
| Soulstone Sanctuary (old) + Soulstone Sanctuary (played this turn) for `{1}` | prompt | prompt (manland sickness matters) |

In Academy the stop for Kyoshi Village + 3 Plains at `{1}{W}` read, per steer
`5d5904ae`: `"(1) Activate ability: Kyoshi Village, (2-4) three separate Tap
Plains for {W}, (5) Pay the cost"`. After: `"Tap Kyoshi Village for {G}/{W}"`
in place of (1); the stop stays because Kyoshi Village vs Plains is a real
choice, and clicking `"Tap Kyoshi Village for {G}/{W}"` skips the color modal
and finishes immediately (1 Plains tapped, 2 Plains untapped), **even when the
3rd Plains was played this turn**.

## Direction received

Quoted from the Task's steers, newest governing:

- Steer `1952c51e`: `Label suggestion from Jack Heart: "Tap Kyoshi Village for {G}/{W} maybe?" — matches fixed sources ("Tap Plains for {W}") cleanly, e.g. "Tap <Name> for {G}/{W}" for duals / explicit color sets and "Tap <Name> for any color" (or "Tap <Name> for mana") for 5-color sources.`
- Steer `69c80076`: "do NOT expand choice-of-color mana sources (dual lands like
  Kyoshi Village, any-color lands like City of Brass, mana creatures/artifacts
  like Birds of Paradise / Arcane Signet, or Great Divide Guide grants) into
  flat per-color tap actions — expanding 5-color sources blows up the action
  space under Great Divide Guide against the 64-action cap. Instead, keep one
  action per choice-of-color mana source with unified tap-for-mana handling and
  wording ("Tap <Name> for mana", never generic "Activate ability: <Name>"),
  followed by the modal color choice when more than one color is genuinely
  available, and auto-skip forced payments / forced color choices consistently
  with basic lands under ManaPayment::Manual (still auditing the Kyoshi Village
  + 3 Plains redundant-tap case from steer 5d5904ae)." Supersedes the earlier
  per-color steers (62057664, 14abfab9, f2013b97).
- "tapping a dual/choice land for mana ... and tapping a basic land for mana
  ... should not be treated or labeled differently — they are the same thing".
- Steer `5d5904ae`: "Audit both (a) forced dual-land payments skipping the
  prompt consistently with basics when there is no strategic choice, and (b)
  whether identical untapped basics / redundant dual-land taps should even
  force a manual payment stop when all combinations leave the same mana/board
  state."
- "Switch from coverage graph to code graph" (done; the Task runs `code`).

## Built (Passes 1–4 + QA fix)

- Pass 1 (`72a12fa277`): exact simple-tap payments skip (one useful output per
  permanent, riders preserved, color choice cannot change a rule-visible
  result), surplus-pruned `automatic_choice_taps` shared with preview, native
  `"Tap <Name> for {G}/{W}"` / `"any color"` labels, immediate
  color-equivalence modal skip.
- Pass 2 (`93bc6923e5`, `7f38f08705`): interchangeable-source classification in
  `managym/src/flow/mana/interchangeable.rs`, single-unit determined-source skip
  in `useful_payment_taps`, determined-completion color-modal skip in
  `equivalent_mana_choices`.
- Pass 3 (`1fc1cda661`): Academy integration tests
  (`tests/academy/test_dual_land_payment.py`, updated
  `tests/academy/test_payment_preview.py`, registered in
  `.github/workflows/academy-integration.yml`), refreshed Academy fixtures,
  updated `managym/README.md` and `docs/academy.md`, verified
  `tests/model/test_pinned_pilot.py` green for `v9` and `v11`.
- QA (`44ebea8e8e`): fixed Academy's `MODAL` prompt for `stage === 'mana_color'`
  to read `"Choose the mana to add"` instead of `"Choose a mode for <Spell>"`
  (`academy/ui/src/lib/board/prompt.ts`).
- Pass 4 (`38aeeb6716`): post-build review fixes — played-this-turn plain
  lands are interchangeable (sickness still compared for creatures, non-lands
  and self-animating lands via `Effect::may_become_creature` /
  `Modification::grants_creature_type`; `entered_this_turn` only with
  `power_up`), and `resolution_mana_abilities` is deleted so `Decision::Payment`
  uses `mana_ability_actions(player, include_simple)`. Real `PlayLand` and
  Soulstone Sanctuary tests in `managym/tests/source_taps.rs` and
  `tests/academy/test_dual_land_payment.py`; three Academy fixtures refreshed.
  Recorded: debug `source_taps` 32/32, acceptance pytest 23/23,
  `test_pinned_pilot.py` v9/v11 green. Review notes: `scratch/concept-review.html`,
  `scratch/qa.html`.

## Determinism rule

Answer to steer `5d5904ae` (b): identical, unreferenced sources alone do not
force a stop — including plain lands played this turn.
- `managym/src/flow/mana/interchangeable.rs`: class predicate requires same
  payment profile, owner, controller, token/kind/form/definition, layered
  characteristics, `spell_entry`, `summoning_sick` (when creature, non-land, or
  self-animating land), `entered_this_turn` (when `power_up` exists), and no
  attachment/stack/trigger/effect/grant/payment reference; anything unvouched is
  a singleton.
- Single-unit count shortcut applies only when every participating source makes
  one unit per useful tap (`Sol Ring` guard stays).
- `equivalent_mana_choices`: color modal skips when every legal mode is fully
  consumed and leaves the same remaining state or the same determined
  completion with equal spend effects.
- When a stop is real, every useful permanent keeps its own offer.

## Consumer contracts

Choice sources retain one `ActivateAbility` per ability, with wording supplied
by native `action.description`. A modal carries the color choices (named
`"Choose the mana to add"` in Academy via `stage === 'mana_color'`). Fixed taps
retain `TapForMana`. Academy reads native labels, and its payment preview and
`PayCost` share the same production prefix. Observation layout, token kinds,
and protocol shapes stay unchanged.

## Absent and error states

- No witness for the cost: nothing offered and nothing skipped.
- A mode that cannot complete payment stays filtered by
  `mana_choice_completes_payment`; 1 remaining mode skips via `skip_trivial`.
- `skip_trivial` off (diagnostic fixtures): `advance_payment` still skips
  determined payments; `tick.rs`'s `ManaChoice` auto-pick respects
  `skip_trivial`.
- Floats `usable`/`all`: extra taps offered, so payment stays open.

## Operational boundary

`advance_payment` runs on every cast in training. Fixed-only affordability uses
the arithmetic fast path; interchangeable classification runs only when a manual
stop would otherwise be published. Deleting `resolution_mana_abilities` removes
an extra `useful_payment_taps` pass on resolution payments. Validate in debug so
cache parity assertions run.

## Exclusions

- Per-color tap actions (superseded by steer `69c80076`).
- Non-simple mana abilities (sacrifice, life, mana/filter, tap-N, counters,
  once-per-turn, non-mana suffix, output-altering triggered mana bonus): remain
  explicit activations with `"Activate ability: <Name>"`.
- Extending `autotapper.rs`'s full hand-followup scoring to choice sources.
- Open multi-source color modal deferral (e.g. 2 Guildgates + Forest + Mountain
  for `{2}{R}` stays faithful to CR 605.3b).
- Merging identical sources into one offer; undo; any change to `Auto`.
- Frozen replays and experiment evidence are not regenerated.

## Docs

`managym/README.md` (`### Automatic source preference`, `### Manual mana
sources`, `### Inspecting automatic payment`) and `docs/academy.md`
(`### Previewing payments`). 
## Remaining — Pass 5 (ship-review decisions, 2026-11-25)

Recorded verbatim in `scratch/ship-review-notes.md` from Jack Heart:
- On Pass 2 & Engine Budget (`search total` +13.5%, `bench_steps` +0.7% wall
  time despite −3.1% steps): *"We should investigate this. this should make
  search go FASTER if done right"* + *"yes"* (keep Pass 2 for basics + duals).
- On ship-review questions 2–5: `2a 3b 4b change 2 to Tap Kyoshi Village for {G}/{W 5c`.

Concrete Pass 5 scope to implement in `build`:

1. **Hot-path performance in `interchangeable.rs`, `mana.rs`, and `payment.rs`
   (keep Pass 2 for basics + duals):**
   - Eliminate the duplicate `announced_mana_actions` / `determined_source_counts`
     work on every payment that stops (`payment.rs:782` checks whether to stop,
     then `action.rs:1142` recomputes the exact same list).
   - In `interchangeable.rs::determined_source_counts`, check whether any two
     taps share `definition_id` / `same_payment_profile` (fast cheap check)
     *before* calling `unbound_mana_source` (which scans `permanents`,
     `effects`, `cast_permissions`, `exile_links`, `stack_objects`,
     `pending_triggers`, and `delayed_triggers` per tap).
   - Avoid full `Characteristics` struct equality in `same_mana_permanent` when
     no continuous effects apply (or cache/short-circuit on unaffected identical
     permanents).
   - In `determined_source_counts`, avoid `model.payable_after_tap` +
     `pool.clone()` + `obligations.to_vec()` per class when `fixed_only`
     arithmetic answers mandatory-class membership directly.
   - Verify with local `bench_steps` that skipping −3.1% decisions yields lower
     total wall time for 512 games, and check `bench_search` per-sim speed.
2. **Resolution payments (`3b`) — per-source manual taps for both basics and
   duals in `Decision::Payment` (`TurnFaceUp`, `Ward`, etc.):**
   - Extend `Decision::Payment` (`managym/src/flow/decision.rs`, `payment.rs`,
     `mana.rs`) so when a resolution payment has a real source choice under
     `ManaPayment::Manual`, it offers per-source manual taps (`TapForMana` and
     simple choice `ActivateAbility`, with the same forced/interchangeable skip
     and color-modal skip rules) for *both* basics and duals — instead of
     omitting simple taps whenever `can_pay`.
   - Keep `2a` (plain non-creature, non-animating lands interchangeable across
     turns; manlands, creatures, non-lands, and `power_up` stay distinct).
   - Add engine tests covering real-choice vs forced `TurnFaceUp` / `Ward`
     resolution payments with both basics and duals.
3. **LLM text observation wording (`4b`, item 2):**
   - Update `manabot/semantic/text_observation.py` (around line 658) so simple
     choice-of-color mana taps follow native `action.description` (e.g.
     `"Tap Kyoshi Village for {G}/{W}"`, `"Tap Birds of Paradise for any color"`)
     instead of `"Activate ability of <Name>"`.
   - Keep verbatim `ManaRestriction::describe()` on restricted simple taps and
     keep `"Activate ability: <Name>"` on non-simple mana abilities (Treasure,
     pain/filter lands).
4. **Academy Moves log mana-pip rendering & resolved color (`5c`, Finding B):**
   - Render mana symbols (`{W}`, `{G}/{W}`, etc.) properly as pips instead of
     literal braces in Academy's Moves log, and record/display the resolved mana
     color actually produced for choice-of-color mana taps rather than only the
     raw offer label. Add frontend/Academy tests covering both basics and duals.
5. **Clear the 7 failing CI jobs on PR #616:**
   - Update the 3 Rust card tests (`ori::pyromancers_goggles_does_not_copy_other_spells_or_without_its_mana`,
     `tla::zhao_taps_nonbasic_lands_and_makes_them_mountains_once_he_conquers`,
     `woe::archons_glory_gains_flying_and_lifelink_only_when_bargained`) that
     assumed exact/identical-basic manual payments stopped.
   - Update the affected Python integration tests (`tests/semantic/test_text_observation.py`,
     `tests/semantic/test_casting_profiles.py`, `tests/env/test_game_snapshot.py`,
     `tests/model/test_input_scales.py`, `tests/sim/test_heuristic.py`,
     `tests/training/test_snapshot_rollouts.py`,
     `tests/verify/test_shared_effect_transport.py`), the authored match parity
     receipt (`tests/academy/test_authored_match_parity.py`), and the Academy UI
     Playwright E2E expectation (`Tap Llanowar Elves for {G}`).
   - **Gate/CI:** After the optimized PR's CI run, verify `Engine Budget`
     timings and adopt the moved digest in `managym/perf/baseline.json` via
     `uv run scripts/engine_budget.py record --reason "..."`.

## Acceptance check at gate

```
cd managym && CARGO_BUILD_JOBS=2 cargo nextest run --profile ci --test-threads 2 -E 'binary(source_taps)'
uv run --extra dev pytest tests/academy/test_dual_land_payment.py tests/academy/test_payment_preview.py tests/env/test_skip_forced_decisions.py -q
```

Rebuild the extension first (`CLAUDE.md`, "After changing Rust under `managym/`").

## Demo

`./scripts/play`, manual payment, a deck with Kyoshi Village: with Village and
one Plains untapped, casting a `{1}{W}` creature puts it on the stack with no
payment prompt. Play a 3rd Plains for turn beside Kyoshi Village + 2 Plains and
cast a `{1}{W}` creature: "All legal choices" reads `"Tap Kyoshi Village for
{G}/{W}"`, three `"Tap Plains for {W}"`, `"Pay the cost"`, and clicking `"Tap
Kyoshi Village for {G}/{W}"` skips the color modal and finishes directly.

## Done when

- A dual land + basic paying an exact cost casts (and turns face up) with no
  prompt under manual payment, in the engine and in an Academy session.
- Simple choice taps read `"Tap <Name> for {G}/{W}"` / `"Tap <Name> for any color"`
  (with unit/restriction detail when needed) in Academy and in LLM text
  observations (`manabot/semantic/text_observation.py`), and never
  `"Activate ability: <Name>"`.
- `automatic_choice_taps` never over-taps surplus lands.
- A color modal appears only when its colors lead to different states — including
  after playing a basic land for turn.
- Identical, unreferenced plain lands (including one played this turn) alone no
  longer stop a payment, while any distinguished, targeted, self-animating
  (manland), creature, or `power_up` source restores the prompt.
- Resolution payments (`TurnFaceUp`, `Ward`) offer per-source manual taps for
  both basics and duals whenever there is a real choice, and skip when forced
  or interchangeable.
- Academy's Moves log renders mana symbols as pips (no literal `{G}/{W}` /
  `{W}` braces) and records the resolved mana color added.
- Hot-path overhead in `interchangeable.rs` / `mana.rs` / `payment.rs` is
  removed so skipping redundant payments makes `bench_steps` faster per game,
  and all failing PR #616 CI tests/parity fixtures are updated.
- Real source, color, float, convoke, improvise, delve, and special mana
  ability choices stay offered, with observation and protocol shapes unchanged.
