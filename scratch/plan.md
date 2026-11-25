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
- `manabot/semantic/text_observation.py:658` (frozen LLM prompt wording).
- Academy Moves log raw `{G}/{W}` brace rendering / recording post-modal color
  (pre-existing for basics; separate UI polish).
- Merging identical sources into one offer; undo; any change to `Auto`.
- Frozen replays and experiment evidence are not regenerated.

## Docs

`managym/README.md` (`### Automatic source preference`, `### Manual mana
sources`, `### Inspecting automatic payment`) and `docs/academy.md`
(`### Previewing payments`). 
## Remaining

1. **Gate/CI only:** After the PR's CI run, download its `engine-budget`
   artifact (`measurement.json`) and run `uv run scripts/engine_budget.py record
   --reason "<skipped identical-source payments move bench_steps>"` to adopt the
   moved digest in `managym/perf/baseline.json`.

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
  (with unit/restriction detail when needed), and never `"Activate ability: <Name>"`.
- `automatic_choice_taps` never over-taps surplus lands.
- A color modal appears only when its colors lead to different states — including
  after playing a basic land for turn.
- Identical, unreferenced plain lands (including one played this turn) alone no
  longer stop a payment, while any distinguished, targeted, self-animating
  (manland), creature, or `power_up` source restores the prompt.
- Resolution payments (`TurnFaceUp`, `Ward`) treat simple choice taps
  identically to fixed sources.
- Real source, color, float, convoke, improvise, delve, and special mana
  ability choices stay offered, with observation and protocol shapes unchanged.
