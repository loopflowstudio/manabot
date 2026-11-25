# MTG-439 — forced mana payments skip, whatever the source

## Outcome

Under `ManaPayment::Manual`, a payment stops only when the player can end it in
two distinguishable states. Tapping a choice-of-color source is the same thing
as tapping a basic: one offer per simple mana tap, worded to match fixed
sources (`"Tap Kyoshi Village for {G}/{W}"`, `"Tap Birds of Paradise for any
color"`), counted by the same rule that already lets exact-mana basics pay
without a prompt, and executed by a surplus-free automatic payer. The color
modal follows only when its colors lead somewhere different. Resolution
payments (`TurnFaceUp`, `Ward`, etc.) treat simple choice taps consistently
with basics.

What the engine offers today, measured at `f75ba48c72` (seed 347, floats off,
manual payment, Gray Ogre `{2}{R}`):

| Untapped lands | Today | After |
| --- | --- | --- |
| Gruul Guildgate, Forest, Forest (Guildgate must make `{R}`) | prompt: `ActivateAbility` Guildgate, `PayCost` | no prompt |
| Guildgate, Mountain, Forest (either color pays) | prompt: Guildgate, `PayCost`; then modal `Add {R}` / `Add {G}` | no prompt |
| three Guildgates | six prompts: three source prompts, each followed by a color modal | no prompt |
| Guildgate, Mountain, Mountain, Forest | prompt: Guildgate, `PayCost` only — the three basics are hidden because fixed units alone equal 3 | Guildgate (`Tap Gruul Guildgate for {R}/{G}`) and all three basics offered beside `PayCost` |
| Mountain × 3 | no prompt | no prompt |
| Mountain × 4 | three prompts choosing among identical Mountains | no prompt (pass 2) |
| Plains, Plains, Kyoshi Village for `{1}{G}` | prompt: Kyoshi Village, `PayCost` (and `PayCost` over-taps both Plains + Village!) | no prompt; taps 1 Plains + Village, leaves 1 Plains untapped (pass 2) |

In Academy the stop for Kyoshi Village + 3 Plains at `{1}{W}` read, per steer
5d5904ae: "(1) Activate ability: Kyoshi Village, (2-4) three separate Tap
Plains for {W}, (5) Pay the cost". After: `"Tap Kyoshi Village for {G}/{W}"` in
place of (1); the stop stays because Kyoshi Village vs Plains is a real choice,
and clicking `"Tap Kyoshi Village for {G}/{W}"` skips the color modal once pass
2 lands (since both `{G}` and `{W}` finish identically with 1 Plains tapped and
2 Plains untapped).

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

## Built

Pass 1 (`72a12fa277`): exact simple-tap payments skip (one useful output per
permanent, riders preserved, color choice cannot change a rule-visible result),
surplus-pruned automatic payer shared with preview, native
`"Tap <Name> for {G}/{W}"` / `"any color"` labels, immediate color-equivalence
modal skip, exact resolution payments (Ward {2} with Village + Sol Ring still
offers the Village). 25 `source_taps` tests passed in debug; fourteen Forests
under Great Divide Guide gave 29 payment actions for Firebrand Archer.

Pass 2 (`93bc6923e5`): interchangeable-source classification, single-unit
determined-source skip in `useful_payment_taps`, determined-completion
color-modal skip in `equivalent_mana_choices`. All 30
`source_taps` tests pass in debug, including Sol Ring and Anthem guards,
identical basics/Guildgates, `Plains × 2 + Kyoshi Village` for `{1}{G}`,
`Village + 3 Plains` for `{1}{W}`, and targeted/enchanted/animated Plains.

Pass 3 (this commit): added `tests/academy/test_dual_land_payment.py` (exact
two-drop with `Kyoshi Village` + `Plains` previews exact and skips payment;
`Kyoshi Village` + 3 `Plains` shows `"Tap Kyoshi Village for {G}/{W}"`, three
`"Tap Plains for {W}"`, `"Pay the cost"`, zero `"Activate ability"`, and tapping
Village skips the color modal), updated `tests/academy/test_payment_preview.py`
for opening-hand keeps and a distinct blue dual (`Tranquil Cove`) so casting
`Ancestral Recall` keeps a real payment stop, registered both tests in
`.github/workflows/academy-integration.yml`, rechecked `managym/README.md` and
tightened `docs/academy.md` ("spent colors"), and refreshed the Academy fixtures
whose prefixes skipped redundant identical-basic payments
(`precision-demo-opening-v1.json`, `convoke-demo-opening-v1.json`, and
`choice-navigation-opening-v1.json`; `learn` and `precision-mana` unchanged).
`tests/model/test_pinned_pilot.py` passed unchanged for `v9` and `v11` (no
re-record needed), and no card definitions changed (state-hash pin untouched).
Release `bench_steps` moved from 118,559 steps (`digest: 10601622128491155944`)
to 118,674 steps (`digest: 17232083860091493640`, 0 capped); adopting
`managym/perf/baseline.json` via `scripts/engine_budget.py record --reason`
waits for the PR's CI `engine-budget` artifact (`measurement.json`).

## Determinism rule (built)

Answer to steer `5d5904ae` (b): identical sources alone do not force a stop.
Implemented in `managym/src/flow/mana/interchangeable.rs` (class predicate:
same payment profile, owner, controller, token status, definition, layered
characteristics and permanent state, and no attachment/stack/trigger/effect/
grant reference; anything unvouched is a singleton) and
`equivalent_mana_choices` (color modal skips when every color reaches a
determined completion with equal per-class tap counts). The count shortcut
applies only when every participating source makes one unit per useful tap
(Sol Ring guard). When a stop is real, every permanent keeps its own offer.

## Consumer contracts

Choice sources retain one `ActivateAbility` per ability, with wording supplied
by native `action.description`. A modal carries the color choices. Fixed taps
retain `TapForMana`. Academy reads the native label. Its payment preview and
`PayCost` must continue to use the same production prefix. Keep complete offers
when a source or color choice matters, including deliberate priority floats
and the sources that may fund a nested activation.

## Absent and error states

- No witness for the cost: nothing offered and nothing skipped.
- A mode that cannot complete payment stays filtered by
  `mana_choice_completes_payment`; 1 remaining mode skips via `skip_trivial`.
- `skip_trivial` off (diagnostic fixtures): `advance_payment` still skips
  determined payments (matching today's fixed shortcut); `tick.rs`'s
  `ManaChoice` auto-pick respects `skip_trivial`.
- Floats `usable`/`all`: extra taps offered, so payment stays open.

## Operational boundary

`advance_payment` runs on every cast in training. Fixed-only affordability
can use the arithmetic fast path. Choice modes check payable continuations.
Pass 2 should classify interchangeable sources only when a stop would
otherwise be published. Check the engine budget after decision counts settle,
and record a changed baseline with its reason. Validate in debug so that
cache parity assertions run.

## Exclusions

- Per-color tap actions (superseded by steer `69c80076`).
- Non-simple mana abilities (sacrifice, life, mana/filter, tap-N, counters,
  once-per-turn, non-mana suffix, output-altering triggered mana bonus): remain
  explicit activations with `"Activate ability: <Name>"`.
- Extending `autotapper.rs`'s full hand-followup scoring to choice sources
  (surplus pruning + reluctance order in `automatic_choice_taps` is sufficient
  for soundness; full follow-up search for choice sources is separate work).
- `manabot/semantic/text_observation.py:658` (frozen LLM prompt wording).
- Merging identical sources into one offer; undo; any change to `Auto`.
- Frozen replays and experiment evidence are not regenerated.

## Docs

Drafted in place before implementation (recheck wording against final implementation in Pass 3):

- `managym/README.md` — `### Automatic source preference`, `### Manual mana sources`, and `### Inspecting automatic payment`.
- `docs/academy.md` — `### Previewing payments` (auto-skip rules for dual lands and identical sources, simple choice-of-color labels and color-modal skipping, preview table).

## Passes

1. Done — `72a12fa277` (see Built).
2. Done — `93bc6923e5` (see Built).
3. Done — Academy tests, docs recheck, refreshed fixtures, pinned-pilot check, and `bench_steps` check (see Built). Remaining CI follow-up after PR run: download `engine-budget` artifact and run `uv run scripts/engine_budget.py record --reason` to adopt the moved digest.

## Tests

- `managym/tests/source_taps.rs`: forced dual + basic; ambiguous dual + two
  basics; 3 Guildgates; Guildgate + 3 basics offers all 4 + `PayCost`;
  `Plains + Plains + Kyoshi Village` for `{1}{G}` skips AND leaves 1 Plains
  untapped; `Great Divide Guide` + 2 Forests; `Jasmine Dragon Tea Shop` +
  Plains (Ally); `Animal Attendant` + Plains; `Hermitic Herbalist` (`{2}` vs
  `{1}` Lesson); converge keeps prompt; Treasure keeps activation; `TurnFaceUp`
  with Village + Plains skips; tapping `Kyoshi Village` for `{2}` with
  `Plains + Island` skips modal; tapping `Kyoshi Village` for `{1}{G}` with
  `Plains + Forest` keeps modal; identical basics skip; Plains targeted by
  spell on stack restores prompt; enchanted/animated Plains restores prompt;
  offer labels (`{G}/{W}`, `any color`, `2 mana (Lesson spells only)`).
- `managym/src/flow/mana/tests.rs`: shared simple-tap predicate, surplus
  pruning in `automatic_choice_taps`, per-source unit count.
- `tests/academy/test_dual_land_payment.py`: Academy Play session verifying
  exact-mana skip and `"Tap Kyoshi Village for {G}/{W}"` label on stops.
- `tests/env/test_skip_forced_decisions.py` stays green.

## Acceptance check at gate

```
cd managym && CARGO_BUILD_JOBS=2 cargo nextest run --profile ci --test-threads 2 -E 'binary(source_taps)'
uv run --extra dev pytest tests/academy/test_dual_land_payment.py tests/academy/test_payment_preview.py tests/env/test_skip_forced_decisions.py -q
```

Rebuild the extension first (`CLAUDE.md`, "After changing Rust under `managym/`").

## Demo

`./scripts/play`, manual payment, a deck with Kyoshi Village: with Village and
one Plains untapped, casting a `{1}{W}` creature puts it on the stack with no
payment prompt. With Village and three Plains, "All legal choices" reads
`"Tap Kyoshi Village for {G}/{W}"`, three `"Tap Plains for {W}"`, `"Pay the cost"`.

## Done when

- A dual land + basic paying an exact cost casts (and turns face up) with no
  prompt under manual payment, in the engine and in an Academy session.
- Simple choice taps read `"Tap <Name> for {G}/{W}"` / `"Tap <Name> for any color"`
  (with unit/restriction detail when needed), and never `"Activate ability: <Name>"`.
- `automatic_choice_taps` never over-taps surplus lands.
- A color modal appears only when its colors lead to different states.
- Identical, unreferenced sources alone no longer stop a payment, while any
  distinguished or targeted source restores the prompt.
- Real source, color, float, convoke, improvise, delve, and special mana
  ability choices stay offered, with observation and protocol shapes unchanged.
