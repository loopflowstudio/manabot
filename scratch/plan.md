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

Pass 2: interchangeable-source classification (`managym/src/flow/mana/interchangeable.rs`,
plus reference checks in `decision.rs`, `payment.rs`, `state/src/payment.rs`),
single-unit determined-source skip in `useful_payment_taps`, and
determined-completion color-modal skip in `equivalent_mana_choices`. All 30
`source_taps` tests pass in debug, including Sol Ring and Anthem guards,
identical basics/Guildgates, `Plains × 2 + Kyoshi Village` for `{1}{G}`,
`Village + 3 Plains` for `{1}{W}`, and targeted/enchanted/animated Plains.

## Remaining determinism rule

### Pass 2 — interchangeable sources (steer `5d5904ae` (b))
- Answer to the audit: no, identical sources alone should not force a stop.
- Partition the useful simple sources into classes of **game-indistinguishable
  permanents**. Two permanents `a` and `b` belong to the same class only if
  ALL of the following hold:
  1. Equal `ManaSource` payment profile (`fixed`, `abilities`, `waterbend`,
     `keyword`).
  2. Same `owner`, same `controller`, same `is_token`, same printed `card_def_id`,
     and neither is face-down (`!object.is_face_down() && !permanent.manifested`).
  3. Equal layered `Characteristics` (`self.characteristics(a) == self.characteristics(b)`).
  4. Equal `Permanent` state: both untapped, equal `summoning_sick`,
     `counters.is_empty()`, `damage == 0 && !dealt_damage_this_turn && !deathtouch_damage`,
     not attacking/blocking/in combat, `attached_to.is_none()`,
     `spent_abilities.is_empty() && activated_this_turn.is_empty()`,
     `transforms == 0`, equal `entered_this_turn(turn)` (or equal `entry_turn`),
     and equal `spell_entry`.
  5. **No external live reference names either permanent or its object:**
     no other permanent attached to it (`attached_to == Some(obj)`), no
     `StackObject` (spell, activated ability, triggered ability) or
     `PendingTrigger` targeting or binding it, no `ContinuousEffect`,
     `DelayedTrigger`, `CastGrant`, or active `SuspendedResolution` /
     `EffectFrame` referencing it.
  Anything the predicate cannot vouch for is a singleton class — failure mode is
  conservatively an extra prompt, never a skipped choice (e.g. a Plains targeted
  by land destruction on the stack stays its own class so the player can tap it
  in response).
- A class `C` is **fully needed** when removing one member of `C`
  (`model.after_tap(rep)`) makes the remaining obligations unpayable with the
  current pool (`!after.payable(pool, obligations, SourcePolicy::All)`).
- The payment is determined when:
  1. Automatic payment can fund the cost, and there is no other legal
     activation, help payment, extra float or spend-effect/color choice.
  2. At most **one** class `U` is not fully needed.
  3. For the new count-based shortcut, every useful output of **every**
     participating source produces exactly one spendable unit and leaves no
     surplus. This includes mandatory fixed sources. Keep the existing
     exact-payment proof for multi-unit production. If the existing pool
     could change what is spent or left floating, retain the choice.
  4. With a fixed number of units to produce, subtract the number of
     mandatory taps. The remainder is the number of members tapped from
     the single optional class. Use the same proof for a residual color
     choice before comparing its class counts.
- Handles: `Mountain × 4` for `{2}{R}`, `Plains × 3` for `{W}{W}`,
  `Plains + Plains + Kyoshi Village` for `{1}{G}` (via surplus-pruned
  `automatic_choice_taps`), `3 identical Guildgates` for `{1}{R}`.
- Preserves stops: `Kyoshi Village + 3 Plains` for `{1}{W}` (two non-forced
  classes), `2 Mountains + 2 Forests` for `{2}{R}`, `Plains #1 (targeted on
  stack) + Plains #2` for `{W}`, enchanted/animated Plains among basics.
- When a stop IS real, every permanent keeps its own offer (`managym/README.md`:
  "eight Forests are eight actions"); offers are never merged.

### Color modal (`Decision::ManaChoice`) auto-skip
When a manual seat with `skip_trivial == true` reaches `Decision::ManaChoice`
with `> 1` legal modes (after `mana_choice_completes_payment` filtering):
- Only eligible during an announced mana payment (`resume.is_none()`,
  `self.paying()` is `Some((payer, payment, CostComponent::Mana { .. }))` with
  `payer == player`, `production` is `ManaProduction::Output` with no
  color-dependent triggered mana bonus, `!operation_reads_spent_colors`, and
  all legal modes share the same `SpendEffect`). Never at priority!
- For each legal mode `m`, simulate `commit_produced_mana`'s effect on
  `(cost, pool)`. Every legal mode must be completely consumed (leaving the
  exact same remaining `pool`, so zero chosen surplus floats).
- Skip the modal (taking `automatic_payment_action`'s ranked pick and
  incrementing `skip_trivial_count`) when **either**:
  1. **Immediate state equivalence (Pass 1):** every legal mode leaves the
     **exact same** remaining `ManaCost` and `pool` (e.g. tapping `Kyoshi
     Village` toward `{2}` with `Plains + Island` left — both `{G}` and `{W}`
     leave `{1}` owed, empty pool, identical untapped lands).
  2. **Determined completion (Pass 2):** every legal mode leaves a remaining
     `(cost, pool)` that satisfies the payment-determinism rule above with the
     same tapped count **in each indistinguishable class**, the same floating
     pool and equivalent spend effects (e.g. tapping `Kyoshi Village` toward `{1}{W}` with 3
     identical `Plains` left — `{G}` leaves `{W}` and `{W}` leaves `{1}`, and
     both auto-finish with 1 Plains tapped and 2 Plains untapped).
- Keeps the modal whenever colors diverge in remaining options (e.g. `Kyoshi
  Village` tapped toward `{1}{G}` with `Plains + Forest` left: `{G}` leaves
  `{1}` choosing Plains vs Forest, `{W}` leaves `{G}` forcing Forest), whenever
  surplus floats, on converge spells, or at priority.

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
2. Done — interchangeable sources and determined-completion modal skip (see Built).
3. **Next — Records & Academy tests.**
   - `tests/academy/test_dual_land_payment.py` (exact-cost two-drop with Village + Plains skips; Village + 3 Plains shows `"Tap Kyoshi Village for {G}/{W}"`, three `"Tap Plains for {W}"`, `"Pay the cost"`, zero `"Activate ability"`).
   - Recheck `managym/README.md` ("Manual mana sources") and `docs/academy.md`.
   - `scripts/engine_budget.py record --reason`, `scripts/refresh_academy_fixtures.py` and state-hash pin only if they move, `tests/model/test_pinned_pilot.py` checked (and declared only if it fails).

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
