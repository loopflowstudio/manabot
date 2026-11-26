# MTG-487 — one offer per class of indistinguishable objects

Stacked on MTG-439 (PR #616, open). `scratch/ship-review-notes.md` is that
Task's record and is not direction for this one.

## Outcome

When several legal actions differ only in which of several
game-indistinguishable objects they name, the engine offers one. Every
consumer sees the shorter list: search branches less, a policy's 64 action
slots hold more real choices, and Academy's choice list loses its repeated
rows. In Academy, every copy on the board or in hand stays clickable and
sends the one offered Command.

| Position | Today | Target |
| --- | --- | --- |
| Manual payment stop, Kyoshi Village + 3 Plains for `{1}{W}` | 5 offers (Village, Plains ×3, Pay) | 3 (Village, Plains, Pay) |
| Manual payment stop, 8 Forests + 1 Mountain, real choice | 9 taps | 2 taps |
| Priority, `mana_floats="all"`, 8 untapped Forests | 8 `TapForMana` | 1 |
| Two First-Time Flyer in hand, both castable | 2 `CastSpell` | 1 |
| Two Forests in hand, land drop open | 2 `PlayLand` | 1 |
| Discard one card, hand is 2 identical cards | 2-way prompt | no prompt (one action, skipped) |
| Target "creature an opponent controls", 3 identical unreferenced tokens | 3 targets | 1 |
| Convoke with 4 identical untapped tokens | 4 `WaterbendTap` | 1 |
| Declare attackers, 3 identical tokens | 3 yes/no prompts, 8 lines for 4 outcomes | 4 lines: after one declines, the rest are not asked |
| One of the two Forests is enchanted, targeted, or has a counter | separate | separate (unchanged) |
| One of the two hand cards is pinned by a reveal (`ObjectBinding.known_to`) | separate | separate |

## Source of truth

One predicate, `Game::interchangeable(a, b)` over two objects in the same
zone, in a new `managym/src/flow/interchangeable.rs`. It replaces the
mana-only identity and binding checks in
`managym/src/flow/mana/interchangeable.rs`; `determined_source_counts` keeps
its counting argument and calls the shared predicate.

Two objects are interchangeable when all three hold:

1. **Same thing.** Same zone, owner, controller, `definition_id`, `kind()`,
   `form`, neither face down, and — on the battlefield — equal layered
   `characteristics` and an equal `Permanent` record field by field, except
   `spell_entry`, which compares only `(payment.is_some(), kicked, x)`.
   A face-down object (`Object::is_face_down`, any zone) never folds:
   `ObjectForm.backing` holds its hidden printing, so comparing it leaks
   hidden state and changes determinized root spaces, and ignoring it merges
   different cards. Outside-game sideboard objects (`LearnTakeLesson`) have
   no zone; they compare `(owner, definition_id)` through `entry_object`.
2. **Nothing names either one.** Neither id appears anywhere in rules state
   outside its own zone slot and record.
3. **The decision being asked does not order them.** Only decisions on the
   allowlist below are folded.

### Fail-closed by construction — chosen for review

The mana predicate lists the places it checks (`all_unbound_board_wide`,
`locally_unbound_mana_source`). That shape fails open: a new id-holding field
is silently ignored. The engine search found fields it already misses:
`TurnState.graveyard_from_battlefield` (read by rules at
`flow/statics.rs:597,745`), `spells_cast[].object`,
`CastPermissions.spent_types` (`PlayRoute::GraveyardType{source}`),
`ExileLink.exiled_object`, `object_counters`, `PooledUnit.origin`, every
`PendingChoice` other than `Pay` (`flow/game.rs:402-467`), and `EffectFrame`
fields (`flow/decision.rs:119-230`). Mana lands rarely hit them; hand and
graveyard cards and tokens will.

The replacement is a **reference census**: one function walks `GameState`
and `Game`'s pending/suspended decision and yields every `ObjectId`,
`PermanentId`, `CardId` and `Target` it holds. Each struct on the walk is
destructured without `..`, so a new field fails to compile until its author
says whether it names an object. Comparison of `Permanent` records uses the
same exhaustive destructuring, with each ignored field (`id`, slot) justified
beside it. Passed over: extending the enumerated scan (cheaper, but the gap
list above is what that approach produced after one careful Task).

Rulings made here, each a line in `scratch/questions.md`:

- **Census reads live references only.** Exact `ObjectId` (slot +
  generation) and `PermanentId`, never bare slots; `CardId` fields resolve
  through `state.object_of`. Live: pending events, effects, stack, active
  `Payment`, `EffectFrame`, `PendingChoice`, `CastPermissions`,
  `exile_links`, `object_counters`, `TurnState`, `CombatState`. Vouched as
  history and skipped: `events`, `observation_events`, `object_lki`,
  `spell_lki`, zone/object/permanent storage, `library_order_pins`, and a
  settled permanent's `spell_entry.payment` (its only reads are on a pending
  `CardMoved{Battlefield}`, `triggers.rs:1156`; `payment.is_some()`,
  `triggers.rs:1838`; `kicked`, `statics.rs:1231`). Walking it would mark
  every land that ever paid for a live permanent as named forever.
- **Candidate lists are not names.** The current decision's own offer lists
  (`ChooseTargets.legal_targets`, `SelectCards.candidates`,
  `attackers_to_declare`, `blockers_to_declare`) do not distinguish their
  members; what is already chosen (`chosen`, `selected`, `this_way`) does.
  Each such field is vouched individually in the census.
- **Timestamps.** `Permanent.timestamp` orders layer effects (CR 613.7), so
  two copies with any layered static ability
  (`characteristics(p).static_abilities`, so granted statics count) or any
  keyword-counter timestamp are never interchangeable. Copies without them ignore timestamp. Equal ordinary
  counters compare equal (three tokens each with one +1/+1 counter fold).
- **Summoning sickness and turn of entry** keep MTG-439's rule (plain lands
  ignore both; creatures, non-lands, self-animating lands and `power_up`
  permanents compare them).
- **Incarnation.** Two hand cards with different zone histories have
  different `generation`, visible to the owner as `incarnation`
  (`flow/identity.rs:84-101`). It carries no rules or strategic meaning and
  is not compared. An opponent's hand knowledge is per definition
  (`Player.known_hand`, `state/player.rs:108`), so it cannot split copies;
  per-copy knowledge lives in `ObjectBinding.known_to` and
  `pinned_objects`, which the census reads as names.
- **Hand and graveyard order** is not compared (the engine stores it; no
  rule in the catalog reads it). Library cards are never folded: knowledge
  there is positional.

### Where folding happens

One helper, `fold_action_space`, returning `(ActionSpace, Aliases)`, and
one publisher, `publish_action_space(space, aliases)`, which overwrites the
aliases on every call and clears them at game over. Every site that
publishes a built space folds first: `Game::tick` (`flow/tick.rs:103`,
before the automatic-payment and `skip_trivial` checks, for both
`skip_trivial` settings so `ConformancePair` stays in lockstep),
`refresh_sampled_decision` (`flow/search.rs:238-292`, folding before its
`unchanged` comparison so `decision_epoch` does not spuriously advance),
`flow/search.rs:367`, and `scenario_refresh_priority` / naming
(`flow/scenario.rs:83,230`). The first
action of each class in the existing emission order stays, so surviving
actions keep their relative order. A space folded to one action is skipped
like any forced decision, and is correctly treated as no choice for the
CR 104.4b loop watch (`tick.rs:111`).

Folded by action: `PlayLand`, `CastSpell`, `ActivateCardAbility`, `Foretell`
(same profile, route and ability index); `ActivateAbility`, `TurnFaceUp`,
`TapForMana`, `WaterbendTap`; `ChooseTarget` on a permanent or graveyard
card; `SelectCard`, `LearnDiscard`, `LearnTakeLesson`; cost-component picks
(sacrifice, discard, crew, exile from hand, delve); `DeclareBlocker` over
interchangeable attackers.

`SelectCard` is emitted by 14+ decisions, opening hands and `Pay`; it
folds only for candidates in Battlefield, Hand, Graveyard, face-up Exile
or outside-game Sideboard (never Library or Stack), and never under the
decisions below.

Never folded: `ChooseMode` (index is the mode, `flow/mana.rs:2258`),
`ScryCard`, `OrderLibrary` picks, `ChooseTrigger` (choice order is the
result), `DistributeCounters`, `PassPriority`, `PayCost`, `Decline`.

Cost control: actions are first bucketed by (variant, definition, owner,
zone, remaining fields). The census and record comparison run only when a
bucket has two members, once per space, under one `BoardScan`.

### Combat

Attack declaration is one yes/no prompt per creature
(`flow/tick.rs:327-376`), so there is no duplicate inside a list. The
symmetry is across prompts and is broken with one rule: a creature is not
offered "attack" when a creature asked earlier in this declaration declined
and the two are interchangeable now. Its prompt then has one action and is
skipped. The same rule applies to "block" for interchangeable blockers.
Attackers already declared compare by combat role (same defender, no
blockers yet) rather than failing on combat membership, so "block token A"
and "block token B" fold.

Declines are not recorded today (`declare_attacker(_, false)` is a
no-op), so `CombatState` gains `declined_attackers` / `declined_blockers`,
hashed in `CanonicalCombat` (`state/hash.rs:154`) with
`MATCH_STATE_HASH_VERSION` 31 → 32. That bump also adds the missing
`TurnState.graveyard_from_battlefield` to `CanonicalTurn`. The census
vouches the decline lists as non-names. Compound
`AtomicCommand::DeclareAttackers` and `structured_blocker_offers`
(`agent/structured_offer.rs:1519-1590`) lower by exact creature id; they
re-home a plan's marks onto the creature being asked and resolve folded
attacker targets through aliases, or `[A:no, B:yes]` hits `WrongDecision`.
Menace, must-attack and must-block are checked after normalization.

Left offered: two interchangeable blockers assigned to two different
attackers in either order (needs a declaration-start snapshot in combat
state).

Academy's board draft lets a person pick any subset of a pile. The engine's
`Game::combat_preview` (`flow/combat_preview.rs:104-199`) is where a draft
becomes steps, and it normalizes there: when the creature being asked is not
in the plan but an interchangeable, not-yet-asked creature is, the plan's
mark moves to the one being asked; a blocker's planned attacker that was
folded into another resolves to the offered one; `undeclared` compares
modulo equivalence. The browser's draft, toggles, marquee and
pile counter stay as they are; `undeclared` no longer refuses such a plan.

### Aliases

`Game` keeps, beside `current_action_space`, the folded members of each
surviving action: `(action index, object)` pairs, boxed, cloned with the
game, outside the rules-state witness — the shape of
`semantic_object_candidates` (`flow/game.rs:880`). They are written only by
`publish_action_space`, taken and restored with `current_action_space` on a
`step` error (`tick.rs:27,42`), and carried in `GameSnapshot`
(`agent/env/snapshot.rs:112,199`) outside `SearchStateWitness`. `ActionSpace` itself is
unchanged, so its serialization, `PartialEq` use in search
(`flow/search.rs:284`) and every digest input keep their form.

Readers:

- `boardView` puts alias anchors only on drawn objects (battlefield and
  hand); `view.list` and `view.tray` stay one per offer, so graveyard,
  exile and sideboard aliases never duplicate tray cards.
- `decision_offer_subjects` (`agent/decision_context.rs:83`) emits them as a
  new optional `aliases` list on experience `InteractionOffer`
  (`agent/experience.rs:212`), omitted when empty. The native
  `structured_offer::InteractionOffer` and `SEMANTIC_DECISION_VERSION` do not
  change, so `decision_fingerprint` and stored frame comparisons hold.
- Compound lowering (`agent/structured_offer.rs:1554-1590`) matches each step
  by exact `Action` equality against the next published space. After a fold,
  an advertised `{A, C}` can meet a space offering only `B`. Lowering resolves
  a missing action through the alias list. Compound candidate lists are not
  themselves deduplicated in this Task.

## Affected surfaces and consumers

- **Policy input:** layout, token kinds and pinned pilots unchanged; fewer
  action rows. `tests/model/test_pinned_pilot.py` must pass untouched.
  `tests/model/test_choice_separability.py:195-207,398-404` taps two of five
  Forests and expects different digests; it picks sources from different
  classes instead.
- **Academy wire:** `aliases` added to `InteractionOffer` in the Rust type,
  `managym/experience.py`, `academy/ui/src/lib/types.ts`, and the schema
  copies in `protocol/experience-v1`, `study-v1`,
  `study-recorded-decisions-v1`, `canonical-replay-v1` (byte-exact Rust
  tests) and generated `advice-v1`. `server.py:2495-2516` copies the field.
  Readers built before the change reject new frames (`deny_unknown_fields`,
  `extra="forbid"`); server and UI ship together, so this is accepted.
- **Academy UI:** `anchorOf`/`boardView` (`academy/ui/src/lib/board/view.ts:123-148,332-379`)
  place an offer on every alias as well as its subject. Every click, drag,
  hover and target path reads the precomputed `item.offers`, so none changes.
  `pileKey` includes offer labels (`view.ts:200`) and keeps piles whole once
  aliases carry the offer. Payment preview marks lands by `source.id.entity`
  (`payment.ts:53-58`); pointing at an alias shows the plan on the copy the
  engine will tap.
- **Command binding:** a click on any copy sends the surviving offer's id at
  the frame's revision. Stale and duplicate rejection, recording
  (`ReplayDecision`, `DecisionAddressV2`) and fork identity see an ordinary
  Command naming the canonical object.
- **Conformance:** `semantic-kernel-w17-v6` traces hash action spaces;
  re-recorded by the procedure in `conformance/README.md`, zero
  reference/optimized divergence required.
- **Engine Budget:** both digests move. `bench_steps` and `bench_search`
  must be no slower than the MTG-439 baseline before
  `scripts/engine_budget.py record --reason`.
- **Index consumers:** `match_fixtures.rs`, `source_taps.rs:115-161`,
  `tests/academy/test_dual_land_payment.py:111-156`,
  `test_combat_preview.py:89-94`, Playwright `mana-payment`, `drag-play`,
  `combat-presentation`, `laptop-comfort` specs.

## Absent and error states

- No interchangeable pair: the space is published as built, `aliases`
  absent.
- A Command naming an offer id that folding removed is `unknown_offer`, as
  for any id not in the frame.
- Recordings made before this change replay by action index only at their
  own revision; `step_matches` (`academy/recording.py:209-223`) reports the
  first mismatch. Frozen replays and experiment evidence are not
  regenerated.
- Determinized worlds fold an opponent's space per world; the root space is
  the chooser's own and depends only on state the chooser sees.

## Exclusions

- Outcome equivalence between different actions (Tap Mountain vs Pay the
  cost) stays MTG-439's `determined_source_counts`.
- Permutation aliases in multi-pick decisions (pick A then C vs C then A).
- Blocker pairs across two different attackers; library cards; stack
  objects as fold members; trigger ordering.
- A count or "×3" on the folded offer's label; hand grouping in the UI.
- Deduplicating compound offer candidates.

## What goes

- `managym/src/flow/mana/interchangeable.rs`: `same_mana_identity_cheap`,
  `same_layered_mana_permanents`, `locally_unbound_mana_source`,
  `all_unbound_board_wide`, and `Payment::names_mana_source` /
  `may_name_mana_source` once the census covers them.
  `determined_source_counts` and `can_become_creature` survive.
- The definition-only dedups in `resolution_cast_candidates` and
  `revealed_hand_objects` (`flow/decision.rs:3437,3554`): they ignore
  `known_to`, counters and exile links, and strip copies before aliases
  exist, so the second copy in hand is unclickable. The shared folder
  replaces them (Pass 2).
- README's "identical sources are not merged" and "eight Forests are eight
  actions" (`managym/README.md:1001-1003,1064-1067`); `docs/academy.md:1828-1830`
  ("duplicate labels remain separate choices") and the per-copy offer wording
  at `:636-639`.
- Test expectations of one offer per identical source, listed above.

Surviving: MTG-439's skip behavior and labels, action order among survivors,
`ActionSpace`'s shape, policy input layout, Command and recording formats.

## Docs

Written in place ahead of implementation so Pass 1–4 read and keep them
aligned:

- `managym/README.md` (`### Indistinguishable objects`, `### Manual mana sources`,
  `### Mana taps at priority and during payment`,
  `### Previewing a whole combat declaration`)
- `docs/academy.md` (`### Playing on the board`, `### Previewing payments`,
  `## Choice navigation`)
- `docs/engine.md` (`### Forced decisions`)

## Passes

1. **Predicate, oracle, permanents** — done (`e698958617`). Oracle: 133 folded pairs, zero divergences; fold share 10.30%.
2. **Cards and selections** — done (`c440b0d8a4`). Fold share 24.07%, mean actions 3.300 → 2.819.
3. **Combat** — done (`4c9a00ae9b`). `declined_attackers`/`declined_blockers`, hash v31 → v32 (adds `graveyard_from_battlefield`). Fold share 25.07%, 3.332 → 2.832.
4. **Academy and records** — done (`2111195908`). `InteractionOffer.aliases` on all wire/schema copies, drawn-object anchors, conformance `evidence-interchangeable-v1.json` (14 cases / 3,077 commands, zero divergence), pinned pilot games re-recorded. Local release `bench_steps` 118,308 steps / 7.02s, digest `4443070906626209330`.
5. **Concept-review fixes** — done (`cd1b497671`). Coupled decline-pruning into `block_requirements.rs`, shared `AttackPlanNormalizer`/`BlockPlanNormalizer` between `combat_preview.rs` and `structured_offer.rs`, paired `PublishedDecision` (`space` + `aliases`) on `Game` and `GameSnapshot`, added pre-census cheap filtering and shared post-census equivalence helpers in `interchangeable.rs`, made `select_card_decision_allows_fold` exhaustive over `PendingChoice` and `Decision`, and removed dead `TurnFaceUp` arms. Zero conformance action-space hash changes (20/20 pass); local release `bench_steps` unchanged at 118,308 steps / 7.87s (`15,038.7 steps/s`), digest `4443070906626209330` (noisy shared local host; CI Engine Budget baseline adoption deferred to gate/CI).

## Concept reviews — choices made for review

Headless; the recommended option was taken each time, none confirmed by a person.

Pre-implementation (28865774d1): **1a** face-down never folds (passed over: fold under shared information); **2a** settled `spell_entry` compares `(payment.is_some(), kicked, x)` and is census history; **3a** one fold/publish seam, `decision.rs` definition dedups deleted; **4a** combat in this Task with hash v32 (passed over: combat as its own Task); **5a** incarnation and hand/graveyard order ignored, layered statics/keyword counters block folding (passed over: 5b compare incarnation, 5c relax timestamps).

Post-implementation (40cbc3f260), all into Pass 5 except 5a:
- **1a** couple decline-pruning into `block_options` + one shared normalizer. Passed over: 1b skip pruning when blocks are coupled (simpler, leaves coupled declarations unpruned); 1c accept the redundant branch.
- **2a** paired published-decision seam, cheap pre-census filter, one shared equivalence helper. Passed over: 2b fix only the two stale sites and the filter.
- **3a** exhaustive `Decision` match in the SelectCard gate.
- **4a** remove `TurnFaceUp` from foldable helpers and docs (docs done here).
- **5a** keep kickoff rulings; compound candidates and `autotapper` tiebreak deferred (below). Passed over: 5b pull autotapper in; 5c compare incarnation.

## Deferred findings

- **Compound offer candidates.** `compound_offers()` (`agent/structured_offer.rs:708-958`) still lists every identical target, waterbend tap, attacker and blocker; lowering resolves aliases, so it is correct but redundant for compound policies. Takes: dedup `CandidateSource` lists through `fold_action_space` aliases, with count-based roles for identical creatures/taps.
- **`autotapper` tiebreak.** `flow/mana/autotapper.rs` breaks equal-reluctance ties by `PermanentId`, so tapping one of two interchangeable Plains can change a later auto-payment's color; the oracle stops before autotapper shortcuts. Takes: tie-break by color/output before `PermanentId`, then let the oracle run through shortcuts.

## Acceptance check at gate

Rebuild the extension first (`CLAUDE.md`, "After changing Rust under
`managym/`").

```
cd managym && CARGO_BUILD_JOBS=2 cargo nextest run --profile ci --test-threads 2 -E 'binary(interchangeable) | binary(source_taps) | binary(semantic_conformance_tests) | binary(rules_tests)'
uv run --extra dev pytest tests/academy/test_interchangeable_offers.py tests/academy/test_dual_land_payment.py tests/academy/test_combat_preview.py tests/model/test_pinned_pilot.py tests/model/test_choice_separability.py -q
cd academy/ui && npx playwright test e2e/mana-payment.spec.ts e2e/drag-play.spec.ts e2e/combat-presentation.spec.ts
```

`tests/academy/test_interchangeable_offers.py` is the one scenario through
every consumer: Kyoshi Village + 3 Plains at `{1}{W}` yields offers
`[Tap Kyoshi Village for {G}/{W}, Tap Plains for {W}, Pay the cost]`; the
Plains offer lists the other two Plains as `aliases`; the Command for it is
accepted, the recorded game replays to the same state, and a Command at the
old revision is rejected as stale.

## Demo

`./scripts/play`, Settings → Auto-tap off, cast a `{1}{W}` spell with Kyoshi
Village and three Plains: **All legal choices** shows three rows; clicking
the second or third Plains taps a Plains and finishes the payment. With two
copies of one creature card in hand, the list shows one cast row and either
card casts.

## Done when

- Every row of the outcome table holds in the engine, and the
  distinguishing rows stay separate.
- The oracle test finds no divergence, and adding an id-holding field to
  rules state does not compile until the census accounts for it.
- In Academy, any copy of a folded group on the board or in hand sends the
  offered Command; piles stay whole; a drafted attack with any subset of
  identical creatures confirms.
- `bench_steps` and `bench_search` are no slower than the MTG-439 baseline,
  the new baseline is adopted, and conformance shows zero divergence.
- `managym/README.md` and `docs/academy.md` state the equivalence-class
  rule.
