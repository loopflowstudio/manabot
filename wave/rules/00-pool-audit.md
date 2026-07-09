# 00: Card-Pool Capability Audit — Cube (primary), TLA/FIN Commons (intermediates)

> Status: audit for review, 2026-07-09. Not committed. Slots into the
> [rules plan](README.md) as the pool-selection input for stage 11
> (rules-driver cardset expansion) and stage 12 (DSL maturation): it says
> *which cards* pull *which rule families*, in what order.

## Scope

The destination pool is the owner's **Elemental cube**
(https://cubecobra.com/cube/list/elemental): 540 cards, 383 unique
(not singleton — 36x Fabled Passage as the land slot, many 4-ofs of TLA
commons). Candidate intermediate milestones: all commons of **TLA**
(Avatar: The Last Airbender, Nov 2025, 96 commons) or **FIN**
(Final Fantasy, Jun 2025, 96 commons). All oracle data from Scryfall
(sets verified: `tla`, `fin`; cube resolved via `/cards/collection`, 383/383 found).

Cube composition: 171/383 unique cards from TLA + 16 from TLE
(**56% of the 540 weighted cards were printed in TLA sets**); then WOE (23),
STX (15), NEO (14), ELD (12), BLB (12), SOC (10), DFT (10), FIN (7), others.
Rarity: 60 common / 136 uncommon / 132 rare / 55 mythic. No planeswalkers.
80 legendary creatures, 14 sagas, 45 multi-face cards.

## Engine today (rung 0)

From `managym/src/state/ability.rs`, `card.rs`, `permanent.rs`,
`flow/combat_actions.rs`, `flow/triggers.rs`, `flow/sba.rs`:

**Can express**
- Types: creature/instant/sorcery/land enum-complete (enchantment/artifact/
  battle/kindred exist as tags but have no behavior).
- Keywords, fully wired into combat: flying, reach, haste, vigilance, trample,
  first strike, double strike, deathtouch, lifelink, defender, menace.
- Effects (`Effect` enum, 6 variants): `ReturnToHand` (target creature),
  `DealDamage` (target creature / creature-or-player — "any target" is
  equivalent while no planeswalkers exist), `CounterSpell`, `ModifyUntilEot`
  (**source permanent only** — firebreathing works, Giant Growth does not),
  `DrawCards` (resolving player), `MassDamage` (each creature).
- Triggers: exactly one condition — `EntersTheBattlefield { source: This }`,
  single effect, required single target, APNAP ordering, CR 603.3d fizzling.
  `intervening_if` scaffolding exists but only the ETB condition is checkable.
- Activated abilities: mana costs only (no {T}, no sacrifice, no targets at
  activation beyond the effect's spec). Mana abilities: fixed single-mana adds.
- SBA: 0-life loss, draw-from-empty, lethal-damage destruction (incl.
  deathtouch). First/double strike damage passes. Menace block legality.
- Registered pool: ~12 cards (alpha + Pyroclasm + Man-o'-War) + basics.

**Cannot express (nothing of the kind exists)**
- Tokens, counters of any kind, +1/+1 counter P/T interaction.
- Auras/attach, equipment/equip, vehicles/crew, sagas.
- Static/continuous effects (no layers; `temp_power` on the permanent is the
  only P/T modifier).
- Death/attack/cast/upkeep/landfall triggers; optional ("you may") or
  reflexive triggers.
- Life gain/loss as an *effect* (internal `gain_life` exists for lifelink only).
- Discard, mill, exile (no exile zone usage in effects), tap/untap effects,
  sacrifice, search/shuffle, scry/surveil, graveyard recursion.
- Cost modifiers (kicker, additional/alternative costs, cycling), X costs,
  modal spells, multi-face cards, replacement/prevention effects.

## Capability taxonomy and cost model

Each card is labeled with the set of capabilities the engine still needs to
express it completely; empty set = VANILLA (expressible today). Costs are
coarse effort estimates: S=1, M=2, L=4 (L ≈ a `wave/rules` stage-sized diff).
Classification is regex-assisted over oracle text + Scryfall keywords with
manual spot-fixes; expect a few percent label noise on individual cards —
histogram shapes and ladder order are robust.

| Capability | Cost | Meaning |
|---|---|---|
| TARGET_FILTERS | 1 | richer TargetSpec: player-only, artifact/enchantment/permanent, you-control, attacking, up-to-N, multi-target |
| EOT_MODIFIERS | 1 | *targeted*/mass until-EOT P/T and keyword grants ("can't be blocked this turn") |
| LIFE_EFFECTS | 1 | gain/lose life as Effect |
| TAP_UNTAP | 1 | tap/untap effects, doesn't-untap |
| DISCARD / MILL / SAC_EFFECTS | 1 each | new zone-move effects + sacrifice as cost/effect |
| SCRY_SURVEIL | 1 | library-top manipulation |
| MODAL | 1 | choose-one |
| CYCLING | 1 | activated ability from hand (landcycling also needs SEARCH_LIBRARY) |
| DYNAMIC_VALUES | 1 | amounts computed from game state ("equal to the number of…", "for each…") |
| LAND_EXT | 1 | enters-tapped, or-choice mana, any-color mana |
| DEATH/ATTACK/CAST_TRIGGER | 1 each | new trigger conditions on the existing framework |
| TOKENS_ARTIFACT | 1 | Treasure/Clue/Food definitions (after TOKENS) |
| SET_JOBSELECT | 1 | FIN: ETB create Hero + attach (after TOKENS+EQUIPMENT) |
| DESTROY_EXILE | 2 | destroy vs damage, exile zone semantics |
| KEYWORD_NEW | 2 | ward/hexproof/flash/indestructible/protection (targeting, timing, SBA hooks) |
| ACTIVATED_COSTS | 2 | {T}/sac/discard costs, targets, timing restrictions |
| TOKENS | 2 | permanents without printed cards |
| COUNTERS_P1P1 / COUNTERS_OTHER | 2 each | counter state + P/T interaction; stun/lore/charge |
| TRIGGER_MISC | 2 | landfall, another-ETB, phase beginnings, sacrifice-/draw-triggers |
| TRIGGER_CONDITIONAL | 2 | "you may", reflexive ("when you do"), intervening-if (raid) |
| AURA / EQUIPMENT | 2 each | attach machinery (effects ride on STATIC_CONTINUOUS) |
| SEARCH_LIBRARY | 2 | tutor/fetch + shuffle |
| GRAVEYARD_PLAY | 2 | recursion, flashback/cast-from-graveyard |
| COST_MODIFIERS | 2 | reductions, kicker, additional/alternative costs |
| X_COSTS | 2 | {X} in costs |
| VEHICLE | 2 | crew |
| SET_WATERBEND | 2 | TLA: convoke-style cost payment |
| SET_FIREBEND | 2 | TLA: attack trigger adding until-end-of-combat mana |
| SET_AIRBEND | 2 | TLA: exile + owner may cast for {2} |
| SET_TIERED | 2 | FIN: modal additional costs |
| STATIC_CONTINUOUS | 4 | layers-lite: as-long-as statics, anthems, enchanted/equipped-gets |
| SAGA | 4 | lore counters, chapters, sacrifice |
| MULTIFACE | 4 | adventure/MDFC/transform |
| REPLACEMENT | 4 | would-instead, prevention, enters-with-counters |
| SET_EARTHBEND | 4 | TLA: land animation + counters + delayed return trigger |
| SPECIAL_HARD | 4 | long tail: copying, control change, outside-the-game (learn), trigger doubling, impulse-play |

## Capability histograms

### Cube (primary target) — 383 unique / 540 weighted, 1 card VANILLA today

Top lines (unique-card counts; % of 383):

```
TARGET_FILTERS 126 (33%)   SPECIAL_HARD    55 (14%)   CAST_TRIGGER   26 (7%)
TOKENS          93 (24%)   MULTIFACE       45 (12%)   SCRY_SURVEIL   25 (7%)
EOT_MODIFIERS   84 (22%)   SAC_EFFECTS     45 (12%)   GRAVEYARD_PLAY 23 (6%)
DYNAMIC_VALUES  70 (18%)   TOKENS_ARTIFACT 43 (11%)   COUNTERS_OTHER 21 (5%)
ACTIVATED_COSTS 65 (17%)   DESTROY_EXILE   42 (11%)   SET_EARTHBEND  20 (5%)
TRIGGER_MISC    61 (16%)   ATTACK_TRIGGER  38 (10%)   LAND_EXT       18 (5%)
COUNTERS_P1P1   58 (15%)   LIFE_EFFECTS    38 (10%)   SET_FIREBEND   16 (4%)
STATIC_CONT.    56 (15%)   COST_MODIFIERS  35 ( 9%)   SAGA           14 (4%)
                           TRIGGER_COND.   35 ( 9%)   SET_WATERBEND  14 (4%)
also: DISCARD 31, DEATH_TRIGGER 29, KEYWORD_NEW 27, SEARCH_LIBRARY 15,
MODAL 14, TAP_UNTAP 14, REPLACEMENT 12, MILL 12, X_COSTS 12, SET_AIRBEND 10,
VEHICLE 7, EQUIPMENT 5, AURA 4, CYCLING 1
```

40 capabilities total, summed cost 76. Bending mechanics touch 59 unique /
81 weighted cards (15%). MULTIFACE (45 unique, 12%) is the single largest
capability that appears in **neither** candidate set's commons.

### TLA commons — 96 cards, 6 VANILLA today

```
EOT_MODIFIERS   23 (24%)   TOKENS_ARTIFACT  8 (8%)    DEATH_TRIGGER   5 (5%)
TARGET_FILTERS  18 (19%)   SEARCH_LIBRARY   7 (7%)    CYCLING         5 (5%)
ACTIVATED_COSTS 18 (19%)   DESTROY_EXILE    7 (7%)    MODAL           5 (5%)
SAC_EFFECTS     17 (18%)   LIFE_EFFECTS     6 (6%)    ATTACK_TRIGGER  4 (4%)
COUNTERS_P1P1   14 (15%)   SET_EARTHBEND    6 (6%)    TRIGGER_COND.   4 (4%)
LAND_EXT        14 (15%)   SET_WATERBEND    6 (6%)    SCRY_SURVEIL    3 (3%)
TOKENS          13 (14%)   DISCARD          6 (6%)    AURA            3 (3%)
STATIC_CONT.    13 (14%)   TAP_UNTAP        6 (6%)    DYNAMIC_VALUES  3 (3%)
COST_MODIFIERS   5 (5%)    TRIGGER_MISC     5 (5%)    SET_FIREBEND    2 (2%)
also: CAST_TRIGGER 2, MILL 2, SET_AIRBEND 1, SPECIAL_HARD 1, COUNTERS_OTHER 1,
GRAVEYARD_PLAY 1, EQUIPMENT 1
```

34 capabilities, summed cost 58. 15/96 commons touch a bending mechanic.
No sagas, no vehicles, no X costs, no multiface at common.

### FIN commons — 96 cards, 8 VANILLA today

```
TARGET_FILTERS  25 (26%)   ACTIVATED_COSTS  9 (9%)    COUNTERS_OTHER  5 (5%)
EOT_MODIFIERS   18 (19%)   LIFE_EFFECTS     9 (9%)    DYNAMIC_VALUES  5 (5%)
STATIC_CONT.    15 (16%)   SET_JOBSELECT    8 (8%)    TRIGGER_COND.   5 (5%)
TOKENS          12 (12%)   EQUIPMENT        8 (8%)    ATTACK_TRIGGER  5 (5%)
DESTROY_EXILE   12 (12%)   TRIGGER_MISC     7 (7%)    SCRY_SURVEIL    5 (5%)
LAND_EXT        12 (12%)   GRAVEYARD_PLAY   7 (7%)    DEATH_TRIGGER   4 (4%)
SAC_EFFECTS     10 (10%)   DISCARD          7 (7%)    MODAL           3 (3%)
SEARCH_LIBRARY   9 ( 9%)   TAP_UNTAP        6 (6%)    SAGA            3 (3%)
CYCLING          6 ( 6%)   COUNTERS_P1P1    6 (6%)    SET_TIERED      3 (3%)
also: COST_MODIFIERS 6, CAST_TRIGGER 6, TOKENS_ARTIFACT 3, SPECIAL_HARD 2,
KEYWORD_NEW 2, REPLACEMENT 2, MILL 2, AURA 1, X_COSTS 1, VEHICLE 1
```

37 capabilities, summed cost 65. 11/96 commons touch a set mechanic
(8 job select equipment, 3 tiered). FIN's commons closure includes SAGA
(3 Summons), VEHICLE, X_COSTS, REPLACEMENT — heavier machinery per set-only
card than TLA's.

## Unlock ladders (greedy set-cover, gain-per-cost)

Greedy caveat: because nearly every card needs 2+ capabilities, early rungs
unlock few cards individually; the curve steepens once the shared substrate
(targets, EOT modifiers, tokens, counters, triggers) is in.

### Cube, weighted by copies (540) — first rungs and landmarks

```
 1. DISCARD          +5   ->   6/540 ( 1%)      16. SEARCH_LIBRARY  +42 -> 199 (37%)
 2. TARGET_FILTERS   +9   ->  15/540 ( 3%)      19. TRIGGER_MISC    +31 -> 263 (49%)
 3. COST_MODIFIERS   +10  ->  25/540 ( 5%)      20. COUNTERS_P1P1   +22 -> 285 (53%)
 4. EOT_MODIFIERS    +6   ->  31/540 ( 6%)      26. MULTIFACE       +32 -> 388 (72%)
 5. DYNAMIC_VALUES   +6   ->  37/540 ( 7%)      30. SET_EARTHBEND   +25 -> 452 (84%)
 6. DESTROY_EXILE    +11  ->  48/540 ( 9%)      36. SAGA            +14 -> 513 (95%)
 9. SPECIAL_HARD     +18  ->  78/540 (14%)      37. REPLACEMENT     +14 -> 527 (98%)
11. STATIC_CONT.     +20  -> 104/540 (19%)      40. CYCLING         +1  -> 540 (100%)
```

Cube-lite sizes: rung 1 = 6 cards (1%), rung 2 = 15 (3%), rung 3 = 25 (5%).
(Unique-card ladder is similar; 50% unique coverage arrives around rung 22-23,
after STATIC_CONTINUOUS.) The big single-rung jumps are SEARCH_LIBRARY (+42,
dominated by 36x Fabled Passage), MULTIFACE (+32), TRIGGER_MISC (+31),
SET_EARTHBEND (+25).

### TLA commons ladder — landmarks

```
 1-6:  COST_MODIFIERS, TOKENS, CAST_TRIGGER, COUNTERS_P1P1, DEATH_TRIGGER,
       TOKENS_ARTIFACT                          -> 16/96 (17%)
 7-11: EOT_MODIFIERS, TARGET_FILTERS, DISCARD, TAP_UNTAP, ATTACK_TRIGGER
                                                -> 27/96 (28%)
12: STATIC_CONTINUOUS  +6 -> 33/96 (34%)    20: SET_EARTHBEND +5 -> 54 (56%)
13: SET_WATERBEND      +4 -> 37/96 (39%)    22: SAC_EFFECTS  +13 -> 69 (72%)
                                            28: CYCLING       +5 -> 89 (93%)
34 rungs -> 96/96 (100%), total cost 58
```

### FIN commons ladder — landmarks

```
 1-8:  LAND_EXT(+11), DYNAMIC_VALUES, EOT_MODIFIERS, LIFE_EFFECTS, MILL,
       SCRY_SURVEIL, GRAVEYARD_PLAY, TARGET_FILTERS -> 30/96 (31%)
16: CYCLING           +5 -> 50/96 (52%)     29: SAGA         +3 -> 81 (84%)
17: STATIC_CONTINUOUS +4 -> 54/96 (56%)     37: EQUIPMENT    +8 -> 96 (100%)
21: ACTIVATED_COSTS   +6 -> 66/96 (69%)     total cost 65
```

Note the FIN shape: its last rung (EQUIPMENT, closing the 8 job-select
equipment) is a +8 cliff, and its long tail (SAGA, VEHICLE, REPLACEMENT,
X_COSTS, KEYWORD_NEW) is machinery the cube *also* wants — but FIN buys **zero
direct cube cards** with it.

## Hardest cards (the tail that defines "done")

**Cube (10 hardest by residual cost):**
1. The Legend of Kuruk // Avatar Kuruk (saga + transform + waterbend + copy)
2. The Rise of Sozin // Fire Lord Sozin (saga + transform + firebend + X)
3. Sephiroth, Fabled SOLDIER // Sephiroth, One-Winged Angel (transform-on-trigger, mass life-drain)
4. The Legend of Kyoshi // Avatar Kyoshi (saga + transform + earthbend + statics)
5. Hama, the Bloodbender (waterbend + X + statics + gy-cast restriction)
6. The Legend of Roku // Avatar Roku (saga + transform + firebend + tokens)
7. Earthbender Ascension (earthbend + counters + reflexive triggers)
8. Phoenix Fleet Airship (vehicle + tokens + statics + "spells you cast from graveyard")
9. The Goose Mother (X + food tokens + replacement + reflexive trigger)
10. Iroh, Tea Master (food statics + trigger-doubling-adjacent conditional triggers)

**TLA commons (5 hardest):** Path to Redemption (aura + sac-activated + token),
Badgermole (earthbend + counters-matter static), Watery Grasp (aura +
waterbend-activated), Curious Farm Animals (death trigger + sac-activated
destroy), Hog-Monkey (exhaust + counter-conditional combat trigger).

**FIN commons (5 hardest):** Summon: Fat Chocobo and Summon: Choco/Mog
(saga-creatures), Sage's Nouliths (job select + granted attack-trigger with
targets), Syncopate (X + unless-pays + exile-instead replacement),
Namazu Trader (attack trigger + reflexive sacrifice + surveil + treasure).

## Recommendation

**Intermediate milestone: TLA commons. FIN is dominated on every axis that
matters for the cube.**

1. **Direct card reuse.** 34 TLA commons are literally in the cube — 75 of
   540 weighted copies (14% of the cube is *registered for free* by the TLA
   commons milestone). FIN commons contribute **0 cube cards**. More broadly,
   56% of the cube (weighted) was printed in TLA/TLE, so TLA idioms
   (bending, Allies, Lessons-in-graveyard conditionals) recur throughout the
   cube's uncommons/rares.
2. **Capability coverage.** After the TLA-commons closure (34 caps, cost 58)
   the cube is 75% unique / 80% weighted expressible. FIN's closure costs
   more (37 caps, cost 65) and lands slightly lower (75% / 78%). TLA's
   set-only capabilities (the four bendings) are exactly the cube's named
   mechanics (59 unique cube cards, 15%); FIN's set-only work (job select,
   tiered) buys nothing downstream.
3. **Reusable machinery per card.** Both sets force the same evergreen core
   (targets, EOT modifiers, tokens, +1/+1 counters, trigger family, statics,
   activated costs). FIN additionally forces SAGA/VEHICLE/X/REPLACEMENT at
   common — genuinely reusable, but the cube needs those for its own rares
   anyway, and doing them against FIN cards produces trace tests for cards
   that will never be in the destination pool. TLA is the only candidate
   where *every* trace test is either an evergreen driver or a cube card.
4. **What TLA does not buy:** MULTIFACE (45 unique cube cards, the largest
   gap, in neither set's commons), KEYWORD_NEW at volume (ward/flash/
   hexproof — 27 cube cards, only ~2 FIN commons would have helped anyway),
   SAGA (14 cube cards incl. the 4 Avatar legends), REPLACEMENT, VEHICLE,
   X_COSTS. These form the post-TLA cube tail and must be scheduled
   explicitly (see milestone C).

**Is a set-commons milestone still right, vs pure "cube-lite at rung N"?**
Yes — as the *second* milestone. Pure cube-greedy is a bad early schedule:
the first three rungs unlock 1%/3%/5% of the cube (scattered singles, no
coherent playable pool, weak trace-test narrative). A commons set is a
coherent, draftable, self-contained pool where "done" is crisp, matching the
stage-11 driver-card philosophy. But the evergreen substrate should be built
first as its own milestone, because both ladders agree on it and it is where
training-signal risk (action-space growth) concentrates.

### Proposed milestone sequence

- **Milestone A — evergreen substrate** (cost ≈ 20): TARGET_FILTERS,
  EOT_MODIFIERS, LIFE_EFFECTS, TAP_UNTAP, DISCARD, MILL, SAC_EFFECTS,
  SCRY_SURVEIL, MODAL, DYNAMIC_VALUES, LAND_EXT, DEATH/ATTACK/CAST triggers,
  TOKENS(+artifact tokens), COUNTERS_P1P1, DESTROY_EXILE, ACTIVATED_COSTS.
  Exit: ~35-40% of TLA commons and ~10-15% of the cube expressible; every new
  Effect variant lands with CR-cited trace tests + observation encoding.
- **Milestone B — TLA commons complete** (adds cost ≈ 38): STATIC_CONTINUOUS
  (the big one), AURA, EQUIPMENT, TRIGGER_MISC/CONDITIONAL, SEARCH_LIBRARY,
  GRAVEYARD_PLAY, COST_MODIFIERS, CYCLING, COUNTERS_OTHER, KEYWORD_NEW(min),
  the four bendings, SPECIAL_HARD single (Lost Days). Exit = the milestone
  definition below; cube is ~80% weighted expressible as a side effect.
- **Milestone C — cube closure** (adds cost ≈ 18+): MULTIFACE, SAGA,
  REPLACEMENT, VEHICLE, X_COSTS, KEYWORD_NEW (full), SPECIAL_HARD long tail
  (the 10 hardest cards above are the acceptance list).

### Milestone definition

**"TLA commons complete"** = all 96 TLA commons (a) registered in a
`managym/src/cardsets/tla.rs` via the declarative DSL (no bespoke per-card
code paths), (b) covered by trace tests — at least one happy-path scenario
per card and one negative/interaction test per new rule family, CR-cited,
(c) observation-encodable — every capability used by the set has a structured
encoding in the observation space (stage-12 requirement), and (d) training
smoke test passes at the enlarged action space.

**"Cube complete"** = same bar over all 383 unique cube cards, plus the cube
list itself (with duplicates and 36x Fabled Passage) loadable as a draftable/
playable environment pool.

## Method notes / caveats

- Classification: regex-assisted multi-label over oracle text (reminder text
  stripped) + Scryfall keyword list, with manual overrides for ~10 cards;
  per-card noise estimated at 2-3%, immaterial to histogram/ladder shape.
- Costs are S/M/L judgment (1/2/4), not estimates from the codebase; the
  ladder ordering is robust to ±1 reweighting except among same-cost peers.
- "Expressible" means the current structs could represent the card faithfully
  for 2-player games with no planeswalkers; "any target" is treated as
  creature-or-player on that basis.
- Raw data + classifier: scratchpad `classify.py`, `analyze.py`,
  `{fin,tla}_commons.jsonl`, `cube_cards.jsonl` (session-local, regenerable
  from Scryfall/CubeCobra).
