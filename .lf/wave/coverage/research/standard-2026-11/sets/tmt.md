# TMT: casting transactions, event provenance, and Classes

Reviewed 2026-11-04: all 195 identities and their full rules text in the cached
TMT inventory. All are single-faced; Class and Saga layouts remain single
cards. The [ledger](tmt.jsonl) retains every Oracle ID, exact name, face name,
primary batch, dependencies, and a card-specific obligation. An exact-name
scan of current definitions found only the five basic lands; this is a planning
audit, not a compiled coverage measurement. Reprints belong in their existing
first-printing modules.

The [official release notes](https://magic.wizards.com/en/news/feature/teenage-mutant-ninja-turtles-release-notes)
confirm the important product boundary: TMT is Standard material, whereas TMC
and PZA have separate legality rules. They also resolve sneak's casting window
and inherited attack destination, cumulative Class levels, Mutagen activation
timing, and the ordered multiplayer choices on Turtles in Time. Current Oracle
text supplies the per-card obligations below; proposals are architectural
inferences, not claims that these mechanisms already work.

## Shared implementation order

The primary assignments count responsibility, not implementation difficulty:
core composition 38; sneak 25; costs 23; object movement/attachments 20;
selection 20; history 19; mana 13; layers 10; combat 7; Classes 5;
permissions 5; scheduling 4; copies 3; replacements 3. Most cards cross batches.

1. **Finish transaction and event foundations before sneak.** `SpellOnStack`
   currently has special-purpose flashback/kicked flags, `cast_from`, and X,
   but no general paid-cost or mana-spent receipt. Sneak needs a legal alternate
   cast route, a returned unblocked attacker as an irrevocable cost, and captured
   defender identity. Its creature enters attacking that defender; an instant
   creating attacking tokens has ordinary destination choices instead. Retain
   spell provenance through resolution for Leonardo, Karai, Turncoat Kunoichi,
   and tokens/copies without manufacturing an attack-declaration event. Ninja
   Teen grants sneak from graveyard and adds a finality counter: zone permission,
   alternative cost, and entry modification are separate components.

2. **Generalize costs and payment predicates.** Reuse hand/graveyard ability
   zones, target specifications, selectors, and existing payment continuations.
   Extend those authorities for return costs, sacrificing three Rats, sacrificing
   as an equip cost, spending counters, and combinations of tap, sacrifice, and
   mana. Stomped by the Foot needs nonmana kicker, Grounded for Life a
   target-dependent discount, and Triceraton Commander one X charged twice.
   Purple Dragon Punks
   and Turtle Lair expose why `ManaUse` must become a typed purpose predicate
   rather than another subtype-specific case. Omni-Cheese Pizza pays mana before
   producing mana; Mona Lisa computes production from power. Tokka & Rahzar
   compares mana actually spent with stack mana value, including X and alternate
   costs. Spell legality, payment enumeration, and payment execution must agree.

3. **Make event results reusable.** Disappear needs controlled-permanent
   departures, including tokens and the source itself, with intervening-if
   evaluation where printed. Turtle Van needs exact crew participant identities;
   April O'Neil, Hacktivist needs turn history by spell type; Shredder's Technique
   needs what was actually destroyed. Add these facts to shared history/results,
   not permanent booleans for individual cards. Reuse the existing nth-resolution
   and once-per-turn mechanisms where applicable. Generalize resolution values
   for toughness, counter counts, rounded-up halves, and departed-source power;
   `SourcePower`'s current live-or-zero behavior is not a safe LKI authority.

4. **Add Classes through existing ability and layer authorities.** The five
   Classes have sequential, sorcery-speed, stack-using level activations,
   cumulative abilities, and sometimes transition triggers. Level is per-object
   rules state, not a counter quantity or station threshold. Leader's Talent
   grants combat capabilities; Does Machines animates artifacts; Cool but Rude
   adds discard triggers and a tutor/random-discard transition. One level
   progression primitive plus conditional grants serves them all. Grant queries
   must use `Game::characteristics`, not a second activated-ability list.

5. **Complete selection and casting continuations.** Turtles Forever combines
   differently named legendary cards from library/outside-game ownership,
   revelation, opponent choice, and movement into hand/library. Reuse candidate
   ownership from Learn, but replace Learn-specific assumptions and do not alias
   `OpponentSplitsRevealed`, whose destinations differ. North Wind Avatar also
   needs general outside-game selection. Mikey & Don grants private top-card
   inspection and restricted play permission; Krang & Shredder grants an optional
   free cast during resolution from linked exile. Turtles in Time makes players
   commit optional shuffle choices in order before execution. Cool but Rude and
   Raph & Mikey require replayable randomness, not policy-selected outcomes.

6. **Compose layered grants, copies, and replacements.** Mondo Gecko's colors
   and color-sensitive hexproof belong in layered characteristics and targeting.
   Does Machines must preserve unrelated artifact abilities. Splinter, Radical
   Rat modifies trigger multiplicity rather than duplicating events.
   Michelangelo, Weirdness to 11 modifies counter placement through replacement
   ordering; Putrid Pals and Bot Bashing Time likewise require event replacement,
   not ad hoc post-processing. The Cloning of Shredder needs linked, durable copy
   information. Old Hob's delayed destruction must remain destruction, including
   indestructible interaction, rather than being approximated by sacrifice.

## Closure, acceptance, and cost

Token definitions must cover Food, Mutagen, red 2/2 Mutants, white 1/1 Ninja
Turtle Spirits, white 2/2 Dinosaur Soldiers, black 1/1 Ninjas, black 1/1 Rats,
black 1/1 flying Insect Warriors, colorless 1/1 Robot artifact creatures, and
dynamic copies. Mutagen combines mana, tapping, sacrifice, targeting, and
sorcery timing; Food is a subtype, not shorthand for a token implementation.

Acceptance witnesses should cross these boundaries:

- Counter a sneak spell after its attacker was returned; preserve payment and
  reject another cast outside the window. Change or remove the captured defender.
- Grant graveyard sneak, change controllers, and blink an object: distinguish
  player turn history, object identity, and surviving permission records.
- Crew Turtle Van with zero-power and excess contributors. Current crew offering
  excludes the former and stops at the threshold; both shortcuts lose choices.
- Resolve a Class transition alongside ability loss and counter replacement;
  level advancement must not become counter placement.
- Resolve a multi-target spell with all targets illegal versus choosing zero
  legal targets; subsequent token creation must follow the correct resolution.
- Suspend at a replacement or opponent choice after committed payment, then
  replay/fork without repeated payment, random draws, or duplicate triggers.

Turtles Forever's qualified-search failure-to-find behavior needs an explicit
rules check before coding; its release note only clarifies outside-game access.
The configured outside-game pool must also be explicit, with private candidates
and public reveals. These are not permission to omit the card indefinitely.

Use staged complete choices instead of Cartesian products of selections, mana
payments, or library permutations. Index departure/crew/cast facts by event
kind; invalidate layered caches through their actual dependencies. Compare both
step and search budgets on representative controlled fixtures, preserving
digests when semantics are unchanged. This review ran no games or benchmarks
and claims no measured speedup. New state must reach native observations,
Python tensors, text, learning tokens, and shared human decisions without
changing frozen observation versions.
