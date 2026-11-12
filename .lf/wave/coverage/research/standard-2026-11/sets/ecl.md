# Lorwyn Eclipsed (ECL)

Reviewed 2026-11-04: all **273 identities and 280 faces**, including seven
transforming pairs, in the retained Scryfall input. The [ledger](ecl.jsonl)
accounts for each identity once and names every face. `full-text-reviewed-plan`
means full Oracle text was inspected; it does **not** claim implemented behavior
or an exhaustive rules proof. A definition-name scan found only the five basics
already registered; reuse their first-printing definitions. ECL survives the
working rotation boundary. Do not incorporate ECC Commander cards or the SPG
bonus sheet merely because they accompany this release.

The [official release notes](https://magic.wizards.com/en/news/feature/lorwyn-eclipsed-release-notes)
resolve blight, behold, vivid, transforming cards, and returning mechanics.
Uncertainty remains in exact interaction sequencing for individual complex
cards; the obligations below are implementation acceptance questions, not a
substitute for directed rules tests.

## Dependency order and shared authorities

The ledger's primary assignments are ownership boundaries, not independent
per-card executors. Dependencies cross batches deliberately.

| Primary authority | Cards |
|---|---:|
| Existing composition | 54 |
| Casting and activation costs | 45 |
| Counters, blight and persist | 35 |
| Resumable selection and instruction results | 30 |
| Types and type choices | 24 |
| Layered colors and vivid | 17 |
| Continuous characteristics | 16 |
| Event history and triggers | 10 |
| Replacement and prohibition | 10 |
| Casting permissions | 10 |
| Stack copies and modal composition | 9 |
| Transforming faces | 6 |
| Object lifetime and duration | 5 |
| Planeswalkers and emblems | 2 |

**1. Repair shared characteristic queries before tribal and vivid definitions.**
Native `CardType::Kindred` already exists. Changeling needs all-zone
characteristic-defining semantics, not a battlefield-only keyword predicate or
an array expanded from whichever creature types currently occur in decks.
Chosen-type effects need the real creature-type vocabulary. A Kindred instant
can satisfy a subtype-only behold requirement without being a creature card.
Layer-4 type replacement and ability removal must interact correctly with
changeling; granting all types is not interchangeable with granting changeling.

`Characteristics` in `flow/statics.rs` lacks colors, while `colors_among` reads
`Card::colors()`. Move effective colors into the shared layered authority before
Tam's all-color changes, vivid and Shimmerwilds Growth. Extend the existing
`StaticBuff.per_color_among` idea into reusable color-set/value queries rather
than copying it into card handlers. Bloom Tender needs the set itself;
Prismabasher needs separate trigger-time targeting and resolution-time values.
Colorless contributes no color.

**2. Extend counter events and payment continuations together.** Existing sparse
counter storage is suitable, but its vocabulary lacks negative, charge, dream
and several keyword counters required here. Append kinds and expose them through
facts/tokens rather than adding permanent fields. Blight as an effect and blight
as a cost share selection and counter placement, but not timing or optionality.
A cost continuation must retain its obligation across replacement decisions.
Persist requires death-event last-known information, including negative counters
before state-based cancellation; wither must preserve damage-event semantics
while changing damage's consequences. Proliferate should operate over selected
players/permanents and their actual counter kinds, not a creature-only shortcut.
These authorities also multiply the value of counter work in other sets.

**3. Unify casting choices, paid facts and restricted resources.** Behold spans
mandatory, optional, alternative and multi-object costs; it needs distinct
objects, hidden-hand reveal lifetimes and a durable paid-cost result. Kindle's
flashback combines mana and behold. Pyrrhic Strike's cost changes modal capacity.
Convoke, Springleaf Drum and conspire share resource selection but require
different payment and event semantics. Evoke is an alternative cost followed by
an ordinary sacrificial entry trigger. Incarnation riders read actual colored
mana spent on the total cost, including increases; spell copies did not pay it.

Delete the closed `SPEND_SUBTYPES = ["Lesson", "Shrine", "Ally"]` restriction
in `state/mana.rs` in favor of typed spend predicates: selected creature types
and mana-value restrictions cannot silently disappear from a string mask.
Keep total-cost calculation, payment permission and paid provenance distinct.
In particular, sacrificing Eirdu during mana generation can remove convoke
before payment. These are shared casting mechanisms, not ECL-only flags.

**4. Finish instruction results and bounded choices before complex spells.**
`Effect::Surveil` documents count-one-only ordering; Unwelcome Sprite and Twilight
Diviner require complete surveil-two ordering. Consolidate that shortcut in the
existing continuation. Commands need ordered chosen modes with per-clause target
legality. Glen Elendra's Answer needs a stack selection and the number actually
countered, excluding uncounterable spells. Sanar's sequential color selections
must preserve revealed-card identity and remaining choices. Spry and Mighty
needs a multi-object choice and one captured power difference; verify insufficient
objects and zero difference explicitly. These extend selectors, values and
instruction results already used by ordinary `Move`/`If` compositions.

**5. Add definitions on established timing and lifetime machinery.** Soulbright
Seeker should reuse `OnNthResolutionThisTurn`. The six ordinary transforming
pairs reuse `BeginningOfYourFirstMainPhase` and `TransformSource` from TLA; Zuko, Conflicted already exercises the first
main-phase trigger. Preserve face-specific characteristics and distinguish
entry-versus-transform events. Oko belongs to the planeswalker batch: transforming
must preserve loyalty and the permanent's once-per-turn activation history.
Use existing effect records for duration, linked exile and copy exceptions.
Mornsong Aria's draw/life-gain prohibitions and Mirrormind Crown's optional first
token-event replacement require general event interception, not post-event undo.

## Refactor triggers, closure and acceptance

Adept Watershaper grants other tapped creatures indestructible; Illusion Spinners
has hexproof while untapped. These invalidate the documented assumption in
`flow/game.rs` that tapping cannot change characteristics. Update revision
ownership when adding the reader, retaining the debug recomputation oracle.
Adding a new static ability while leaving that optimization intact is a
correctness bug even if isolated card tests pass.

Created-object closure includes matching two-color Kithkin, Merfolk, Faerie,
Goblin, Elf and Worm tokens; Treefolk, Elk, changelings, Treasure, Mutavault land
tokens, arbitrary copied tokens, and Oko's chosen-type emblem. Reuse a token only
when every characteristic matches. Emblems and created objects need identity,
controller, observation and learning-token coverage after their source leaves.

Focused witnesses should cover: tapping Watershaper's protected creature through
an actual cost; lethal blight with both counter signs and persist; interrupted
counter-cost payment; evoke paid with taxed/hybrid mana versus a copied spell;
changeling under type replacement and ability loss; Kindred hand behold; vivid
colors changing between trigger and resolution; private surveil ordering through
reconnect/fork; and Oko transforming without resetting activation limits.

No games or benchmarks were run. Cache color/type summaries under correct
revisions, index trigger/history queries, and avoid catalog scans or eager
subset enumeration at each decision. Preserve every strategic choice with
resumable selection. Extend bounded ordinary-step and search fixtures together;
compare equivalent workloads on one runner and retain cumulative milestone cost.
