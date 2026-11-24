# Behavior review: MTG-420 at `94f0a91f`

**Verdict: the working phase is not done. Return to implement for Batch 1's lead slice (plan items 1–3).**

The commits since compress `9e297f0c` changed only `scratch/` files (realign `dbbc24a8`, concept-review `94f0a91f`). The engine code is the same code compress tested, so its focused nextest run (42 cases) and the 13 Python checks still stand as implementation evidence. I did not rerun them. Every Batch 1 row is still unimplemented.

## Done work: no regressions found

The transform, meld, face-down, departed Clone, copied-trigger and named-Coil slices listed under "Done" in the plan keep their directed fixtures and their native fork/restore checks. The plan's "Open checks on finished work" still applies as written:

- Lost Days viewer privacy and replay.
- Horde, Driver and Fire Crystal viewer projections and replay from a recorded root.
- The named Coil seed-to-position file.
- Manifest constrained sampling and paired searches.
- The audit of immediate battlefield move callers.

None of these blocks Batch 1. All must close before the Batch 4 representation review.

## Defects confirmed in code (each blocks a Batch 1 row)

1. **A departure trigger matches the printed graveyard card.**
   - *Where:* `flow/triggers.rs` `subject_matches_moved_object`, around line 2142. The predicate runs against `self.state[moved_object]` (the new graveyard object) and `object_power(moved_object)`. Only `animated_creature` is restored from LKI.
   - *Reproduction:* A Gray Ogre returned as an Angel by Valkyrie's Call dies again, and the "nontoken Angel" exclusion reads the printed Ogre.
   - *Same gap elsewhere:* a manifested Forest dies as a noncreature, and a Mooned creature dies as a creature.
   - *Fix:* add effective `types`/`subtypes`/`colors`/`name`/`mana_value`/`attached_to` to `ObjectLki` and give it one shared `CharacteristicView` matcher (plan item 1).
2. **`EnchantedCreature` requires a live Aura.** Same function, around line 2166: it reads `permanent_of(source_object)` → `attached_to`. When Angelic Destiny dies alongside its creature, nothing fires. *Fix:* read `ObjectLki.attached_to` for a departed Aura.
3. **`ObjectSelector::Triggering` resolves only live permanents.**
   - *Where:* `flow/resolution.rs` around line 2301 reads `context_target` as `Target::Permanent`.
   - *Effect:* Valkyrie's Call, Infernal Vessel and Fake Your Own Death can't reach the dead creature's graveyard card.
   - *Fix:* follow `departed_permanent_lki` → `zones.successor`, as `This` does, and replace `MayExileTriggeringWithSource`'s own lookup.
4. **`handle_leaves_battlefield` deletes `WithSource` links.**
   - *Where:* `flow/zones.rs` around line 1076 partitions out every link whose `source == departing_object`, whether or not it returns.
   - *Effect:* Hoarding Dragon loses its linked artifact before its dies trigger is put on the stack.
   - *Fix:* end only `UntilSourceLeaves` links. Prune `WithSource` links on liveness alongside `retain_forms`, and add `ObjectSelector::ExiledWithSource`.
5. **Implicit `min: 0` for any `UntilSourceLeaves` move.**
   - *Where:* `state/ability.rs` `effect_target_requirements`, around line 2199.
   - *Effect:* Banishing Light and Stasis Snare would be authored as "up to one". The silent optional choice changes the legal action space.
   - *Fix:* use an explicit `TargetClause` with `min: 1` for those two and `min: 0` for Prayer of Binding. Then delete the inference.
6. **`Effect::Move { with_counter: Option<CounterKind> }` can't express N counters, and a delayed trigger can't freeze a count.**
   - *Where:* `state/ability.rs` around line 949.
   - *Effect:* Nine-Lives Familiar can't return with `k − 1`.
   - *Fix:* plan item 3, keeping serialization for an absent counter or a count of 1.

Batch 2 is blocked by the split cast provenance on `Permanent` (`was_cast`/`cast_x`/`cast_spent`, `permanent.rs:190`), which has no record of a kick. Batch 4 is blocked by the battlefield `move_object` in `cast_permission::exile_linked` (around line 268). Their batches own these, so neither is a Batch 1 blocker.

## What the next implement pass must show

- Directed `fdn.rs` fixtures for each defect above, constructed before the card that needs it:
  - a second death of a returned Angel/Demon
  - a manifested Forest dying
  - simultaneous Aura and creature death
  - two Hoarding Dragons
  - Banishing Light with zero legal targets offers no "choose nothing" option, while Prayer does
  - Familiar's count frozen at death
- A forked native snapshot at each new pending choice (Kykar mode, Prayer target, Familiar return), with the stale Command rejected.
- An `observability.rs` entry for each new `ObjectSelector` and `Effect` field, and learning tokens unchanged for existing cards (`ExileLinked` keeps its serialized name).
- Rows updated in `fdn.jsonl`.

The gate step owns the suites: full debug nextest, clippy/fmt, Python package gates and Engine Budget.
