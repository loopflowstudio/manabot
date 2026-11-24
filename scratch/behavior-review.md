# Behavior review: MTG-420 Batch 1 at `bf5db745`

**Verdict: Batch 1's working phase is satisfied. Advance to Batch 2's foundation (plan model items 1–2). Nothing here needs another implement pass.**

The iteration direction handed to this step repeats the review from `66556cde`, which predates the build. Batch 1 has since been built (`e6f6ef27`) and compressed (`47dec0aa`, `2955a3d8`). This pass traced the built code against each of the six defects that review named.

## The six earlier defects, as built

| # | Earlier defect | Now | Evidence |
| --- | --- | --- | --- |
| 1 | Departure triggers matched the printed graveyard card | `subject_matches_moved_object` reads `layers::matches_departure` over `ObjectLki.battlefield`, sharing `CharacteristicView` with `matches_view` | `departure_returns_use_last_creature_types…` (a returned Angel Ogre or Demon Vessel dying again does not trigger); `departure_manifested_land_and_mooned_creature…` (a manifested Forest returns, a mooned Ogre does not) |
| 2 | `EnchantedCreature` needed a live Aura | It falls back to `ObjectLki.attached_to` | `departure_angelic_destiny_and_enchanted_creature_die_together` |
| 3 | `Triggering` resolved live permanents only | It follows `departed_permanent_lki` → `departure_arrivals` (graveyard/exile successors, both meld cards). Permanent slots are append-only (`zones.rs:869`), so a recycled id can't misdirect it | Valkyrie's Call, Vessel and Fake Your Own Death cases; a card that left the graveyard and came back stays put |
| 4 | Leaving deleted `WithSource` links | Only `UntilSourceLeaves` ends on departure. `WithSource` is pruned in `release_resolved_forms`, and `ObjectSelector::ExiledWithSource` reads the links | Two Hoarding Dragons each recover only their own artifact; after a bounce and replay, the Dragon can't reach the old link |
| 5 | `UntilSourceLeaves` implied `min: 0` | The inference is gone. `exile_until_source_leaves(spec, min)` is explicit | Banishing Light and Stasis Snare offer no Decline and are removed with no target. Prayer offers Decline, gains life with zero targets and loses its whole resolution when its sole target is illegal |
| 6 | No N counters, no frozen count | `EntryCounters`, plus `EffectValue::DeathCountersMinus` frozen to a literal when the delayed trigger is created; `EntersWithCounters.if_cast` | Familiar: cast gives 8, death at 4 returns with 3, a blink gives 0 and no return, and a card that left the graveyard and came back does not return |

The fixtures commit through `support::play::command`, which forks the game, compares the fork's commit and checks that the spent Command is rejected without changing state. This covers Prayer's target, Kykar's mode, the Dragon's mode and search, and the Familiar's end-step pass.

## This pass

- I extended `departure_angelic_destiny_and_enchanted_creature_die_together` to cover two cases. The Aura moving out of the graveyard and back before its trigger resolves stays in the graveyard. The Aura leaving alone puts no trigger on the stack. Both close the plan row's "only the eligible Aura appearance is recovered".
- Ran in debug (dev profile): `cargo test -p managym-cards --test cards fdn::` (26 passed) and `observability::` (77 passed). The new `ObjectSelector::ExiledWithSource`, `LinkedExile` lifetimes and the Batch 1 effect fields are listed in `observability.rs`. `Effect::ExileLinked` keeps its serialized name.
- FDN ledger rows for the 11 Batch 1 cards name their source and directed check.

## Not covered (carried, not blocking)

- Undying Malice or Valkyrie's Call with a blink in response. Carry it to Batch 2's integrated cases as the plan says.
- Viewer projections, text and tokens for Batch 1 cards on a constructed position, and replay of their Commands from a recorded root. These belong to Batch 4's consumer pass along with the existing open checks (Lost Days, Horde/Driver/Fire Crystal, named Coil seed file, manifest sampling).
- `requirement.another` still excludes only a live source (Fiendish Panda), and `resolution_targets` does not recheck `another`/`within_source_bound`. Both are Batch 2 model item 2.
- Battlefield `move_object` in `cast_permission::exile_linked` (Batch 4, model item 5).

## Owned by gate

Full debug nextest, clippy and fmt, `tests/model/test_pinned_pilot.py` after rebuilding the extension (Jailer's program gained an explicit `TargetClause`; declare it with `scripts/pinned_pilot.py record --reason` if it fails), Python package gates and Engine Budget.
