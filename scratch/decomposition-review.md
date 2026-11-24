# Decomposition review: MTG-420 Batch 2B at `3e4cab0cf5`

**Result: Batch 2B is not met yet. Trigger targets share one authority, but activated-ability targets have a second one with a rules defect (Diaochan). Fix it in 2B′ (`plan.md`) before 2C.**

## Consumers traced

- **Fiendish Panda** (`fdn.rs:43`). The death trigger uses `TargetClause { another: true, GraveyardCardYouOwnWithinSourcePower }`. Placement (`triggers.rs:404`), mode availability (`triggers.rs:479`), choice (`triggers.rs:292`), retargeting (`retarget.rs:52`) and resolution (`resolution.rs:525`) all call `target_fits_clause` (`resolution.rs:2610`) with the trigger's `AbilitySource`. `target_is_source` matches the live permanent or the exact `departure_arrivals`, and the power bound reads `battlefield_power`, so the value is live or comes from LKI. The test is `fdn.rs:1706`.
- **Driver of the Dead**. Same clause path, with the predicate and no bound. The same fixture shows that only Llanowar Elves qualifies.
- **Retargeting and resolution** read one `stack_target_requirements` (`24ac46fe41`).

## Deletion cut

`fits_trigger_clause`, `within_source_bound` and `within_target_mana_budget` are gone. `rg` finds no callers. Distinctness and the X budget live together in `can_add_clause_target`, which trigger choice, retargeting and spell casting (`play.rs:828`) share. No card-specific dispatch was found in the flow files.

## Findings

1. **Rules defect: activated targets bypass the clause.** `activated_ability_targets` (`play.rs:1496`) filters `legal_targets_for_spec` and drops the source permanent from every clause. Compare `target_fits_clause`:
   - Diaochan, Artful Beauty (`ptk.rs:53`): “destroy target creature of an opponent's choice”. The opponent can never choose Diaochan. That hides a strategic choice, and CR 115 allows it because the text says no “another”.
   - Trusty Boomerang's granted “Tap target creature” cannot tap the equipped creature. Koma's “tap target permanent” mode cannot tap Koma.
   - `another` and `GraveyardCardYouOwnWithinSourcePower` are ignored at offer time. Resolution then applies `target_fits_clause` to the same activation, so offer and resolution use different rules.
   - The blanket exclusion covers for two TLA definitions whose text says “another” and lack the flag: Deserter's Disciple (`tla.rs:2793`) and North Pole Patrol (`tla.rs:6844`).
   The plan said the exclusion was kept "for TLA compatibility". It is the shortcut for those two flags. Fixing it removes a second authority.
2. **Minor: mode availability checks one candidate per required clause.** `legal_trigger_modes` checks only `.next().is_some()`. A clause with `min > 1` and fewer distinct legal targets is offered, and placement then drops the trigger (`continue 'placement`). This is pre-existing and no current card is affected. Recorded in `questions.md`.
3. **Shape, carried:** the power bound is a `TargetSpec` twin (`GraveyardCardYouOwnWithinSourcePower`, from TLA `4c8324a595`) and not a clause field. `target_fits_clause` is the only reader that applies the bound, so the twin causes no harm once 2B′ routes activations through it. Leave it; the serialized tokens predate this Task.

## Evidence

No tests ran in this review. Finding 1 comes from reading the code, and the Diaochan fixture (`cards/tests/cards/ptk.rs:20`) checks only the opponent choosing the controller's Ogre. The 2B witnesses recorded in `plan.md` (67 focused tests at `24ac46fe41`) still apply: the merge `361823be59` did not touch `flow/` targeting. 2B′ changes activated offers, so rerun the `ptk::`, `tla::` and `khm::` filters with `fdn::` after it. Gate owns the full suites and the pinned pilot.
