Managym could offer Gnarled Professor as castable and then reject the cast because paying Prophetic Prism's activation cost consumed the only green source. Mana-ability offer probes and execution now use the same completion-safe payment plan, preserving colors needed by the announced spell.

## What changes

- Search legal generic and hybrid payment remainders for intermediate activation costs, retaining Lesson mana restrictions and existing combat-pool spending order.
- Choose a minimal set of fixed mana sources that leaves at least one legal output for the announced payment. Source tapping and pool reconciliation each retain one implementation.
- Cover Forest-first and Forest-last ordering, required blue mana already pooled, and the original seed 2165384 with its retained 34-action prefix. The historical reproduction remains bound to its original authored pack.

## Checks

- Current-main delivery checkout: 18 Temur/payment debug tests, three Elemental-world tests (including the original retained prefix and all 100 ordered pair/mirror games), formatting and all-target/all-feature Clippy passed. The cp312 native extension was rebuilt. Current-world Python admission and learnability checks are running before merge.
- Original investigation checkout: all 292 rules tests, six pool tests, 100 ordered native matchup smoke games, 31 arena/monitoring tests and full learnability including reward-driven PPO passed. The first learnability attempt hit its collection deadline; the unchanged-budget serial retry passed in 338 seconds. These are plumbing and learning-path checks, not evidence of adaptive learning gains.
- The original full local Rust invocation was interrupted in lengthy search-state tests after its rules suite passed. The full debug matrix remains owned by CI.

This delivery includes only the mana repair and regressions. The unfinished curriculum/evaluation work and the paused experiment allocation remain in their existing checkout. General failure capture and recovery is tracked separately as MTG-176.
