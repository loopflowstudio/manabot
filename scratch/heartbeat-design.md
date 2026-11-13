# MTG-297 review decisions (2026-11-12)

Contract: `tle-production-design.md`. The published review Artifact is
"Heartbeat of Spring Design". Earlier detail is preserved at
`2af8bc650:scratch/heartbeat-design.md`.

## Review answers, verbatim

- A, on narrowing to production only: "no include all here"
- B, on reusing `ManaChoice` plus a `producer` field with no new v10 block:
  "yeah if it works thats fine"
- C, on firing bonuses when mana enters the pool rather than at tap time:
  "Follow the mtg rules exactly"

The scope is Heartbeat of Spring (CHK), Animal Attendant, Toph, Greatest
Earthbender, and Uncle's Musings (TLE), with production **and** spending choices.
Badgermole Cub migrates; The Great Henge and Tarnished Citadel keep their
source-owned suffix behavior. Cards need Scryfall ids, directed behavior,
observability and tokens; `docs/engine.md` tracks corpus admission.

Encoding may reuse `ManaChoice` and current fact rows; no new v10 block unless
the existing joins cannot express the choice and a person reviews that need.
The old proposal's new version, fact kinds and v1–v9 rejection are superseded.

## Rules findings used by the review

- CR 106.12: tapping for mana means activating a mana ability with its own {T}
  cost. Waterbend and Food taps do not qualify.
- CR 106.12a: "is tapped for mana" triggers when the ability resolves and
  produces mana. No output means no trigger; actual types include colorless.
- CR 605.1b / 605.4a: triggered mana abilities resolve immediately after the
  producing ability, without the stack. Heartbeat's tapping player receives
  and chooses the bonus regardless of its controller.
- CR 601.2h/card rulings: Toph, converge and Attendant use actual mana spent to
  cast, additional costs included, excluding fees for other mana abilities.

Concept review required trigger sources to be selected at production, after
costs/suffix: removing Heartbeat before production prevents its bonus, while a
sacrificed producing land still qualifies through captured tap information.
It also required bonus choices to name Heartbeat separately from the tapped
land in frames and decision text. Both are built; the predicate-snapshot
limitation is in `questions.md`. The review called this a "kind-14 frame";
`ManaChoice` actually uses the kind-6 obligation frame, decision code 3.

## Coordination

The direction was: "you may also want to coordinate with mtg-242 which is also touching mana payment".
MTG-242 reworked restriction classes into `ManaRestriction` / `SpendingContext`.

Steer 41eb2776 (2026-11-12), verbatim: "Do the production half now. Build spending
(pool classes, receipts, Animal Attendant, Toph, Uncle's Musings) on MTG-242's
ManaRestriction/SpendingContext and proposed restricted-mana fact row; don't
rewrite pool storage in parallel. If spending must wait for MTG-242 to reach
main, finish production and stop there."

MTG-242 PR #496 merged 2026-11-12T21:37Z; main merged here as 824753626.
Spending is built on those types. Spending choices remain required, not
approved out of scope; see the active plan.
