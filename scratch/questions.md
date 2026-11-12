# MTG-307 decisions

Inherited MTG-286 notes (parent PR #469) removed; read them at
`5a7ab52ea:scratch/questions.md`.

- Assumption (reversible): the trigger player actor is a small `Actor` enum
  (`You`, `Opponent`, `AnyPlayer`), not `PlayerSelector`, which resolves at
  resolution and carries `TriggeringPlayer`, `Target` and `Owner`.
- Assumption (reversible): search, scry, surveil and the bends share one
  `Performs { who, any_of }` event.
- Decision (implement, reversible): learning tokens and definition digests
  move. Tokens are serde-derived names, so stability would need a custom
  serializer (a compatibility shim); input shapes are unchanged, w17 is the
  open world and the content digest already moves with every card. Frozen
  checkpoints and evidence keep their revisions.
- Out of scope, follow-up Task: `DelayedEvent` (`YouCastSpell`,
  `EndStep { your_turn }`, `Upkeep`) is a second vocabulary for the same
  events. Kept on purpose: `YouCastThis`, `SpellTargetsOnlyThis` and
  `YourAttackingCreatureTriggers` (actor fixed by the rules or a
  single-consumer meta-event). `Attacks`, `Sacrificed` and `PutsCounters` are
  folded into event-over-actor form in this change.
- Assumption (reversible): Beastmaster Ascension's subject is
  `TriggerSubject::you_control(creature)` (printed "a creature you control"); a `This`
  subject's `Attacks` uses `Per::Object`, which cannot differ from `Batch`.
