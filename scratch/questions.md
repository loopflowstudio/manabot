# Rules audit assumptions — 2026-10-26

- This pass is a draft audit of the five named rule areas. Unsupported mechanics
  can be marked missing only after inspecting the relevant representation and
  authority; an untested or unresolved obligation stays unknown or partial.
- Directed synthetic fixtures expose precise obligations, not proof that the
  selected authored decks can reach them. No new Task or identity migration is
  inferred from a synthetic probe alone.
- MTG-195 owns trigger ordering; MTG-196 owns replacement ordering. No mechanism
  changes or external messages are part of this implementation step.
- Inherited MTG-184 questions are preserved at
  `324db65f2:scratch/questions.md`; they do not direct this audit.
- Resolution repair (2026-10-26): use clause-specific eligibility slots in the
  existing EffectFrame; preserve invalid positions rather than compacting targets.
  Untargeted trigger context is used only when no targets were chosen. Existing
  stack CardId identity remains unchanged; no broader stack ABI is inferred.
- MTG-195 still has uncommitted trigger-order/w14 work. Final world identity and
  fresh integrated receipts require its supported handoff; local implementation
  checks do not certify either frozen world.
- The serialized frame changes canonical hashing. Use hash 10 (MTG-195 holds
  9) and the provisional `w15-resolution-draft` identity to prevent frozen-world
  admission while integration is unavailable. Final world assignment is pending.
