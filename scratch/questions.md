# Rules audit assumptions — 2026-10-26

- The five-area audit is complete at its recorded evidence depth; the accepted
  2026-10-26 resolution repair is implemented as an isolated draft. Unsupported mechanics
  can be marked missing only after inspecting the relevant representation and
  authority; an untested or unresolved obligation stays unknown or partial.
- Directed synthetic fixtures expose precise obligations, not proof that the
  selected authored decks can reach them. No new Task or identity migration is
  inferred from a synthetic probe alone.
- MTG-195 owns trigger ordering; MTG-196 owns replacement ordering. Their
  mechanisms remain outside this repair; no external messages are selected.
- Inherited MTG-184 questions are preserved at
  `324db65f2:scratch/questions.md`; they do not direct this audit.
- Resolution repair (2026-10-26): use clause-specific eligibility slots in the
  existing EffectFrame; preserve invalid positions rather than compacting targets.
  Untargeted trigger context is used only when no targets were chosen. Existing
  stack CardId identity remains unchanged; no broader stack ABI is inferred.
- Integration (2026-10-27): Jack Heart requested stacking on MTG-195 and retried
  after its delivery. PR386 has merged, so sync main incorporates it directly.
  Preserve trigger-ordering context/action joins and resolution eligibility together.
- Combined draft contracts: world `w15-resolution-draft`, hash 10, policy-input
  and capture/text v4, embedded decision schema 7. V2/v3 keep 32 relation columns;
  v4 adds eligibility at column 32. Validity remains column 31. Parent world w14
  and v3 evidence remain frozen. Final world admission and new receipts are pending.
