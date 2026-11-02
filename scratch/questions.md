# T5 questions and assumptions

Accepted decisions and pending checks live in
[jack-heart/a7234f-tla-t5-permanent.md](jack-heart/a7234f-tla-t5-permanent.md).
Historical parent notes (D1, T2–T4, archive/CI work and MTG-214's Wave rename)
are evidence, not directions for T5. Complete pre-compression notes:
`bbb5101eb:scratch/questions.md`; older stacked notes:
`1e9fd2d7c:scratch/questions.md` (lines 5–472).

## Current implementation assumptions

- Surveil reuses LookAndSelect: select moves the inspected card to the
  graveyard, decline keeps it. Kept cards retain order; exact for surveil 1,
  the only amount authored. Supporting larger amounts needs ordering choices.
- Copy exceptions share `Modification` with continuous records. The protocol
  calls the field `copy_of` because Pydantic's `BaseModel.copy` occupies `copy`.
- Airship uses self-relative `CreatureWithNamesakes` because
  `check_static_condition` has no source. It reads private layer stages,
  never recursively reads `characteristics()` during layer computation.
- T4 owns `PlayerTurnHistory`, sacrifice bookkeeping and its column after
  attacked. Text uses singular/plural correctly; no retained fixture holds
  the old “1 permanents” / “1 cards” wording.
- The engine measured 314 supported definitions (257 TLA eligible plus 57
  others; four TLA and one ths unsupported). Earlier 239/266 notes were
  pre-sync or based on stale main counts, not current measurements.
- The permanent row's name/creature-types fields are Aura-set markers; the
  card row states layered characteristics. Ward is printed unless abilities
  are lost; no authored effect grants ward. Text and mana cost stay printed
  until a modification changes them, then join `Characteristics`.

## Reversible choice awaiting a separate product decision

Since 1a/2a the projection's `base_power` / `base_toughness` contain layered
card-row P/T on the battlefield. No UI component reads them for printed P/T,
and no other Python-visible source exposes printed P/T. Keep and document
that value for T5; a printed-P/T accessor or field removal requires a follow-up
protocol decision by Jack Heart. This is not a T5 blocker.

The choice-navigation re-record (`889948346`) preserves recorded offer ids
when labels repeat, keeping original state digests and offer order. Gate must
publish that local change and verify the Playwright cases at the new head.
Shipping remains stopped by steer f39fef34; `/verify` is not a pre-merge check
because that skill finishes delivered Tasks.
