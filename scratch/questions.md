# MTG-238 assumptions and unresolved boundaries — 2026-11-02

[The plan](mtg-238-catalog-scale-types.md) owns accepted scope, implementation,
evidence and remaining validation. Jack Heart approved keyword capacity option
1a and MTG-238-first delivery; no product choice blocks this batch. PR #414 stays
against main; T6 integrates afterward. On 2026-11-03 Jack Heart authorized
shipping once the engine-speed batch finishes validation, superseding the
earlier review-only/no-auto-merge boundary.

- KeywordBits is private ordinal storage so controlled word-boundary fixtures
  exercise production operations without declaring fictional mechanics. Storage
  grows at compile time and preserves Copy; serde still emits enum variants.
- KeywordData/PyKeywords remain boolean observation contracts. Future non-evergreen
  keywords require fact rows and a coordinated Python API migration, not frozen
  column expansion. No such mechanic is included now.
- Source-relative names read layers 1–3 without the static index. Conditional type
  grants may be evaluated while that index is built; recursive animation reads
  are unsafe. Empty continuous-effect batches preserve derived-cache revisions
  but still allocate a timestamp, so they are not state/hash no-ops.
- T5 copy exceptions share Modification; Python uses `copy_of` because Pydantic
  defines `copy`. Battlefield `base_power`/`base_toughness` hold layered P/T.
  Printed-P/T access or removal remains an unresolved protocol choice for Jack
  Heart, outside this Task. Surveil 1 uses LookAndSelect; larger amounts need
  explicit ordering choices.
- Decklist disclosure remains native: closed opponents have zero decklist rows,
  search refuses them, and disclosure is not deck/world identity. No observation
  width or checkpoint-default change is included.
- Fixture refreshes preserve recorded offer ids when labels repeat; selecting by
  label alone can change the physical copy, digest and permanent order. Preserve
  frozen receipts; only mutable w17 evidence is rebound. Logs without completion
  summaries establish no broad-suite pass; current-head CI still needs assessment.

Historical inherited notes are retained at `3ed5b5baa:scratch/questions.md`
(including `bbb5101eb`, `1e9fd2d7c`, `4c8324a59` references). Recovery history is at
`cfa3aa182:scratch/questions.md`; MTG-237's assumptions remain at
`a1a02a76f:scratch/questions.md`. Uncommitted pre-compression notes are preserved
in `.runs/mtg-238-compress/questions-before.md`. Engine memory now records the
accepted 1a direction and its focused compatibility evidence.


Speed investigation and MTG-235 integration (2026-11-03): see the current plan's
engine-speed section and `.runs/mtg-238-speed/`. Main's accepted effect vocabulary
landed at 22e4aab71; its prior notes remain at that commit's scratch/questions.md.
The local inlining repair is measured separately from the required integration.
No measured speedup or passing budget is yet claimed for their combined source.
