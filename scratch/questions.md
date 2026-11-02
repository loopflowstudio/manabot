# MTG-214 Wave rename assumptions — 2026-10-29

- `.lf/wave/rules` → `.lf/wave/engine`: lf's `engine` Wave owns the Rules
  chapter Projects and the same Linear initiative. Added the engine Wave id
  (`74d9efbc…`) to its GOAL frontmatter, matching training's explicit id.
- Academy and eval have empty lf goals and no checked-in files; none created.
  Only training gets `config.yaml` (`workflow: iterate`), as directed.
- The top-level `wave/` tree (old `pm:` charters, numbered plans) is historical
  and cited by frozen experiments and code comments; left unrenamed, as are
  `.lf/chapters/` records and wave body prose.
- Live doc pointers moved to `.lf/wave/{engine,training}`. ARCHITECTURE now
  assigns evaluation to Eval (previously Intelligence); reversible.
- Eval split (Jack Heart's steer, 2026-10-29): MTG-212 and MTG-188 memory sections
  moved verbatim to `.lf/wave/eval/MEMORY.md` with cross-pointers; MTG-175 moved
  to eval in lf. No eval GOAL.md until Jack sets an objective.

## Concept review (2026-10-29)

- Resolved in `4dc25ad61`: `wave/README.md` opens with a note marking `wave/`
  as the historical tree and naming the current Waves (academy, engine, eval,
  training). Other `wave/intelligence` / `wave/rules` references cite existing
  historical files and stay.
- Not blocking, outside this Task: training's objective ends with "demonstrate
  a repeatable challenge to the User under a protocol fixed before scoring",
  which reads like evaluation. Moving it to an eval objective is Jack's choice;
  this Task changes no objective.
- Second pass after implement: no further change; no live `.lf/wave/{intelligence,rules}`
  pointers remain.
# Questions and assumptions

## MTG-235 — accepted scope and remaining dependency

Jack Heart's 2026-11-02 concept-review decisions amend PR #412 in place:
object verbs use selectors, AffectedSpec folds into ObjectSelector, value twins
merge, and Chosen applies to any object verb. “After that, ship it” remains the
delivery direction. The current plan is `scratch/mtg-235-effect-vocabulary-verbs.md`.

- MTG-225 (T6) has no PR on this base. Chapter → CountersReach and T6's new
  effects remain an explicit integration dependency, not a completed acceptance.
- `Target` holds `TargetSpec`, rather than a clause index, because implicit
  target clauses are derived from it; the existing clause cursor picks the slot.
- `Grantor` names the exact granting Equipment; `linked` names exile links.
- `Move` has `under`, `with_counter`, `linked`; no current card needs the
  sketch's transforming or timed move. Library moves shuffle; positional
  placement remains a separate rule.
- `frame.moved` and `frame.this_way` are distinct: Avatar Destiny must retain
  its mill result across moving itself. Chosen Move/Sacrifice with no candidates
  clears moved before a subsequent “if you do”.
- A player and a stack ability are not card ObjectRefs. Player-affecting verbs
  use `resolve_affected` over the same selector; CounterAbility reads its retained
  clause target. Static layers keep private selector evaluation to avoid recursive
  characteristics computation.
- Optional sacrifice is supported over Chosen. Optional fixed-object sacrifice
  has no current card; no speculative payment path is retained.
- Heuristic polarity treats battlefield-target moves away as harmful; graveyard
  return keeps its prior neutral treatment. Current authored choice prefixes must
  preserve the same decisions and offer counts after the evidence refresh.
- Maturin is in the `play` extra. The extension is installed atomically from
  the cp312 wheel, never overwritten while mapped.

## Historical context

Complete inherited TLA T2/T3/T4, decklist, archive, characteristics, CI timing,
and original MTG-235 notes remain at
`d1bda612103909ead5321163d889818911e8a36e:scratch/questions.md`.
They are evidence, not new work. In particular: frozen advice failures remain
blockers rather than fixtures to regenerate; source-sensitive parity receipts
must be rebuilt after final Rust edits; current w17 evidence is refreshed in
place, whereas earlier worlds remain frozen.
