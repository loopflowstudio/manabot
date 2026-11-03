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

# T5 questions and assumptions

# Questions and assumptions

## MTG-237 — headless, 2026-11-02

Jack Heart's steer permits publication and demo, without shipping approval.
Headless decisions below are delegated judgment, not human statements.

- Interpret the committed baseline as a reference commit measured A/B on one
  runner: identical code varied 6.8k–10.3k steps/s across CPU models. Retain
  10%; two EPYC 7763 runs agreed within 2.5%. Jack Heart's PR review decides
  this interpretation and the native-throughput interpretation below.
- Use native `Env::step` random-legal play, 512 seeds, skip_trivial on. The
  retired rul-9 Python semantic-Command runner was deleted in #343; its 588.5
  steps/s is not comparable. Every ordered search score is part of correctness,
  including losing scores. Rules/deck/benchmark changes reaching this workload
  require a deliberate reference move when digests change.
- A CI job owns its VM; pin `ubuntu-24.04`, report the CPU, and compare only
  within it. Candidate-first order preserves recovery when the reference no
  longer builds; retain it absent evidence of bias exceeding tolerance.
- `measurement.json` identifies the candidate; `record` ignores the nested
  reference run and preserves tolerance. Reference-build recovery is tested
  with injected binaries/Cargo failure/metadata and a copy of the upload path;
  only comparison-failure transport has hosted proof.
- Keep merged main `51fbacbbf`'s optimization. Run 41555845634 demonstrated
  30.26% less worst-reply time, 28.36% less total search time and 36.65% less
  time per step, with matching digests/counts. Production adoption and its
  publication at `fb590b3f9`, PR text correction and pass-4 demo verification
  are complete. Run 41558633938 passed Engine Budget against the adopted
  reference; exact artifact identity, comparison replay and temporary adoption
  passed. Delegated judgment advances demo, without shipping approval; final
  main-integration/Tests Result and post-merge verification remain. The
  [design](mtg-237-engine-performance-budget.md)
  and [demo evidence](engine-budget-demo.md) own details.

## Reversible choice awaiting a separate product decision

Since 1a/2a the projection's `base_power` / `base_toughness` contain layered
card-row P/T on the battlefield. No UI component reads them for printed P/T,
and no other Python-visible source exposes printed P/T. Keep and document
that value for T5; a printed-P/T accessor or field removal requires a follow-up
protocol decision by Jack Heart. This is not a T5 blocker.

The choice-navigation re-record (`889948346`) preserves recorded offer ids
when labels repeat, keeping original state digests and offer order. It is
published; Academy UI passed at `bbb5101eb` (CI 41539137476). Gate subsequently
confirmed all required CI at `1de85f7aa` (run 41541743395), plus local copy-fact
parity coverage and its CI selection. The latter remain uncommitted; see the
design's gate results for commands and evidence boundaries.
Shipping remains stopped by steer f39fef34; `/verify` is not a pre-merge check
because that skill finishes delivered Tasks.

## MTG-216 (decklist disclosure) — 2026-10-30, implementation (headless)

Jack Heart accepted one open/closed option, default open (supersedes the
kickoff's four-way assumption). Reversible choices made while implementing:

- No player-row "known" flag: an open main deck is never empty, so all-zero
  `decklists` rows mean closed. The player width is untouched (67 after the
  sync with main's T4 v5 widening) and the native width-binding rule is
  unchanged.
- No separate Python `DecklistHidden` class: refusals are `AgentError`
  ("decklist closed: …"). `Game::determinize` and
  `HiddenHandConstraints::for_viewer` assert; `Env` entry points check first.
- Search refusal at match build lists observation-only kinds
  (`checkpoint`, `heuristic`, `random`, `scripted_greedy`); every other kind is
  refused in closed matches. Academy refuses its search villain in closed games.
- Decklist pooling is semantic-pack only, like `known_hand`; generic-model
  policies ignore the `decklists` buffer. Adding `decklist_projection` changes
  the v5 semantic `ActorCritic` parameters, so v5 semantic checkpoints saved
  before this change fail strict load (open world; none frozen).
- Disclosure is not part of `world_binding` or deck identity: a checkpoint can
  play both modes, and in a closed game it sees zero opponent rows.
- `Deck.player()` (training deck pools) seats an open decklist; it takes no
  disclosure argument. Closed games come from `MatchConfig.decklists` or
  Academy only.
- `canonical_replay`/`study` privacy validators do not check decklist maps:
  frames carry no disclosure, and the engine projection is the authority.


## Sync onto main 230b2f563 — 2026-11-02

The selector amendment and upstream decklist disclosure both remain. Current
w17 parity receipts and choice-navigation frames are regenerated against their
combined source; neither side’s stale digest is treated as merged evidence.
The authored command tape and retained choice prefixes must remain unchanged.

Inherited evaluation, MTG-17x/18x and card-evaluation notes (none select T4
work) are at `ee42a8685:scratch/questions.md`, above "TLA direct-Rust decisions".


## Sync onto main 230b2f563 — 2026-11-02

The design link above is absent in this checkout (removed with added scratch
files at 2c6cdae5f); the retained assumptions supply the resolution context.
Preserve T5's CopyView and layered battlefield P/T alongside main's decklist
disclosure fields. Navigation prefixes, offer IDs and state digests stay fixed;
recompute derived frame hashes for the combined projections. Reconcile the
current authored-match parity receipt's source manifest with the merged source,
then verify the unchanged command tape and behavioral evidence.

The first focused run passed all navigation cases but found stale parity viewer
hashes. The producer confirms that only `viewer_observation_sha256` and
`viewer_state_sha256` differ: combined layered P/T and decklist projections
require new hashes. State witnesses, command tape, transitions, presentation,
and stale-object proof remain identical. Refreshed those derived hashes from
the producer; no behavioral assertion was relaxed.

Focused verification: navigation plus authored-match parity initially reported
9 passed / 1 stale receipt; after correcting only the derived viewer hashes,
the failed parity test passed (1 passed). The advice schema equals its Pydantic
producer. Native extension rebuilt and replaced atomically. No push or landing.

## Sync onto main a8abf01b1 — 2026-11-02

Preserve MTG-235 selectors and main's T5 token copies, surveil, layered
characteristics and stable continuation codes. Migrate incoming old selector
names to ObjectSelector; ChooseObject keeps code 4 and Surveil adds code 22.
Current w17 evidence must be regenerated from the combined implementation,
retaining authored commands and choice prefixes. Historical evidence stays fixed.
The T5 shipping stop quoted above is historical context for its own delivery.

Inherited MTG-236 decisions (including deferred interning and quiet-machine
absolute targets) are at `51fbacbbf:scratch/questions.md`. T5/decklist assumptions,
the unresolved printed-P/T decision and sync evidence are at
`24cfc0a8b:scratch/questions.md`; older Wave/evaluation context is at
`460732f81:scratch/questions.md`. They select no MTG-237 work. Full preceding
MTG-237 assumptions, including the temporary T6 measurement ref (now deleted),
are at `4b5e9c0e4:scratch/questions.md`. Pre-curation uncommitted notes are
preserved under `.cache/engine-budget-demo/pre-compress-4b5e9c0e4/`.

## Sync onto main a1a02a76f — 2026-11-02

Preserve MTG-235's selector/value/Chosen model while retaining incoming typed
state accessors and cache revisions. Transfer mana-pool writes to the unified
AddMana arm and pending-decision writes to ChooseObject and Surveil. Removed
fused effects stay removed; generic Move and Untap already own their behavior.
Both branches' notes remain evidence, not new scope. Regenerate the two current
w17 parity receipts from the combined source; historical receipts remain fixed.
The waiting sync caller owns the push and broader gate validation.

The duplicate local untap helper is removed in favor of main's shared
cache-aware implementation. The rebuilt cp312 extension was installed
atomically. Regenerated receipts differ from HEAD only in source manifests:
142 authored commands, 69 public commitments, zero gaps, unchanged behavior.
