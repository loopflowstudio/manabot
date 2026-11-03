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

## MTG-235 — accepted scope and corrected delivery order

Jack Heart's 2026-11-02 concept-review decisions amend PR #412 in place:
object verbs use selectors, AffectedSpec folds into ObjectSelector, value twins
merge, and Chosen applies to any object verb. “After that, ship it” remains the
delivery direction. The current plan is `scratch/mtg-235-effect-vocabulary-verbs.md`.

- Jack Heart corrected delivery order in comment
  `2e1c66f1-d00a-4319-b5f4-c631bcf608c9`: land MTG-235 before T6. T6 adapts
  Chapter → CountersReach and its new effects in its own work. Earlier
  dependency statements in these historical notes are superseded.
- Headless demo uses bounded native scenarios and the actual Academy parity
  CLIs, with prior browser evidence retained. This is delegated judgment,
  not new human observation; no new UI design is under review.
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

Notes inherited from merged stacked parents are retained in git: D1 through
MTG-234 at `1e9fd2d7c:scratch/questions.md`; T5 (plan, demo and operator choices,
merged to main as `a8abf01b1`) at `1bc8b33fb:scratch/`. None selects work here.

# T6 (MTG-225) operator choices — 2026-11-02, headless

- Base: developed on main `4c8324a59` (T3+T4) while T5 #405 was mid-sync;
  T5's synced head is merged when it publishes, so the stack is kept.
- A double-faced card is one definition function; `back_face` holds the back
  face, which the pack registers as its own row right after the front.
  `back_face` serializes as the back face's name (a definition reference in
  tokens and digests), so a back face carries no `key`/`scryfall_id` of its
  own; pool rows mark it "not a deck entry".
- Back-face colors follow Scryfall's face data: no color indicator except
  Fire Lord Sozin (black), so Avatar Kuruk/Kyoshi/Roku/Yangchen and the Aang
  back faces are colorless here. Flag for review against the printed cards.
- Entering with a Saga's lore counter emits `CountersPut` (so chapter I
  triggers, CR 714.2b/122.6); other entry counters still emit nothing.
- Extra turns are observed as a per-player count (player column
  `PLAYER_EXTRA_TURNS_INDEX`, player width +1); the resume order is not a
  column.
- A transformed permanent's mana value is its front face's (CR 712.8e),
  read by `Card::mana_value_cost` for predicates and targets.
- Airbending a spell exiles it from the stack with the {2} recast grant
  (Aang, Swift Savior's "creature or spell").
- Copies of double-faced permanents follow CR 707.8a (post-2021): the token
  has both faces and is transformed if the original is, so it can transform.
- Foggy Swamp Visions: "your next end step" uses the next end step, the
  same for a sorcery. Its heuristic polarity is pinned "harmful" (exile).
- Additional combat phases are a stack on `TurnState` observed as a count on
  the active player's row; a pending phase's attacker limit is visible only
  once it applies (attack offers).
- Yangchen II's "you may have target opponent draw three" is a resolution
  `Modal` (draw both / nothing) inside its target clause, not a new variant.
- The Rise of Sozin II: the card name is a resolution `Modal`, one mode per
  deck-eligible name in the world (front faces, no tokens or fixtures). The
  search exiles as many as it finds up to four (graveyard, hand, library)
  rather than offering to find fewer.
- Fire Lord Sozin: targets are chosen as the trigger goes on the stack and X
  is then their total mana value, paid or declined at resolution — the
  reverse of "pay {X}, when you do choose targets", with the same outcomes.
- Bending events: "you waterbend" fires once per paid waterbend cost (spell,
  activation or resolution payment), "you firebend" once per resolved
  until-end-of-combat mana (firebending's only source), "you earthbend" and
  "you airbend" once per resolved instruction. The turn history keeps the four
  kinds as bits (`PlayerTurnHistory.bent`, four player columns).
- Koh: "has all activated and triggered abilities of the last chosen card" is
  a `HasAbilitiesOfSource` continuous-effect record whose source is the chosen
  definition (latest record wins). Inherited activated abilities follow the
  printed ones; an inherited activation resolves from the record while Koh
  remains, and inherited leaves-the-battlefield triggers do not fire as Koh
  leaves. "You may exile it" is a resolution `Modal`.
- 2026-11-02 (CR check against docs/rules/MagicCompRules-20260227.pdf): copies
  follow 707.8a, so a copy of a transformed permanent is a double-faced token
  that can transform. This differs from the proposal in t6-direction.md ("cannot
  transform"); question for Jack Heart if that was meant as a house rule.
- 701.27f is approximated by comparing faces: an ability on the stack transforms
  only if the face it came from is still up. A permanent flipped twice while the
  ability waits would still transform. A departed back face's triggers read the
  front face; no current TLA back face has a leaves trigger on itself.
- Exactly-X targets ("X target creature cards") are `max_x` with a nonzero `min`;
  "up to X" keeps `min` 0.

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

## T6 sync onto main 22e4aab71 — 2026-11-03

Resolve the existing owned merge only; the waiting sync caller owns the push.
Preserve T6's faces, Sagas, copy behavior, extra turns/combats and one catalog
while adopting MTG-235's object selectors and value-bearing effects. T6's
transformed flicker becomes two Move instructions with a `transformed` entry
flag; bulk untap/destroy use Each selectors. Keep spell airbending and all four
bending events, including earthbend zero, in the unified resolvers. The existing
Chapter trigger remains for this merge; the broader Chapter-to-CountersReach
amendment and MTG-238 integration remain the owning T6 workflow's work.

Focused proof covers the existing T6 card scenarios and effect/trigger
observability. Broader gate checks, native Python rebuild, and current-world
receipt refresh remain with the caller; no retained evidence is rewritten here.
