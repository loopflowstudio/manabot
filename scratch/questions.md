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

## MTG-236 (derived state revisions) — 2026-11-02, headless

- `Revisions` has `characteristics`, `mana_pools` and `triggers`; the brief's
  `offers` counter is omitted because no derivation caches offers today
  (`current_action_space` is recomputed per decision), and a counter nothing
  reads would be a setup nothing uses. Add it with its first consumer.
- Mana pools get their own counter rather than joining `characteristics`:
  one static condition reads them (`UnspentManaAtLeast`, The Lion-Turtle),
  and pool writes happen on every mana tap and step boundary. `StaticIndex`
  records whether a source reads the pools, and the views follow
  `mana_pools` only then.
- Only the write sites the probe showed hot in playouts moved to typed
  accessors (events, taps, damage, life, pools, step flags, stack, trigger
  queues, incarnations, zone moves). Every other write still advances all
  counters through `DerefMut`; converting more is safe whenever a profile
  asks for it, because debug builds recompute every cached value on read.
- `TrackedState::move_card` advances `characteristics` for moves touching
  the battlefield or a graveyard and `triggers` for the battlefield only;
  hand, library, stack and exile moves advance nothing. Graveyard triggers
  (`InGraveyard`) are matched by scanning graveyards, outside the index.
- The bench machine carried load averages of 25–40 from other sessions
  during every run, so absolute times here are not comparable to the
  brief's (4.97 s / 151.8 s); deltas against the same-session baseline
  and identical score digests are the evidence.

## MTG-236 decision pass 1 — 2026-11-02, headless

- Item 5 (interned subtypes) is deferred, not built: after items 1-4 the
  release profile puts all of `Game::characteristics` at ~2% inclusive and
  its string building under 1%, so interning cannot move the acceptance
  numbers. Reversible: it stays listed in the PR body as deferred evidence
  for Jack Heart, and a separate Task covers it or mana payment search if
  the quiet-machine bench misses acceptance.
- The quiet-machine `bench_search` thresholds are owned by the gate/CI
  owner; this machine carried load 25-70 throughout (35 at this pass). The
  headless equivalent is the 31% CPU-second drop with identical score
  digests, projecting ~3.4 s / ~105 s against the 4.34 s / 128.7 s bound.
