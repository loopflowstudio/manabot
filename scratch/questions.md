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

Inherited MTG-236 decisions (including deferred interning and quiet-machine
absolute targets) are at `51fbacbbf:scratch/questions.md`. T5/decklist assumptions,
the unresolved printed-P/T decision and sync evidence are at
`24cfc0a8b:scratch/questions.md`; older Wave/evaluation context is at
`460732f81:scratch/questions.md`. They select no MTG-237 work. Full preceding
MTG-237 assumptions, including the temporary T6 measurement ref (now deleted),
are at `4b5e9c0e4:scratch/questions.md`. Pre-curation uncommitted notes are
preserved under `.cache/engine-budget-demo/pre-compress-4b5e9c0e4/`.
